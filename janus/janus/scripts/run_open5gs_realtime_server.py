#!/usr/bin/env python3
"""Real-time HTTP server for Open5GS log inference using the Janus stack."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Sequence

from janus.infer.inference import build_csv_row, load_model, run_inference
from janus.utils.paths import load_repo_config, project_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = project_root()
POD_APP_RE = re.compile(r"^(?P<pod>[^/\s]+)/(?P<app>[^/\s]+)\s+(?P<rest>.*)$")
TIMESTAMP_RE = re.compile(r"(?P<ts>\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


def _isoformat(value: datetime) -> str:
    """Return an ISO-8601 timestamp with millisecond precision."""

    return value.isoformat(timespec="milliseconds")


def _prepare_calibration_dir(path: Optional[str]) -> Optional[Path]:
    """Resolve an optional calibration directory argument."""

    if not path:
        return None
    resolved = resolve_path(path)
    if not resolved.exists():
        logger.warning("Calibration directory %s does not exist; ignoring.", resolved)
        return None
    if not resolved.is_dir():
        logger.warning("Calibration path %s is not a directory; ignoring.", resolved)
        return None
    return resolved


def _parse_log_line(line: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse a Kubernetes-prefixed log line into app, message and timestamp."""

    stripped = line.strip()
    if not stripped:
        return None
    prefix = POD_APP_RE.match(stripped)
    if not prefix:
        return None
    app = prefix.group("app").lower()
    message = prefix.group("rest")
    timestamp_match = TIMESTAMP_RE.search(message)
    timestamp = timestamp_match.group("ts") if timestamp_match else None
    return app, message, timestamp


@dataclass
class WorkItem:
    """Container describing a batch of logs to be scored."""

    app_name: str
    nf: str
    lines: List[str]
    first_received_at: datetime
    log_timestamp: Optional[str]


@dataclass
class LogBatch:
    """Accumulates logs for a single Open5GS application."""

    app_name: str
    lines: List[str] = field(default_factory=list)
    first_received_at: Optional[datetime] = None
    log_timestamp: Optional[str] = None

    def add_line(
        self, line: str, log_timestamp: Optional[str], received_at: datetime
    ) -> None:
        """Append a line to the batch updating timestamps."""

        if self.first_received_at is None:
            self.first_received_at = received_at
        if self.log_timestamp is None and log_timestamp:
            self.log_timestamp = log_timestamp
        self.lines.append(line)

    def to_work_item(self) -> Optional[WorkItem]:
        """Convert the batch into a :class:`WorkItem`."""

        if not self.lines or self.first_received_at is None:
            return None
        return WorkItem(
            app_name=self.app_name,
            nf=self.app_name.lower(),
            lines=list(self.lines),
            first_received_at=self.first_received_at,
            log_timestamp=self.log_timestamp,
        )


class CSVLogger:
    """Thread-safe CSV writer that augments inference rows with metadata."""

    def __init__(self, csv_path: Path, overwrite: bool):
        self._lock = threading.Lock()
        self._header: Optional[List[str]] = None
        self._app_index: Optional[int] = None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            if overwrite:
                csv_path.unlink()
            else:
                raise FileExistsError(
                    f"CSV file {csv_path} already exists. Use --overwrite to replace it."
                )
        self._handle = csv_path.open("w", newline="")
        self._writer = csv.writer(self._handle)

    def close(self) -> None:
        """Close the underlying CSV file handle."""

        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def write_row(
        self,
        header: Sequence[str],
        row: Sequence[str],
        app_name: str,
        log_timestamp: Optional[str],
        received_timestamp: str,
        inference_timestamp: str,
    ) -> None:
        """Persist a single inference row with application metadata."""

        with self._lock:
            row_data = list(row)
            if self._header is None:
                header_list = list(header)
                try:
                    nf_index = header_list.index("nf")
                except ValueError:
                    nf_index = len(header_list) - 1
                insert_index = nf_index + 1
                header_list.insert(insert_index, "app_name")
                header_list.extend(
                    [
                        "log_generated_timestamp",
                        "log_received_timestamp",
                        "inference_saved_timestamp",
                        "true_labels",
                    ]
                )
                self._header = header_list
                self._app_index = insert_index
                self._writer.writerow(self._header)
            assert self._header is not None
            assert self._app_index is not None
            row_data.insert(self._app_index, app_name)
            row_data.extend(
                [
                    log_timestamp or "",
                    received_timestamp,
                    inference_timestamp,
                    "[0]",
                ]
            )
            self._writer.writerow(row_data)
            self._handle.flush()


class BatchManager:
    """Coordinate per-application batching and dispatch to inference workers."""

    def __init__(
        self,
        work_queue: "queue.Queue[WorkItem]",
        flush_interval: float,
        batch_size: int,
    ) -> None:
        self._work_queue = work_queue
        self._flush_interval = max(flush_interval, 0.1)
        self._batch_size = max(batch_size, 1)
        self._batches: Dict[str, LogBatch] = {}
        self._condition = threading.Condition()
        self._stopped = False
        self._monitor = threading.Thread(target=self._run, daemon=True)
        self._monitor.start()

    def add_line(
        self,
        app_name: str,
        line: str,
        log_timestamp: Optional[str],
        received_at: datetime,
    ) -> None:
        """Add a log line to the queue for ``app_name``."""

        with self._condition:
            batch = self._batches.get(app_name)
            if batch is None:
                batch = LogBatch(app_name=app_name)
                self._batches[app_name] = batch
            batch.add_line(line, log_timestamp, received_at)
            if len(batch.lines) >= self._batch_size:
                self._dispatch_batch(app_name)
            self._condition.notify_all()

    def stop(self) -> None:
        """Flush remaining batches and stop the monitor thread."""

        with self._condition:
            for app_name in list(self._batches):
                self._dispatch_batch(app_name)
            self._stopped = True
            self._condition.notify_all()
        self._monitor.join()

    def _dispatch_batch(self, app_name: str) -> None:
        batch = self._batches.pop(app_name, None)
        if not batch:
            return
        work_item = batch.to_work_item()
        if not work_item:
            return
        logger.info(
            "Dispatching %d log lines for app=%s",
            len(work_item.lines),
            work_item.app_name,
        )
        self._work_queue.put(work_item)

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
                now = datetime.utcnow()
                ready_apps = [
                    app
                    for app, batch in list(self._batches.items())
                    if batch.first_received_at
                    and (now - batch.first_received_at).total_seconds()
                    >= self._flush_interval
                ]
                for app_name in ready_apps:
                    self._dispatch_batch(app_name)
                timeout = min(self._flush_interval, 0.5)
                self._condition.wait(timeout=timeout)


class InferenceWorker(threading.Thread):
    """Background worker running inference on queued log batches."""

    def __init__(
        self,
        work_queue: "queue.Queue[Optional[WorkItem]]",
        csv_logger: CSVLogger,
        model,
        tokenizer,
        device,
        pad_id: int,
        max_tokens: int,
        calibrate_dir: Optional[Path],
    ) -> None:
        super().__init__(daemon=True)
        self._queue = work_queue
        self._csv_logger = csv_logger
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._pad_id = pad_id
        self._max_tokens = max_tokens
        self._calibrate_dir = calibrate_dir
        self.start()

    def stop(self) -> None:
        """Signal the worker to exit once the queue is drained."""

        self._queue.put(None)
        self.join()

    def run(self) -> None:
        while True:
            work_item = self._queue.get()
            try:
                if work_item is None:
                    logger.info("Inference worker stopping")
                    return
                self._process(work_item)
            finally:
                self._queue.task_done()

    def _process(self, item: WorkItem) -> None:
        if not item.lines:
            logger.debug("Skipping empty batch for app=%s", item.app_name)
            return
        received_iso = _isoformat(item.first_received_at)
        tmp_path: Optional[Path] = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                suffix=f"_{item.app_name}.log",
            ) as tmp:
                tmp.write("\n".join(item.lines))
                tmp.write("\n")
                tmp.flush()
                tmp_path = Path(tmp.name)
            summary = run_inference(
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                pad_id=self._pad_id,
                log_path=tmp_path,
                nf=item.nf,
                max_tokens=self._max_tokens,
                csv_path=None,
                calibrate_dir=self._calibrate_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed for app=%s: %s", item.app_name, exc)
            return
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.warning("Failed to remove temporary file %s", tmp_path)
        metrics_mean = summary.get("metrics_mean")
        metrics_max = summary.get("metrics_max")
        stats = summary.get("reference_stats")
        header, row = build_csv_row(
            result=summary,
            log_path=tmp_path if tmp_path is not None else Path(""),
            nf=item.nf,
            num_lines=int(summary.get("num_lines", 0)),
            metrics_mean=metrics_mean,
            metrics_max=metrics_max,
            stats=stats,
        )
        inference_iso = _isoformat(datetime.utcnow())
        self._csv_logger.write_row(
            header=header,
            row=row,
            app_name=item.app_name,
            log_timestamp=item.log_timestamp,
            received_timestamp=received_iso,
            inference_timestamp=inference_iso,
        )
        logger.info(
            "Processed %d lines for app=%s | cls_prob_mean=%.4f",
            len(item.lines),
            item.app_name,
            summary.get("cls_prob_mean", float("nan")),
        )


class LogRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that accepts POSTed log batches."""

    server_version = "JanusRealtime/1.0"

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/logs":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return
        length_header = self.headers.get("Content-Length")
        if not length_header:
            self.send_error(HTTPStatus.LENGTH_REQUIRED, "Missing Content-Length header")
            return
        try:
            length = int(length_header)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header")
            return
        raw = self.rfile.read(length)
        try:
            payload = raw.decode("utf-8")
        except UnicodeDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Payload must be UTF-8 encoded text")
            return
        received_at = datetime.utcnow()
        processed = 0
        lines = payload.splitlines()
        ignored = 0
        for line in lines:
            parsed = _parse_log_line(line)
            if not parsed:
                ignored += 1
                continue
            app_name, message, log_timestamp = parsed
            self.server.batch_manager.add_line(  # type: ignore[attr-defined]
                app_name=app_name,
                line=message,
                log_timestamp=log_timestamp,
                received_at=received_at,
            )
            processed += 1
        response = {
            "status": "accepted",
            "received_lines": len(lines),
            "processed_lines": processed,
            "ignored_lines": ignored,
        }
        if ignored:
            logger.debug("Ignored %d lines that lacked pod/application prefix", ignored)
        body = json.dumps(response).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)


class LogServer(ThreadingHTTPServer):
    """Custom HTTP server exposing the log batching manager."""

    def __init__(self, server_address: tuple[str, int], batch_manager: BatchManager):
        self.batch_manager = batch_manager
        super().__init__(server_address, LogRequestHandler)


def _parse_args(default_seq_len: int) -> argparse.Namespace:
    """Parse CLI arguments for the real-time server."""

    parser = argparse.ArgumentParser(
        description="Run a real-time HTTP server for Open5GS log inference.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9000, help="Listen port")
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=5.0,
        help="Seconds to wait before processing a partial batch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum number of log lines per batch before immediate processing",
    )
    parser.add_argument(
        "--csv-log-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "janus_realtime_metrics.csv"),
        help="Destination CSV file for inference metrics",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default_seq_len,
        help="Maximum number of tokens per window. Defaults to infer.seq_len",
    )
    parser.add_argument(
        "--calibrate-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts/calibrate"),
        help="Optional directory containing calibration score distributions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the CSV file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point spinning up the real-time inference HTTP server."""

    config = load_repo_config()
    default_seq_len = config.get("infer", {}).get("seq_len", 1024)
    args = _parse_args(default_seq_len)

    csv_path = resolve_path(args.csv_log_file)
    calibrate_dir = _prepare_calibration_dir(args.calibrate_dir)

    logger.info("Loading Janus model for real-time inference")
    model, tokenizer, device, pad_id = load_model(config)

    csv_logger = CSVLogger(csv_path=csv_path, overwrite=args.overwrite)
    work_queue: "queue.Queue[Optional[WorkItem]]" = queue.Queue()
    batch_manager = BatchManager(
        work_queue=work_queue,
        flush_interval=args.flush_interval,
        batch_size=args.batch_size,
    )
    worker = InferenceWorker(
        work_queue=work_queue,
        csv_logger=csv_logger,
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_id=pad_id,
        max_tokens=args.max_tokens,
        calibrate_dir=calibrate_dir,
    )

    server = LogServer((args.host, args.port), batch_manager=batch_manager)
    logger.info("Listening for logs on http://%s:%d/logs", args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down server")
    finally:
        server.shutdown()
        server.server_close()
        batch_manager.stop()
        work_queue.join()
        worker.stop()
        csv_logger.close()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
