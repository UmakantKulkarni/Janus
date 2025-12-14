#!/usr/bin/env python3
"""
Dataset for Janus with smarter negative generation and detailed (optional) logging.
- 2000-char chunk cap
- HardNegativeGenerator chains several realistic ops
- CFGCorruptor first, then hard negatives, then simple fallback
- __len__ pre-scan with JSON cache
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import ijson
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer

from janus.tagger import LogTagger, FieldTag

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
SUPPORTED_FILE_TYPES = [".c", ".h", ".py", ".md", ".yaml", ".yml", ".json", ".sh", ".txt", ".conf", ".cfg", ".ini", ".build", ".in"]
MAX_CODE_CHARS = 2000
MAX_JSON_CHARS = 2000
_LEN_CACHE_FILENAME = ".janus_len_cache.json"

# Regex to strip ANSI escape sequences (e.g. "\x1b[32m") from log lines
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

# ------------------------------------------------------------------ #
# Helpers for reading data
# ------------------------------------------------------------------ #
def iter_spec_json(path: Path) -> Iterator[str]:
    with path.open("rb") as f:
        for item in ijson.items(f, "item"):
            if isinstance(item, dict) and "content" in item:
                yield item["content"]

def extract_functions(content: str, ext: str) -> List[str]:
    funcs: List[str] = []
    if ext in {".c", ".h"}:
        pattern = re.finditer(r"(\w[\w\s\*]+)\s+(\w+)\s*\([^)]*\)\s*\{", content)
        for m in pattern:
            start = m.start()
            brace = 1
            idx = m.end()
            while idx < len(content) and brace:
                brace += (content[idx] == "{") - (content[idx] == "}")
                idx += 1
            funcs.append(content[start:idx])
    elif ext == ".py":
        pattern = re.finditer(r"^(def\s+\w+\s*\(.*?\):)", content, re.M)
        for m in pattern:
            lines = content[m.start():].splitlines(True)
            indent = len(lines[0]) - len(lines[0].lstrip())
            body = lines[0]
            for ln in lines[1:]:
                if ln.startswith(" " * (indent + 1)):
                    body += ln
                else:
                    break
            funcs.append(body)
    return funcs

def chunk_content(text: str, max_chars: int) -> Iterator[str]:
    if len(text) <= max_chars:
        yield text
        return
    buf = ""
    for ln in text.splitlines(True):
        if len(buf) + len(ln) > max_chars:
            yield buf
            buf = ln
        else:
            buf += ln
    if buf:
        yield buf

def code_chunks(path: Path) -> Iterator[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pieces = extract_functions(text, path.suffix.lower())
    if not pieces:
        pieces = [text]
    for piece in pieces:
        for chunk in chunk_content(piece, MAX_CODE_CHARS):
            yield chunk


def nf_from_filename(path: Path) -> str:
    """Infer network function identifier from a log filename.

    The last ``-<word>`` segment before the ``.log`` extension is used. If no
    such segment exists, the stem of the filename is returned.
    """
    m = re.search(r"-([A-Za-z0-9]+)\.log$", path.name)
    return m.group(1) if m else path.stem

# ------------------------------------------------------------------ #
# CFG-based corruption
# ------------------------------------------------------------------ #
class CFGCorruptor:
    def __init__(self, map_path: Path, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        with map_path.open() as f:
            self.map = json.load(f)
        logger.info("Loaded CFG map with %d entries", len(self.map))

    def corrupt(self, token_ids: List[int]) -> Optional[List[int]]:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        for pos, neg in self.map.items():
            if pos in text:
                new_text = text.replace(pos, neg, 1)
                new_ids = self.tokenizer.encode(new_text, add_special_tokens=False)
                if len(new_ids) == len(token_ids):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("CFG corruption: '%s' -> '%s'", pos, neg)
                    return new_ids
        return None

# ------------------------------------------------------------------ #
# Hard Negative Generator
# ------------------------------------------------------------------ #
@dataclass
class Span:
    start: int
    end: int  # inclusive
    tag: int

def _spans_from_tags(tags: Sequence[int]) -> List[Span]:
    spans: List[Span] = []
    if not tags:
        return spans
    s = 0
    curr = tags[0]
    for i in range(1, len(tags)):
        if tags[i] != curr:
            spans.append(Span(s, i - 1, curr))
            s = i
            curr = tags[i]
    spans.append(Span(s, len(tags) - 1, curr))
    return spans

class HardNegativeGenerator:
    def __init__(self, tokenizer: PreTrainedTokenizer, pool_size: int = 8000):
        self.tok = tokenizer
        self.pool_size = pool_size
        # tag -> length -> list[token_id_list]
        self.span_pool: Dict[int, Dict[int, List[List[int]]]] = {}
        self.cause_choices = ["UNKNOWN", "MISSING_IE", "BAD_FORMAT", "204", "0"]
        self.state_choices = ["ACTIVE", "INACTIVE", "ERR", "NULL", "1", "0"]
        self.http_methods  = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        self.http_codes    = ["200", "201", "204", "400", "404", "409", "415", "500", "503"]

    def observe(self, token_ids: Sequence[int], tag_ids: Sequence[int]) -> None:
        for sp in _spans_from_tags(tag_ids):
            length = sp.end - sp.start + 1
            bucket = self.span_pool.setdefault(sp.tag, {}).setdefault(length, [])
            if len(bucket) < self.pool_size:
                bucket.append(list(token_ids[sp.start: sp.end + 1]))

    def _sample_span(self, tag: int, length: int) -> Optional[List[int]]:
        bucket = self.span_pool.get(tag, {}).get(length)
        if not bucket:
            return None
        return random.choice(bucket)

    def corrupt(self, token_ids: List[int], tag_ids: List[int], prob: float, max_ops: int = 3) -> Tuple[List[int], bool]:
        if random.random() > prob:
            return token_ids, False

        ops = [self._id_swap, self._time_warp, self._step_warp, self._http_mismatch, self._cause_state_flip, self._random_span_noise]
        random.shuffle(ops)

        changed = False
        out = token_ids[:]
        applied = []

        for _ in range(max_ops):
            for op in ops:
                trial = op(out[:], tag_ids)
                if trial is not None and len(trial) == len(out):
                    out = trial
                    applied.append(op.__name__)
                    changed = True
                    break

        if logger.isEnabledFor(logging.DEBUG) and changed:
            pass
            #logger.debug("HardNeg ops applied: %s", applied)
        return out, changed

    # ---- individual ops ----
    def _choose_spans(self, tag_ids: Sequence[int], wanted: Sequence[int]) -> List[Span]:
        return [sp for sp in _spans_from_tags(tag_ids) if sp.tag in wanted]

    def _replace_span(self, token_ids: List[int], sp: Span, new_tokens: Sequence[int]) -> bool:
        if len(new_tokens) != (sp.end - sp.start + 1):
            return False
        token_ids[sp.start:sp.end + 1] = new_tokens
        return True

    def _id_swap(self, token_ids: List[int], tag_ids: Sequence[int]) -> Optional[List[int]]:
        id_tags = [
            int(FieldTag.IMSI), int(FieldTag.SUPI), int(FieldTag.SUCI),
            int(FieldTag.GUTI), int(FieldTag.TEID), int(FieldTag.FSEID),
            int(FieldTag.PDRID), int(FieldTag.FARID), int(FieldTag.QERID),
            int(FieldTag.PDU_SESS_ID)
        ]
        spans = self._choose_spans(tag_ids, id_tags)
        random.shuffle(spans)
        for sp in spans:
            repl = self._sample_span(sp.tag, sp.end - sp.start + 1)
            if repl and self._replace_span(token_ids, sp, repl):
                return token_ids
        return None

    def _time_warp(self, token_ids: List[int], tag_ids: Sequence[int]) -> Optional[List[int]]:
        t_tags = [int(FieldTag.TIMESTAMP_ISO), int(FieldTag.TIMESTAMP_SHORT)]
        spans = self._choose_spans(tag_ids, t_tags)
        if len(spans) < 2:
            return None
        a, b = random.sample(spans, 2)
        seg_a = token_ids[a.start:a.end + 1]
        seg_b = token_ids[b.start:b.end + 1]
        if len(seg_a) != len(seg_b):
            return None
        token_ids[a.start:a.end + 1], token_ids[b.start:b.end + 1] = seg_b, seg_a
        return token_ids

    def _step_warp(self, token_ids: List[int], tag_ids: Sequence[int]) -> Optional[List[int]]:
        spans = self._choose_spans(tag_ids, [int(FieldTag.CALLFLOW_STEP)])
        if not spans:
            return None
        if len(spans) > 1 and random.random() < 0.5:
            a, b = random.sample(spans, 2)
            seg_a = token_ids[a.start:a.end + 1]
            seg_b = token_ids[b.start:b.end + 1]
            if len(seg_a) == len(seg_b):
                token_ids[a.start:a.end + 1], token_ids[b.start:b.end + 1] = seg_b, seg_a
                return token_ids
        src, dst = spans[0], spans[-1]
        seg = token_ids[src.start:src.end + 1]
        if len(seg) == (dst.end - dst.start + 1):
            token_ids[dst.start:dst.end + 1] = seg
            return token_ids
        return None

    def _http_mismatch(self, token_ids: List[int], tag_ids: Sequence[int]) -> Optional[List[int]]:
        changed = False
        spans_m = self._choose_spans(tag_ids, [int(FieldTag.HTTP_METHOD)])
        spans_c = self._choose_spans(tag_ids, [int(FieldTag.HTTP_CODE)])

        if spans_m:
            sp = random.choice(spans_m)
            new = random.choice(self.http_methods)
            tid = self.tok.encode(new, add_special_tokens=False)
            if self._replace_span(token_ids, sp, tid):
                changed = True

        if spans_c and random.random() < 0.5:
            sp = random.choice(spans_c)
            new = random.choice(self.http_codes)
            tid = self.tok.encode(new, add_special_tokens=False)
            if self._replace_span(token_ids, sp, tid):
                changed = True

        return token_ids if changed else None

    def _cause_state_flip(self, token_ids: List[int], tag_ids: Sequence[int]) -> Optional[List[int]]:
        spans = self._choose_spans(tag_ids, [int(FieldTag.CAUSE_CODE), int(FieldTag.STATE)])
        if not spans:
            return None
        sp = random.choice(spans)
        pool = self.cause_choices if sp.tag == int(FieldTag.CAUSE_CODE) else self.state_choices
        tid = self.tok.encode(random.choice(pool), add_special_tokens=False)
        if self._replace_span(token_ids, sp, tid):
            return token_ids
        return None

    def _random_span_noise(self, token_ids: List[int], tag_ids: Sequence[int]) -> Optional[List[int]]:
        spans = _spans_from_tags(tag_ids)
        if not spans:
            return None
        sp = random.choice(spans)
        for i in range(sp.start, sp.end + 1):
            token_ids[i] = random.randint(0, self.tok.vocab_size - 1)
        return token_ids

# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #
class JanusDataset(IterableDataset):
    def __init__(
        self,
        files: Optional[List[Path]] | None = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        *,
        root: Optional[Path] = None,
        include_specs: bool = False,
        include_code: bool = False,
        seq_len: int = 2048,
        corruption_prob: float = 0.2,
        cfg_map_path: Optional[Path] = None,
        hard_neg_prob: float = 0.35,
        span_pool_size: int = 8000,
        estimate_len: bool = True,
        len_cache_path: Optional[Path] = None,
        exclude_nfs: Optional[Sequence[str]] = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.tokenizer = tokenizer
        self.tagger = LogTagger(tokenizer)

        self.exclude_nfs = {nf.lower() for nf in (exclude_nfs or [])}

        if files is None:
            if root is None:
                raise ValueError("Either files or root must be provided")
            files = self._discover_files(root, include_specs, include_code)
        self.files = [
            fp
            for fp in (files or [])
            if self._extract_network_function(fp.name).lower() not in self.exclude_nfs
        ]

        self.seq_len = seq_len
        self.corruption_prob = corruption_prob
        self.hard_neg_prob = hard_neg_prob

        self.cfg_corruptor: Optional[CFGCorruptor] = None
        if cfg_map_path and cfg_map_path.exists():
            try:
                self.cfg_corruptor = CFGCorruptor(cfg_map_path, tokenizer)
            except Exception as exc:
                logger.warning("Failed to load CFG map: %s", exc)

        self.neg_gen = HardNegativeGenerator(tokenizer, pool_size=span_pool_size)

        self._len_cache: Optional[int] = None
        if estimate_len:
            if len_cache_path is None and root is not None:
                len_cache_path = root / _LEN_CACHE_FILENAME
            self._len_cache = self._load_or_compute_len(len_cache_path)

        logger.info("Dataset initialized with %d files", len(self.files))

    def set_corruption_prob(self, p: float) -> None:
        self.corruption_prob = max(0.0, min(1.0, p))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("corruption_prob set to %.4f", self.corruption_prob)

    def __len__(self) -> int:
        if self._len_cache is None:
            self._len_cache = self._compute_num_windows()
        return self._len_cache

    def _load_or_compute_len(self, cache_path: Optional[Path]) -> Optional[int]:
        if cache_path and cache_path.exists():
            try:
                val = int(json.loads(cache_path.read_text())["num_windows"])
                logger.info("Loaded dataset length from cache: %d", val)
                return val
            except Exception as e:
                logger.warning("Failed to read len cache: %s", e)
        val = self._compute_num_windows()
        if cache_path:
            try:
                cache_path.write_text(json.dumps({"num_windows": val}))
            except Exception as e:
                logger.warning("Failed to write len cache: %s", e)
        return val

    def _compute_num_windows(self) -> int:
        logger.info("Pre-scanning dataset to compute number of windows...")
        total = 0
        window_tokens: List[int] = []
        for path in self.files:
            nf_name = self._extract_network_function(path.name).lower()
            if nf_name in self.exclude_nfs:
                continue
            try:
                if "spec_3gpp" in path.parts and path.suffix == ".json":
                    for content in iter_spec_json(path):
                        total += self._accumulate_len_tokens(content, window_tokens)
                elif "open5gs_source_code" in path.parts:
                    for chunk in code_chunks(path):
                        total += self._accumulate_len_tokens(chunk, window_tokens)
                else:
                    with path.open(encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            total += self._accumulate_len_tokens(line.rstrip("\n"), window_tokens)
            except Exception as exc:
                logger.warning("Length scan failed for %s: %s", path, exc)
        if window_tokens:
            total += 1
        logger.info("Estimated windows: %d", total)
        return total

    def _accumulate_len_tokens(self, text: str, buf: List[int]) -> int:
        cnt = 0
        tagged = self.tagger.tag_line(text)
        for tt in tagged:
            buf.append(tt.token_id)
            if len(buf) >= self.seq_len:
                cnt += 1
                buf.clear()
        return cnt

    def _discover_files(self, root: Path, include_specs: bool, include_code: bool) -> List[Path]:
        """Return files under ``root`` in call flow/NF order."""
        files: List[Path] = []
        seen: set[Path] = set()

        logs_dir = root / "logs"
        if logs_dir.exists():
            for callflow in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
                for fp in sorted(callflow.glob("*.log")):
                    if fp.is_file() and fp not in seen:
                        files.append(fp)
                        seen.add(fp)

        for subdir in ["data", "raw_data", "preprocessed_data"]:
            p = root / subdir
            if p.exists():
                for fp in p.rglob("*"):
                    if fp.is_file() and fp not in seen:
                        files.append(fp)
                        seen.add(fp)

        if include_specs:
            spec_dir = root / "spec_3gpp"
            files.extend(p for p in spec_dir.glob("*.json") if p.is_file())
        if include_code:
            code_dir = root / "open5gs_source_code"
            files.extend(
                p for p in code_dir.rglob("*")
                if p.is_file() and p.suffix in SUPPORTED_FILE_TYPES
            )
        return files

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        files = self.files
        if worker is not None:
            per_worker = int(math.ceil(len(self.files) / float(worker.num_workers)))
            start = worker.id * per_worker
            files = files[start:start + per_worker]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Worker %d processing %d files", worker.id, len(files))

        window_tokens: List[int] = []
        window_tags: List[int] = []

        for path in files:
            if window_tokens:
                yield self._make_example(window_tokens, window_tags)
                window_tokens.clear()
                window_tags.clear()

            nf_name = self._extract_network_function(path.name)
            if nf_name.lower() in self.exclude_nfs:
                continue
            nf_tokens: List[int] = []
            nf_tags: List[int] = []
            if nf_name:
                nf_tokens, nf_tags = self._nf_prefix_tokens(nf_name)
                window_tokens.extend(nf_tokens)
                window_tags.extend(nf_tags)

            try:
                if "spec_3gpp" in path.parts and path.suffix == ".json":
                    for content in iter_spec_json(path):
                        ex = self._process_line(content, window_tokens, window_tags)
                        if ex is not None:
                            yield ex
                            if nf_name:
                                window_tokens.extend(nf_tokens)
                                window_tags.extend(nf_tags)
                elif "open5gs_source_code" in path.parts:
                    for chunk in code_chunks(path):
                        ex = self._process_line(chunk, window_tokens, window_tags)
                        if ex is not None:
                            yield ex
                            if nf_name:
                                window_tokens.extend(nf_tokens)
                                window_tags.extend(nf_tags)
                else:
                    with path.open(encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            ex = self._process_line(line.rstrip("\n"), window_tokens, window_tags)
                            if ex is not None:
                                yield ex
                                if nf_name:
                                    window_tokens.extend(nf_tokens)
                                    window_tags.extend(nf_tags)
            except Exception as exc:
                logger.warning("Failed reading %s: %s", path, exc)

        if window_tokens:
            yield self._make_example(window_tokens, window_tags)

    def _extract_network_function(self, filename: str) -> str:
        """Return network function identifier from ``filename``."""
        m = re.search(r"open5gs-([^-]+)-", filename)
        return m.group(1).upper() if m else ""

    def _nf_prefix_tokens(self, nf_name: str) -> tuple[List[int], List[int]]:
        """Tokenize a network function prefix."""
        prefix = f"NF:{nf_name}\n"
        tagged = self.tagger.tag_line(prefix)
        tokens = [t.token_id for t in tagged]
        tags = [t.tag_id for t in tagged]
        return tokens, tags

    def _process_line(self, line: str, window_tokens: List[int], window_tags: List[int]) -> Optional[dict]:
        line = ANSI_ESCAPE_RE.sub("", line)
        tagged = self.tagger.tag_line(line)
        for tt in tagged:
            window_tokens.append(tt.token_id)
            window_tags.append(tt.tag_id)
            if len(window_tokens) >= self.seq_len:
                example = self._make_example(window_tokens, window_tags)
                window_tokens.clear()
                window_tags.clear()
                return example
        return None

    def _make_example(self, tokens: List[int], tags: List[int]) -> dict:
        label = 0
        orig_tokens = tokens[:]

        if self.cfg_corruptor:
            corrupted = self.cfg_corruptor.corrupt(tokens)
            if corrupted is not None:
                tokens = corrupted
                label = 1

        if label == 0:
            tokens, did = self.neg_gen.corrupt(tokens, tags, self.hard_neg_prob, max_ops=3)
            if did:
                label = 1

        if label == 0 and random.random() < self.corruption_prob and tags:
            label = 1
            chosen_tag = random.choice(list(set(tags)))
            idxs = [i for i, t in enumerate(tags) if t == chosen_tag]
            if idxs:
                tokens[idxs[0]] = random.randint(0, self.tokenizer.vocab_size - 1)

        if logger.isEnabledFor(logging.DEBUG):
            pass
            #logger.debug("Example built, label=%d", label)

        self.neg_gen.observe(orig_tokens, tags)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "tags": torch.tensor(tags, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def _ensure_pairs_for_all_nfs(dataset: Any) -> None:
    """Populate ``dataset.pairs`` with NF-level pairings when available."""

    if not getattr(dataset, "pair_windows", False):
        return

    items: List[dict] = getattr(dataset, "items", [])
    if not items:
        return

    pairs: List[Tuple[int, int]] = getattr(dataset, "pairs", [])
    pairs_by_nf: Dict[str, List[Tuple[int, int]]] = getattr(
        dataset, "pairs_by_nf", {}
    )
    nf_to_indices: Dict[str, List[int]] = getattr(
        dataset, "nf_to_indices", {}
    )

    if not nf_to_indices:
        return

    existing_pairs: Set[Tuple[int, int]] = {tuple(pair) for pair in pairs}

    for nf, idxs in nf_to_indices.items():
        normals: List[int] = []
        anomalies: List[int] = []
        for idx in idxs:
            label = int(items[idx]["label"])
            if label == 0:
                normals.append(idx)
            elif label == 1:
                anomalies.append(idx)
        if not normals or not anomalies:
            continue
        normals.sort(key=lambda i: items[i]["nf_idx"])
        anomalies.sort(key=lambda i: items[i]["nf_idx"])
        pair_count = min(len(normals), len(anomalies))
        if pair_count == 0:
            continue
        nf_pairs = pairs_by_nf.setdefault(nf, [])
        for normal_idx, anomaly_idx in zip(
            normals[:pair_count], anomalies[:pair_count]
        ):
            first, second = sorted((normal_idx, anomaly_idx))
            pair_tuple = (first, second)
            if pair_tuple in existing_pairs:
                continue
            pairs.append(pair_tuple)
            nf_pairs.append(pair_tuple)
            existing_pairs.add(pair_tuple)

    if not pairs:
        return

    pairs.sort(key=lambda p: p[0])
    for nf in list(pairs_by_nf.keys()):
        nf_pairs = pairs_by_nf[nf]
        if not nf_pairs:
            pairs_by_nf.pop(nf, None)
            continue
        nf_pairs.sort(key=lambda p: p[0])

class PerNFDataset(Dataset):
    """Dataset yielding sequential windows grouped by network function.

    This dataset loads log lines for each network function (NF) in timestamp
    order and exposes sliding windows without any shuffling. Each item
    contains the NF name together with tokenized ``input_ids`` and ``tags``
    tensors and a ``label`` indicating whether corruption was applied.
    Optionally, explicit anomaly metadata can be supplied via
    ``anomaly_metadata_path``. When present, windows overlapping the provided
    anomaly line numbers are labeled as anomalous, and synthetic corruption is
    disabled. The negative generator pools are scoped per NF to avoid
    cross‑NF mixing. When ``pair_windows`` is ``True``, clean and anomalous
    windows from the same log file are paired to enable contrastive training.
    Set ``keep_clean_in_anomalous`` to ``True`` to retain normal windows from
    files located under ``anomalous_logs``; by default such windows are
    dropped to avoid inflating the normal set. Training code may also set
    ``exclude_anomalous_logs`` to ensure the dataset root points only to clean
    logs, completely skipping the ``anomalous_logs`` directory.
    """

    def __init__(
        self,
        root: Path,
        tokenizer: PreTrainedTokenizer,
        *,
        max_seq_len: int = 512,
        stride: int = 256,
        corruption_prob: float = 0.0,
        hard_neg_prob: float = 0.0,
        cfg_map_path: Optional[Path] = None,
        anomaly_metadata_path: Optional[Path] = None,
        pair_windows: bool = False,
        keep_clean_in_anomalous: bool = False,
        exclude_nfs: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Base directory containing log files. When
                ``exclude_anomalous_logs`` is set in the training config, this
                path should omit the ``anomalous_logs`` directory.
            tokenizer: Tokenizer used for encoding log lines.
            max_seq_len: Maximum number of tokens per window.
            stride: Step size for the sliding window.
            corruption_prob: Probability of random token corruption.
            hard_neg_prob: Probability of applying hard negative generation.
            cfg_map_path: Optional path to a CFG replacement map.
            anomaly_metadata_path: Optional path to anomaly line metadata.
            pair_windows: Pair clean and anomalous windows from the same file.
            keep_clean_in_anomalous: Retain normal windows from files under
                ``anomalous_logs`` when ``True``. Defaults to ``False``.
            exclude_nfs: Iterable of network function names to skip entirely.
        """

        self.tokenizer = tokenizer
        self.tagger = LogTagger(tokenizer)
        self.max_seq_len = int(max_seq_len)
        self.stride = int(stride)
        self.corruption_prob = corruption_prob
        self.hard_neg_prob = hard_neg_prob
        self.pair_windows = bool(pair_windows)
        self.keep_clean_in_anomalous = bool(keep_clean_in_anomalous)
        self.exclude_nfs = {nf.lower() for nf in (exclude_nfs or [])}

        self.root = Path(root)

        self.cfg_corruptor: Optional[CFGCorruptor] = None
        if cfg_map_path and cfg_map_path.exists():
            try:
                self.cfg_corruptor = CFGCorruptor(cfg_map_path, tokenizer)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load CFG map: %s", exc)

        # per‑NF negative generators
        self.neg_gens: Dict[str, HardNegativeGenerator] = {}

        # Map of relative file path -> anomaly line numbers
        self.anomaly_map: Dict[Path, Set[int]] = {}
        if anomaly_metadata_path and anomaly_metadata_path.exists():
            try:
                with anomaly_metadata_path.open() as f:
                    if anomaly_metadata_path.suffix == ".json":
                        meta = json.load(f)
                        rows = meta.get("rows", [])
                    else:
                        reader = csv.DictReader(f)
                        rows = []
                        for row in reader:
                            try:
                                line_info = json.loads(row.get("lines", "[]"))
                            except json.JSONDecodeError:
                                line_info = []
                            row["lines"] = line_info
                            rows.append(row)
                for row in rows:
                    rel = Path(row.get("file", ""))
                    lines = {
                        int(line["anomaly_line_number"])
                        for line in row.get("lines", [])
                        if "anomaly_line_number" in line
                    }
                    if rel and lines:
                        self.anomaly_map[rel] = lines
                logger.info("Loaded anomaly metadata for %d files", len(self.anomaly_map))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load anomaly metadata: %s", exc)

        self.items: List[dict] = []
        self.files: List[Path] = []
        self.nf_to_indices: Dict[str, List[int]] = {}
        # Global list of (clean_idx, anomaly_idx) tuples
        self.pairs: List[Tuple[int, int]] = []
        # Mapping of NF -> list of pairs for that NF
        self.pairs_by_nf: Dict[str, List[Tuple[int, int]]] = {}

        self._build_windows(self.root)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_windows(self, root: Path) -> None:
        """Read log files recursively and build sliding windows per NF."""
        pair_groups: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: {"clean": [], "anomaly": []}
        ) if self.pair_windows else {}

        explicit = bool(self.anomaly_map)

        for fp in sorted(root.glob("**/*.log")):
            nf = nf_from_filename(fp)
            if nf.lower() in self.exclude_nfs:
                continue
            self.files.append(fp)
            rel_fp = fp.relative_to(self.root)
            anomaly_lines = self.anomaly_map.get(rel_fp)

            tokens: List[int] = []
            tags: List[int] = []
            anoms: Set[int] = set()
            line_no = 1
            with fp.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    tagged = self.tagger.tag_line(line.rstrip("\n"))
                    start_idx = len(tokens)
                    for tt in tagged:
                        tokens.append(tt.token_id)
                        tags.append(tt.tag_id)
                    end_idx = len(tokens)
                    if anomaly_lines and line_no in anomaly_lines:
                        anoms.update(range(start_idx, end_idx))
                    line_no += 1

            self.neg_gens.setdefault(nf, HardNegativeGenerator(self.tokenizer))
            nf_indices = self.nf_to_indices.setdefault(nf, [])
            nf_win_idx = len(nf_indices)

            is_anom_file = "anomalous_logs" in fp.parts
            pair_key: Optional[str] = None
            if self.pair_windows:
                if is_anom_file:
                    parts = fp.name.split("_", 1)
                    base = parts[1] if len(parts) == 2 else fp.name
                else:
                    base = fp.name
                pair_key = f"{nf}:{base}"

            total_tokens = len(tokens)
            # Skip empty logs which would otherwise yield zero-length windows
            if total_tokens == 0:
                continue

            for start in range(
                0, max(1, total_tokens - self.max_seq_len + 1), self.stride
            ):
                end = start + self.max_seq_len
                win_tokens = tokens[start:end]
                win_tags = tags[start:end]
                orig_tokens = win_tokens[:]
                label = 1 if any(start <= a < end for a in anoms) else 0
                if not explicit:
                    if label == 0 and self.cfg_corruptor:
                        corrupted = self.cfg_corruptor.corrupt(win_tokens)
                        if corrupted is not None:
                            win_tokens = corrupted
                            label = 1
                    if label == 0 and self.hard_neg_prob > 0:
                        gen = self.neg_gens[nf]
                        win_tokens, changed = gen.corrupt(
                            win_tokens, win_tags, self.hard_neg_prob
                        )
                        if changed:
                            label = 1
                    if label == 0 and self.corruption_prob > 0 and win_tags:
                        if random.random() < self.corruption_prob:
                            label = 1
                            chosen_tag = random.choice(list(set(win_tags)))
                            idxs = [i for i, t in enumerate(win_tags) if t == chosen_tag]
                            if idxs:
                                win_tokens[idxs[0]] = random.randint(
                                    0, self.tokenizer.vocab_size - 1
                                )
                self.neg_gens[nf].observe(orig_tokens, win_tags)
                if is_anom_file and label == 0 and not self.keep_clean_in_anomalous:
                    continue
                item = {
                    "nf": nf,
                    "nf_idx": nf_win_idx,
                    "file": rel_fp,
                    "start_line_idx": start,
                    "end_line_idx": min(end, total_tokens),
                    "input_ids": torch.tensor(win_tokens, dtype=torch.long),
                    "tags": torch.tensor(win_tags, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.long),
                }
                idx = len(self.items)
                nf_indices.append(idx)
                self.items.append(item)
                if self.pair_windows and pair_key:
                    grp = pair_groups.setdefault(pair_key, {"clean": [], "anomaly": []})
                    grp["anomaly" if is_anom_file else "clean"].append(idx)
                nf_win_idx += 1

        if self.pair_windows and pair_groups:
            for key, grp in pair_groups.items():
                nf = key.split(":", 1)[0]
                for c_idx, a_idx in zip(grp["clean"], grp["anomaly"]):
                    first, second = sorted((c_idx, a_idx))
                    self.pairs.append((first, second))
                    self.pairs_by_nf.setdefault(nf, []).append((first, second))

            for nf, pairs in self.pairs_by_nf.items():
                pairs.sort(key=lambda p: p[0])
            self.pairs.sort(key=lambda p: p[0])

        # Pair balancing for the base dataset is handled by existing metadata
        # pairings; defect fine-tuning injects additional balancing logic.

    def set_corruption_prob(self, p: float) -> None:
        """Update random corruption probability."""
        self.corruption_prob = max(0.0, min(1.0, p))

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

class PerNFDatasetDefect(Dataset):
    """Dataset variant that merges clean and defect corpora for fine-tuning.

    This class mirrors :class:`PerNFDataset` but extends it to read from
    multiple directory roots.  It normalises anomaly metadata file paths so
    entries resolved relative to any configured root can be matched at runtime.
    The interface is kept identical so that downstream training code can treat
    both datasets interchangeably.
    """

    def __init__(
        self,
        root: Path,
        tokenizer: PreTrainedTokenizer,
        *,
        max_seq_len: int = 512,
        stride: int = 256,
        corruption_prob: float = 0.0,
        hard_neg_prob: float = 0.0,
        cfg_map_path: Optional[Path] = None,
        anomaly_metadata_path: Optional[Path] = None,
        pair_windows: bool = False,
        keep_clean_in_anomalous: bool = False,
        exclude_nfs: Optional[Sequence[str]] = None,
        additional_roots: Optional[Sequence[Path]] = None,
    ) -> None:
        """Initialize the dataset."""

        self.tokenizer = tokenizer
        self.tagger = LogTagger(tokenizer)
        self.max_seq_len = int(max_seq_len)
        self.stride = int(stride)
        self.corruption_prob = corruption_prob
        self.hard_neg_prob = hard_neg_prob
        self.pair_windows = bool(pair_windows)
        self.keep_clean_in_anomalous = bool(keep_clean_in_anomalous)
        self.exclude_nfs = {nf.lower() for nf in (exclude_nfs or [])}

        self.root = Path(root)
        addl_roots = [Path(p) for p in (additional_roots or [])]
        seen_roots: set[str] = set()
        self.roots: list[Path] = []
        for candidate in [self.root, *addl_roots]:
            resolved = str(candidate.resolve(strict=False))
            if resolved in seen_roots:
                continue
            seen_roots.add(resolved)
            self.roots.append(candidate)

        self.cfg_corruptor: Optional[CFGCorruptor] = None
        if cfg_map_path and cfg_map_path.exists():
            try:
                self.cfg_corruptor = CFGCorruptor(cfg_map_path, tokenizer)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load CFG map: %s", exc)

        # per‑NF negative generators
        self.neg_gens: Dict[str, HardNegativeGenerator] = {}

        # Map of normalised file path -> anomaly line numbers
        self.anomaly_map: Dict[str, Set[int]] = {}
        if anomaly_metadata_path and anomaly_metadata_path.exists():
            try:
                with anomaly_metadata_path.open() as f:
                    if anomaly_metadata_path.suffix == ".json":
                        meta = json.load(f)
                        rows = meta.get("rows", [])
                    else:
                        reader = csv.DictReader(f)
                        rows = []
                        for row in reader:
                            try:
                                line_info = json.loads(row.get("lines", "[]"))
                            except json.JSONDecodeError:
                                line_info = []
                            row["lines"] = line_info
                            rows.append(row)
                for row in rows:
                    raw_file = str(row.get("file", "")).strip()
                    if not raw_file:
                        continue
                    rel = Path(raw_file)
                    lines = {
                        int(line["anomaly_line_number"])
                        for line in row.get("lines", [])
                        if "anomaly_line_number" in line
                    }
                    if not lines:
                        continue
                    keys: Set[str] = {rel.as_posix()}
                    if rel.is_absolute():
                        try:
                            keys.add(rel.resolve(strict=False).as_posix())
                        except OSError:
                            pass
                    else:
                        for base in self.roots:
                            candidate = (base / rel).resolve(strict=False)
                            keys.add(candidate.as_posix())
                    for key in keys:
                        existing = self.anomaly_map.setdefault(key, set())
                        existing.update(lines)
                logger.info(
                    "Loaded anomaly metadata for %d files", len(self.anomaly_map)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load anomaly metadata: %s", exc)

        self.items: List[dict] = []
        self.files: List[Path] = []
        self.nf_to_indices: Dict[str, List[int]] = {}
        # Global list of (clean_idx, anomaly_idx) tuples
        self.pairs: List[Tuple[int, int]] = []
        # Mapping of NF -> list of pairs for that NF
        self.pairs_by_nf: Dict[str, List[Tuple[int, int]]] = {}

        self._build_windows()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_windows(self) -> None:
        """Read log files recursively across all configured roots."""

        pair_groups: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: {"clean": [], "anomaly": []}
        ) if self.pair_windows else {}

        explicit = bool(self.anomaly_map)
        seen: Set[str] = set()

        for base_root in self.roots:
            for fp in sorted(base_root.glob("**/*.log")):
                resolved = fp.resolve(strict=False)
                key = resolved.as_posix()
                if key in seen:
                    continue
                seen.add(key)

                nf = nf_from_filename(fp)
                if nf.lower() in self.exclude_nfs:
                    continue
                self.files.append(fp)

                rel_fp: Optional[Path] = None
                for candidate_root in self.roots:
                    try:
                        rel_fp = fp.relative_to(candidate_root)
                        break
                    except ValueError:
                        continue
                if rel_fp is None:
                    rel_fp = Path(fp.name)

                lookup_keys = {
                    fp.as_posix(),
                    resolved.as_posix(),
                    rel_fp.as_posix(),
                }
                for candidate_root in self.roots:
                    try:
                        rel = fp.relative_to(candidate_root).as_posix()
                        lookup_keys.add(rel)
                    except ValueError:
                        continue
                anomaly_lines = None
                for lookup in lookup_keys:
                    anomaly_lines = self.anomaly_map.get(lookup)
                    if anomaly_lines:
                        break

                tokens: List[int] = []
                tags: List[int] = []
                anoms: Set[int] = set()
                line_no = 1
                with fp.open(encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        tagged = self.tagger.tag_line(line.rstrip("\n"))
                        start_idx = len(tokens)
                        for tt in tagged:
                            tokens.append(tt.token_id)
                            tags.append(tt.tag_id)
                        end_idx = len(tokens)
                        if anomaly_lines and line_no in anomaly_lines:
                            anoms.update(range(start_idx, end_idx))
                        line_no += 1

                self.neg_gens.setdefault(nf, HardNegativeGenerator(self.tokenizer))
                nf_indices = self.nf_to_indices.setdefault(nf, [])
                nf_win_idx = len(nf_indices)

                is_anom_file = "anomalous_logs" in fp.parts
                pair_key: Optional[str] = None
                if self.pair_windows:
                    if is_anom_file:
                        parts = fp.name.split("_", 1)
                        base = parts[1] if len(parts) == 2 else fp.name
                    else:
                        base = fp.name
                    pair_key = f"{nf}:{base}"

                total_tokens = len(tokens)
                # Skip empty logs which would otherwise yield zero-length windows
                if total_tokens == 0:
                    continue

                for start in range(
                    0, max(1, total_tokens - self.max_seq_len + 1), self.stride
                ):
                    end = start + self.max_seq_len
                    win_tokens = tokens[start:end]
                    win_tags = tags[start:end]
                    orig_tokens = win_tokens[:]
                    label = 1 if any(start <= a < end for a in anoms) else 0
                    if not explicit:
                        if label == 0 and self.cfg_corruptor:
                            corrupted = self.cfg_corruptor.corrupt(win_tokens)
                            if corrupted is not None:
                                win_tokens = corrupted
                                label = 1
                        if label == 0 and self.hard_neg_prob > 0:
                            gen = self.neg_gens[nf]
                            win_tokens, changed = gen.corrupt(
                                win_tokens, win_tags, self.hard_neg_prob
                            )
                            if changed:
                                label = 1
                        if label == 0 and self.corruption_prob > 0 and win_tags:
                            if random.random() < self.corruption_prob:
                                label = 1
                                chosen_tag = random.choice(list(set(win_tags)))
                                idxs = [i for i, t in enumerate(win_tags) if t == chosen_tag]
                                if idxs:
                                    win_tokens[idxs[0]] = random.randint(
                                        0, self.tokenizer.vocab_size - 1
                                    )
                    self.neg_gens[nf].observe(orig_tokens, win_tags)
                    if is_anom_file and label == 0 and not self.keep_clean_in_anomalous:
                        continue
                    item = {
                        "nf": nf,
                        "nf_idx": nf_win_idx,
                        "file": rel_fp,
                        "start_line_idx": start,
                        "end_line_idx": min(end, total_tokens),
                        "input_ids": torch.tensor(win_tokens, dtype=torch.long),
                        "tags": torch.tensor(win_tags, dtype=torch.long),
                        "label": torch.tensor(label, dtype=torch.long),
                    }
                    idx = len(self.items)
                    nf_indices.append(idx)
                    self.items.append(item)
                    if self.pair_windows and pair_key:
                        grp = pair_groups.setdefault(pair_key, {"clean": [], "anomaly": []})
                        grp["anomaly" if is_anom_file else "clean"].append(idx)
                    nf_win_idx += 1

        if self.pair_windows and pair_groups:
            for key, grp in pair_groups.items():
                nf = key.split(":", 1)[0]
                for c_idx, a_idx in zip(grp["clean"], grp["anomaly"]):
                    first, second = sorted((c_idx, a_idx))
                    self.pairs.append((first, second))
                    self.pairs_by_nf.setdefault(nf, []).append((first, second))

            for nf, pairs in self.pairs_by_nf.items():
                pairs.sort(key=lambda p: p[0])
            self.pairs.sort(key=lambda p: p[0])

        if self.pair_windows:
            _ensure_pairs_for_all_nfs(self)

    def set_corruption_prob(self, p: float) -> None:
        """Update random corruption probability."""
        self.corruption_prob = max(0.0, min(1.0, p))

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

