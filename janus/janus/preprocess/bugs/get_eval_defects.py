#!/usr/bin/env python3
"""Scrape Open5GS issues and extract labelled network-function logs.

This module combines the existing scraping and preprocessing utilities into a
single workflow tailored for evaluation data generation.  It downloads issues
within a user-specified window, categorises them by label, flags anomalies
based on label taxonomy, extracts Open5GS network-function logs, and
materialises those logs in ``data/eval_data/defect_data`` grouped by network
function.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from github import Github
from github.GithubException import GithubException
from github.Issue import Issue as GithubIssue

from prepare_bug_data import (
    VALID_NFS,
    build_buggy_log_snippet,
    extract_code_snippets,
)
from janus.utils.paths import resolve_path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
LOGGER = logging.getLogger(__name__)

DEFAULT_REPO = "open5gs/open5gs"
DEFAULT_SINCE = "2024-08-05"
DEFAULT_UNTIL = "2025-09-20"
DEFAULT_OUTPUT_DIR = resolve_path("data/eval_data/defect_data")
ANOMALY_LABELS = (
    "type:bug",
    "type:security",
    "housekeeping:bugtoreview",
)
LABEL_ANOMALY = "label1-anomaly"
LABEL_NORMAL = "label0-normal"


@dataclass
class CommentRecord:
    """Lightweight representation of a GitHub issue comment."""

    body: str
    author: str | None
    created_at: datetime
    url: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Serialise the comment for metadata output."""

        return {
            "author": self.author,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "url": self.url,
        }


@dataclass
class IssueRecord:
    """Container for GitHub issue information used during processing."""

    issue_id: int
    number: int
    title: str
    body: str
    html_url: str
    created_at: datetime
    updated_at: datetime
    state: str
    author: str | None
    labels: list[str] = field(default_factory=list)
    comments: list[CommentRecord] = field(default_factory=list)

    def context(self) -> dict[str, str]:
        """Return the minimal issue context consumed by log heuristics."""

        return {
            "title": self.title or "",
            "body": self.body or "",
        }

    def to_metadata(self) -> dict[str, object]:
        """Convert the issue record to a JSON-friendly structure."""

        metadata = {
            "id": self.issue_id,
            "number": self.number,
            "title": self.title,
            "url": self.html_url,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "updated_at": self.updated_at.astimezone(timezone.utc).isoformat(),
            "state": self.state,
            "author": self.author,
            "labels": self.labels,
            "comment_count": len(self.comments),
        }
        if self.comments:
            metadata["comments"] = [comment.as_dict() for comment in self.comments]
        return metadata


def parse_iso_date(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse an ISO-like string and normalise it to UTC."""

    sanitized = value.strip()
    if sanitized.endswith("Z"):
        sanitized = sanitized[:-1] + "+00:00"
    dt = datetime.fromisoformat(sanitized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    if end_of_day:
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return dt


def ensure_utc(dt: datetime | None) -> datetime:
    """Return a timezone-aware datetime in UTC."""

    if dt is None:
        raise ValueError("Timestamp is required for issue processing.")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalise_label(label: str | None) -> str:
    """Return a lower-case label string without surrounding whitespace."""

    return (label or "").strip().lower()


def classify_issue(labels: Iterable[str]) -> str:
    """Map GitHub labels to anomaly or normal categories."""

    lower_labels = [normalise_label(label) for label in labels if label]
    for label in lower_labels:
        for target in ANOMALY_LABELS:
            if label == target:
                return LABEL_ANOMALY
            if label.startswith(f"{target}:"):
                return LABEL_ANOMALY
            if label.startswith(f"{target}-"):
                return LABEL_ANOMALY
            if label.startswith(f"{target} "):
                return LABEL_ANOMALY
            if label.startswith(f"{target}("):
                return LABEL_ANOMALY
            if label.startswith(f"{target}/"):
                return LABEL_ANOMALY
    return LABEL_NORMAL


def github_client(token: str | None) -> Github:
    """Return an authenticated GitHub client if a token is supplied."""

    if token:
        LOGGER.info("Using authenticated GitHub session.")
        return Github(token, per_page=100)
    LOGGER.warning("No GitHub token provided. Falling back to unauthenticated mode.")
    return Github(per_page=100)


def fetch_issues_in_range(
    repo, since: datetime, until: datetime
) -> Iterable[GithubIssue]:
    """Yield GitHub issues created within ``[since, until]``."""

    pager = repo.get_issues(state="all", sort="created", direction="asc")
    for raw_issue in pager:
        created = ensure_utc(raw_issue.created_at)
        if created < since:
            continue
        if created > until:
            break
        if getattr(raw_issue, "pull_request", None) is not None:
            continue
        yield raw_issue


def build_comment_record(comment) -> CommentRecord:
    """Translate a PyGithub comment into :class:`CommentRecord`."""

    return CommentRecord(
        body=comment.body or "",
        author=getattr(comment.user, "login", None),
        created_at=ensure_utc(comment.created_at),
        url=getattr(comment, "html_url", None),
    )


def materialise_issue(raw_issue: GithubIssue) -> IssueRecord:
    """Convert a PyGithub issue into a serialisable :class:`IssueRecord`."""

    comments = []
    if raw_issue.comments:
        try:
            for comment in raw_issue.get_comments():
                comments.append(build_comment_record(comment))
        except GithubException as exc:  # pragma: no cover - network edge case
            LOGGER.warning(
                "Failed to fetch comments for issue #%s: %s",
                raw_issue.number,
                exc,
            )
    return IssueRecord(
        issue_id=raw_issue.id,
        number=raw_issue.number,
        title=raw_issue.title or "",
        body=raw_issue.body or "",
        html_url=raw_issue.html_url,
        created_at=ensure_utc(raw_issue.created_at),
        updated_at=ensure_utc(raw_issue.updated_at),
        state=raw_issue.state,
        author=getattr(raw_issue.user, "login", None),
        labels=[label.name for label in raw_issue.labels],
        comments=comments,
    )


def collect_issues(
    client: Github, repo_name: str, since: datetime, until: datetime
) -> list[IssueRecord]:
    """Fetch and serialise issues within the requested range."""

    LOGGER.info(
        "Fetching issues from %s created between %s and %s.",
        repo_name,
        since.isoformat(),
        until.isoformat(),
    )
    repo = client.get_repo(repo_name)
    issues: list[IssueRecord] = []
    for raw_issue in fetch_issues_in_range(repo, since, until):
        issues.append(materialise_issue(raw_issue))
    LOGGER.info("Retrieved %d issues in the requested window.", len(issues))
    return issues


def categorise_by_label(issues: Iterable[IssueRecord]) -> dict[str, list[int]]:
    """Group issue numbers by their label names."""

    buckets: dict[str, list[int]] = defaultdict(list)
    for issue in issues:
        if not issue.labels:
            buckets["(no label)"].append(issue.number)
            continue
        for label in issue.labels:
            buckets[label].append(issue.number)
    return {label: sorted(numbers) for label, numbers in buckets.items()}


def extract_logs_for_issue(issue: IssueRecord) -> dict[str, list[str]]:
    """Return log snippets grouped by network function for an issue."""

    context = issue.context()
    aggregated: dict[str, list[str]] = defaultdict(list)

    def process_text(text: str, *, strict_mode: bool, merged_body: str) -> bool:
        snippet = extract_code_snippets(text, strict_mode=strict_mode)
        if not snippet:
            return False
        issue_context = {"title": context["title"], "body": merged_body}
        log_mapping, _ = build_buggy_log_snippet(issue_context, snippet, None)
        if not log_mapping:
            return False
        found_any = False
        for nf, log_text in log_mapping.items():
            nf_key = (nf or "").strip().lower()
            if nf_key not in VALID_NFS:
                continue
            cleaned = log_text.strip()
            if not cleaned:
                continue
            aggregated[nf_key].append(cleaned)
            found_any = True
        return found_any

    sources: list[tuple[str, str]] = [("issue_body", issue.body or "")]
    for idx, comment in enumerate(issue.comments, start=1):
        sources.append((f"comment_{idx}", comment.body))

    found_logs = False
    for name, text in sources:
        if not text:
            continue
        if name == "issue_body":
            merged_body = context["body"]
        else:
            merged_body = f"{context['body']}\n{text}"
        if process_text(text, strict_mode=True, merged_body=merged_body):
            found_logs = True

    if not found_logs:
        for name, text in sources:
            if not text:
                continue
            if name == "issue_body":
                merged_body = context["body"]
            else:
                merged_body = f"{context['body']}\n{text}"
            if process_text(text, strict_mode=False, merged_body=merged_body):
                found_logs = True

    unique_snippets = {
        nf: list(dict.fromkeys(snippets))
        for nf, snippets in aggregated.items()
        if snippets
    }
    return unique_snippets if found_logs else {}


def write_logs(
    issue: IssueRecord,
    logs_by_nf: dict[str, list[str]],
    output_dir: Path,
    classification: str,
    nf_counters: dict[str, int],
) -> list[dict[str, str]]:
    """Persist log snippets and return metadata entries for each file."""

    entries: list[dict[str, str]] = []
    for nf, snippets in sorted(logs_by_nf.items()):
        nf_lower = nf.lower()
        nf_dir = output_dir / nf_lower
        nf_dir.mkdir(parents=True, exist_ok=True)
        for snippet in snippets:
            nf_counters[nf] = nf_counters.get(nf, 0) + 1
            sample_idx = nf_counters[nf]
            filename = (
                f"issue-{issue.number}-sample-{sample_idx}-{nf_lower}-{classification}.log"
            )
            path = nf_dir / filename
            path.write_text(snippet + "\n", encoding="utf-8")
            entries.append({
                "nf": nf_lower,
                "file": path.relative_to(output_dir).as_posix(),
            })
    return entries


def build_metadata(
    issues: list[IssueRecord],
    label_groups: dict[str, list[int]],
    saved_records: list[dict[str, object]],
    since: datetime,
    until: datetime,
    output_dir: Path,
    *,
    repo_name: str,
) -> Path:
    """Persist a metadata manifest summarising the scraping run."""

    classification_counts: dict[str, int] = defaultdict(int)
    for record in saved_records:
        classification = record.get("classification")
        if classification:
            classification_counts[classification] += 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo_name,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "issue_count": len(issues),
        "label_groups": label_groups,
        "saved_issue_count": len(saved_records),
        "saved_files": sum(len(record.get("log_files", [])) for record in saved_records),
        "classification_counts": dict(classification_counts),
        "issues": saved_records,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Create the command-line interface for the scraper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub personal access token. Falls back to the GITHUB_TOKEN env var.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help="Target GitHub repository in the form 'owner/name'.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=DEFAULT_SINCE,
        help="Start date (inclusive) in ISO format. Default: 2024-08-05.",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=DEFAULT_UNTIL,
        help="End date (inclusive) in ISO format. Default: 2025-09-20.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for extracted log files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before writing new data.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """Entry point used by the command-line interface."""

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    since = parse_iso_date(args.since)
    until = parse_iso_date(args.until, end_of_day=True)
    if since > until:
        raise ValueError("The start date must be earlier than or equal to the end date.")

    client = github_client(args.token)
    issues = collect_issues(client, args.repo, since, until)
    label_groups = categorise_by_label(issues)

    output_dir = resolve_path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_records: list[dict[str, object]] = []
    nf_counters: dict[str, int] = {}

    for issue in issues:
        logs_by_nf = extract_logs_for_issue(issue)
        if not logs_by_nf:
            continue
        classification = classify_issue(issue.labels)
        log_entries = write_logs(issue, logs_by_nf, output_dir, classification, nf_counters)
        if not log_entries:
            continue
        record = issue.to_metadata()
        record["classification"] = classification
        record["log_files"] = log_entries
        saved_records.append(record)

    metadata_path = build_metadata(
        issues,
        label_groups,
        saved_records,
        since,
        until,
        output_dir,
        repo_name=args.repo,
    )
    LOGGER.info(
        "Stored %d log file(s) across %d issue(s). Metadata: %s",
        sum(len(record["log_files"]) for record in saved_records),
        len(saved_records),
        metadata_path,
    )
    return metadata_path


if __name__ == "__main__":  # pragma: no cover - manual execution utility
    main()