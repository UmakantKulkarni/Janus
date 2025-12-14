"""Prepare filtered GitHub issue data for training."""

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from janus.utils.paths import resolve_path

# --- Configuration ---
VALID_NFS = {
    "amf",
    "smf",
    "upf",
    "ausf",
    "pcf",
    "udm",
    "udr",
    "nrf",
    "nssf",
    "bsf",
    "scp",
}
INPUT_FILE = resolve_path("data/preprocessed_data/open5gs_bugs.json")
HIGH_CONF_OUTPUT_FILE = resolve_path("data/preprocessed_data/high_confidence_data.json")
MEDIUM_CONF_OUTPUT_FILE = resolve_path("data/preprocessed_data/medium_confidence_data.json")
CUTOFF_DATE = datetime(2024, 8, 4, tzinfo=timezone.utc)
LOG_LEVEL = logging.DEBUG
REFERENCE_LOG_DIR = resolve_path("data/raw_data/logs")
LOG_LINE_PATTERN = re.compile(
    r"^(?P<date>\d{2}/\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*:\s*"
    r"\[(?P<module>[^\]]+)\]\s+"
    r"(?P<level>[A-Z]+)\s*:\s*(?P<message>.+)$"
)

# --- Logger Setup (same as before) ---
class ColorFormatter(logging.Formatter):
    GREY, YELLOW, RED, BOLD_RED, GREEN, RESET = "\x1b[38;20m", "\x1b[33;20m", "\x1b[31;20m", "\x1b[31;1m", "\x1b[32;20m", "\x1b[0m"
    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.INFO: GREY + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.WARNING: YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s - %(levelname)s - %(message)s" + RESET,
    }
    SUCCESS = GREEN + "%(asctime)s - SUCCESS - %(message)s" + RESET
    def format(self, record):
        log_fmt = self.SUCCESS if record.levelno == 25 else self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, "SUCCESS")

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(ColorFormatter())
        logger.addHandler(ch)
    def success(message, *args, **kws):
        if logger.isEnabledFor(logging.SUCCESS):
            logger._log(logging.SUCCESS, message, args, **kws)
    logger.success = success
    return logger

logger = setup_logger()

# --- Log Reference Helpers ---

def normalize_nf_name(name):
    """Return a lower-case NF identifier if it is part of the supported set."""
    if not name:
        return None
    candidate = str(name).strip().lower()
    return candidate if candidate in VALID_NFS else None


def load_reference_log_metadata(log_root):
    """Load modules and severity levels observed in reference Open5GS logs."""
    metadata = {
        "modules": {nf: set() for nf in VALID_NFS},
        "levels": {nf: set() for nf in VALID_NFS},
        "all_modules": set(),
        "all_levels": set(),
    }

    path = Path(log_root)
    if not path.exists():
        logger.warning(
            "Reference log directory '%s' does not exist. "
            "Falling back to pattern-only validation.",
            log_root,
        )
        return metadata

    for log_path in path.rglob("*.log"):
        nf_match = re.search(r"open5gs-([a-z0-9]+)-", log_path.name)
        if not nf_match:
            continue
        nf_name = normalize_nf_name(nf_match.group(1))
        if not nf_name:
            continue
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
                for raw_line in log_file:
                    match = LOG_LINE_PATTERN.match(raw_line.strip())
                    if not match:
                        continue
                    module = match.group("module").strip().lower()
                    level = match.group("level").strip().upper()
                    metadata["modules"][nf_name].add(module)
                    metadata["levels"][nf_name].add(level)
                    metadata["all_modules"].add(module)
                    metadata["all_levels"].add(level)
        except OSError as exc:
            logger.warning("Failed to read reference log '%s': %s", log_path, exc)

    return metadata


REFERENCE_LOG_METADATA = load_reference_log_metadata(REFERENCE_LOG_DIR)


def build_module_index(metadata):
    """Build a reverse lookup from log module name to network functions."""

    module_map = defaultdict(set)
    for nf_name, modules in metadata.get("modules", {}).items():
        for module in modules:
            module_map[module].add(nf_name)
    return {module: frozenset(nfs) for module, nfs in module_map.items()}


MODULE_TO_NFS = build_module_index(REFERENCE_LOG_METADATA)


def extract_log_line_info(log_text):
    """Return structured metadata for valid log lines found in the text."""
    if not log_text:
        return []

    line_infos = []
    modules_whitelist = REFERENCE_LOG_METADATA.get("all_modules", set())
    levels_whitelist = REFERENCE_LOG_METADATA.get("all_levels", set())

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LOG_LINE_PATTERN.match(line)
        if not match:
            continue
        module = match.group("module").strip().lower()
        level = match.group("level").strip().upper()
        if modules_whitelist and module not in modules_whitelist:
            continue
        if levels_whitelist and level not in levels_whitelist:
            continue
        line_infos.append(
            {
                "line": line,
                "module": module,
                "level": level,
                "message": match.group("message"),
            }
        )

    return line_infos


def nf_token_score(line_lower, nf_name):
    """Return a heuristic score showing whether a line references an NF."""

    score = 0
    if f"/{nf_name}/" in line_lower:
        score += 2
    if f"{nf_name}.yaml" in line_lower or f"{nf_name}.log" in line_lower:
        score += 2
    if f"[{nf_name}]" in line_lower:
        score += 3
    if re.search(rf"\b{re.escape(nf_name)}\b", line_lower):
        score += 1
    return score


def identify_nf_for_line(line_info, primary_nf, fallback_nf):
    """Identify the most likely NF responsible for a single log line."""

    module = line_info["module"]
    module_candidates = MODULE_TO_NFS.get(module, frozenset())
    fallback_normalized = normalize_nf_name(fallback_nf)

    considered_nfs = set(module_candidates) if module_candidates else set(VALID_NFS)
    scores = Counter()
    line_lower = line_info["line"].lower()

    for nf in considered_nfs:
        if module_candidates:
            scores[nf] += 4
        token_bonus = nf_token_score(line_lower, nf)
        if token_bonus:
            scores[nf] += token_bonus

    if primary_nf and primary_nf in considered_nfs:
        scores[primary_nf] += 1
    elif primary_nf and not module_candidates:
        scores[primary_nf] += 1

    if fallback_normalized and fallback_normalized in considered_nfs:
        scores[fallback_normalized] += 0.5
    elif fallback_normalized and not module_candidates:
        scores[fallback_normalized] += 0.5

    positive_scores = [(nf, score) for nf, score in scores.items() if score > 0]
    if not positive_scores:
        return primary_nf or fallback_normalized

    # Restrict to module candidates when available to avoid accidental matches.
    if module_candidates:
        best_nf = max(module_candidates, key=lambda nf: scores.get(nf, 0))
        if scores.get(best_nf, 0) > 0:
            return best_nf
        return None

    best_nf = max(positive_scores, key=lambda item: (item[1], item[0]))[0]
    return best_nf


def infer_log_nf(issue, line_infos, fallback_nf):
    """Infer the Open5GS NF responsible for a log snippet."""
    if not line_infos:
        return normalize_nf_name(fallback_nf)

    nf_scores = Counter()
    fallback_normalized = normalize_nf_name(fallback_nf)
    if fallback_normalized:
        nf_scores[fallback_normalized] += 5

    for nf in VALID_NFS:
        modules_for_nf = REFERENCE_LOG_METADATA["modules"].get(nf, set())
        module_matches = sum(1 for info in line_infos if info["module"] in modules_for_nf)
        nf_scores[nf] += module_matches * 2
        for info in line_infos:
            line_lower = info["line"].lower()
            if f"/{nf}/" in line_lower:
                nf_scores[nf] += 3
            if f"{nf}.yaml" in line_lower or f"{nf}.log" in line_lower:
                nf_scores[nf] += 3
            if f"[{nf}]" in line_lower:
                nf_scores[nf] += 4

    issue_title = (issue.get("title") or "").lower()
    issue_body = (issue.get("body") or "").lower()
    for nf in VALID_NFS:
        if issue_title:
            nf_scores[nf] += 2 * len(re.findall(rf"\b{nf}\b", issue_title))
        if issue_body:
            nf_scores[nf] += len(re.findall(rf"\b{nf}\b", issue_body))

    if not nf_scores:
        return fallback_normalized

    best_nf, best_score = nf_scores.most_common(1)[0]
    if fallback_normalized and nf_scores[fallback_normalized] >= best_score:
        return fallback_normalized
    return best_nf if best_score > 0 else fallback_normalized


def build_buggy_log_snippet(issue, log_text, fallback_nf):
    """Sanitize log text and return a mapping of NF to log lines and the NF guess."""

    line_infos = extract_log_line_info(log_text)
    if not line_infos:
        return None, None

    primary_nf = infer_log_nf(issue, line_infos, fallback_nf)
    nf_line_map = defaultdict(list)

    for info in line_infos:
        assigned_nf = identify_nf_for_line(info, primary_nf, fallback_nf)
        if not assigned_nf or assigned_nf not in VALID_NFS:
            continue

        nf_modules = REFERENCE_LOG_METADATA["modules"].get(assigned_nf, set())
        module_candidates = MODULE_TO_NFS.get(info["module"], frozenset())
        if nf_modules and info["module"] not in nf_modules and module_candidates:
            # Known module but not associated with the assigned NF → skip for strictness.
            continue

        nf_line_map[assigned_nf].append(info["line"])

    sanitized_map = {
        nf: "\n".join(dict.fromkeys(lines))
        for nf, lines in nf_line_map.items()
        if nf in VALID_NFS and lines
    }

    if sanitized_map:
        return sanitized_map, primary_nf

    if primary_nf and primary_nf in VALID_NFS:
        fallback_lines = [info["line"] for info in line_infos]
        if fallback_lines:
            unique_lines = list(dict.fromkeys(fallback_lines))
            return {primary_nf: "\n".join(unique_lines)}, primary_nf

    return None, primary_nf

# --- Data-Specific Processing Functions ---

def load_issues(filename):
    """Loads the list of GitHub issues from the filtered JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            logger.info(f"Reading filtered bug reports from '{filename}'...")
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: '{filename}'.")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{filename}'.")
        return []

def extract_code_snippets(text_body, strict_mode=True):
    """
    Extracts code/log data from a text body.
    - In strict mode, it looks for specific log patterns.
    - In relaxed mode (strict_mode=False), it grabs any code block.
    """
    if not text_body:
        return None
    
    code_blocks = re.findall(r"```(.*?)```", text_body, re.DOTALL)
    if not code_blocks:
        return None

    if not strict_mode:
        # Relaxed mode: return the first non-empty code block
        for block in code_blocks:
            if block.strip():
                return "\n".join(line.strip() for line in block.strip().split('\n'))
        return None

    # Strict mode (original logic)
    log_snippets = []
    log_keywords = [
        r'\d{2}/\d{2} \d{2}:\d{2}:\d{2}', r'\[(app|core|amf|smf|upf|ausf|pcf|udm|udr|nrf|nssf|bsf|scp|mme|sgwc)\]',
        'FATAL:', 'ERROR:', 'WARNING:', 'backtrace()', 'segmentation fault'
    ]

    for block in code_blocks:
        found_keywords = sum(1 for keyword in log_keywords if re.search(keyword, block, re.IGNORECASE))
        if found_keywords >= 2:
            cleaned_lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            log_snippets.append("\n".join(cleaned_lines))
    
    return "\n---\n".join(log_snippets) if log_snippets else None

def determine_faulty_nf(commits):
    """Heuristically determines the faulty NF from commit data."""
    if not commits:
        return "Unknown"
    nf_counter = Counter()
    nf_map = {
        'src/amf/': 'AMF', 'src/smf/': 'SMF', 'src/upf/': 'UPF', 'src/ausf/': 'AUSF', 'src/pcf/': 'PCF', 'src/udm/': 'UDM', 'src/udr/': 'UDR', 'src/nrf/': 'NRF', 'src/nssf/': 'NSSF', 'src/bsf/': 'BSF', 'src/scp/': 'SCP', 'src/mme/': 'MME', 'src/sgwc/': 'SGW-C', 'src/sgwu/': 'SGW-U', 'lib/core/': 'Core', 'lib/gtp/': 'Core', 'lib/pfcp/': 'Core'
    }
    for commit in commits:
        for file_info in commit.get('files_changed', []):
            for prefix, nf in nf_map.items():
                if file_info.get('filename', '').startswith(prefix):
                    nf_counter[nf] += 1
                    break
    return nf_counter.most_common(1)[0][0] if nf_counter else "Core"

def generate_analysis_text(issue):
    """Generates a clean, instructional summary of the bug."""
    if issue.get('commits'):
        msg = issue['commits'][0].get('message', '').split('\n')[0].strip()
        msg = re.sub(r'^\[\w+\]\s*', '', msg).strip()
        msg = re.sub(r'\s*\(\#\d+\)$', '', msg).strip()
        if len(msg) > 20:
            return f"The anomaly is related to: {msg}."
    return f"The anomaly is related to: {issue.get('title', 'Unspecified issue')}."

def process_and_prepare_data(issues):
    """
    Processes bug reports into high-confidence and medium-confidence datasets.
    """
    high_confidence_data = []
    medium_confidence_candidates = []

    logger.info(f"Processing {len(issues)} filtered bug reports into two tiers...")

    for issue in issues:
        # Filter by date first
        created_at_str = issue.get('created_at')
        try:
            if datetime.fromisoformat(created_at_str.replace('Z', '+00:00')) > CUTOFF_DATE:
                continue
        except (ValueError, TypeError):
            logger.warning(f"Skipping issue {issue['id']} due to invalid date: {created_at_str}")
            continue

        # Tier 1: Strict log extraction
        log_snippet_text = extract_code_snippets(issue.get('body'), strict_mode=True)

        training_example = {
            "issue_id": issue.get('id'),
            "issue_title": issue.get('title'),
            "faulty_nf": determine_faulty_nf(issue.get('commits')),
            "analysis_text": generate_analysis_text(issue)
        }

        if log_snippet_text:
            log_mapping, inferred_nf = build_buggy_log_snippet(
                issue, log_snippet_text, training_example["faulty_nf"]
            )
            if log_mapping:
                training_example["buggy_log_snippet"] = log_mapping
                if inferred_nf and inferred_nf in VALID_NFS:
                    training_example["faulty_nf"] = inferred_nf.upper()
                high_confidence_data.append(training_example)
                continue

        # If strict extraction fails, add to medium-confidence candidates
        medium_confidence_candidates.append(training_example)

    # Tier 2: Process the discarded issues with relaxed rules
    medium_confidence_data = []
    for candidate in medium_confidence_candidates:
        # Find the original full issue object to access the body again
        original_issue = next(
            (iss for iss in issues if iss.get("id") == candidate["issue_id"]),
            None,
        )
        if not original_issue:
            continue

        # Use relaxed extraction on the issue body
        relaxed_snippet = extract_code_snippets(original_issue.get('body'), strict_mode=False)
        if relaxed_snippet:
            log_mapping, inferred_nf = build_buggy_log_snippet(
                original_issue, relaxed_snippet, candidate["faulty_nf"]
            )
            if log_mapping:
                candidate["buggy_log_snippet"] = log_mapping
                if inferred_nf and inferred_nf in VALID_NFS:
                    candidate["faulty_nf"] = inferred_nf.upper()
                medium_confidence_data.append(candidate)
                continue
        logger.warning(
            f"Issue {candidate['issue_id']} was discarded from both tiers "
            "(no valid log blocks found)."
        )

    return high_confidence_data, medium_confidence_data

def save_data(data, filename, tier_name):
    """Saves a dataset to a JSON file."""
    if not data:
        logger.warning(f"No data generated for the {tier_name} dataset. File will not be created.")
        return
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.success(f"Successfully prepared {len(data)} bug reports for the {tier_name} dataset.")
        logger.success(f"Training data saved to '{filename}'.")
    except IOError as e:
        logger.error(f"Failed to write to file '{filename}'. Error: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting tiered GitHub issue preprocessing for Janus training.")
    github_issues = load_issues(INPUT_FILE)
    
    if github_issues:
        high_conf, med_conf = process_and_prepare_data(github_issues)
        save_data(high_conf, HIGH_CONF_OUTPUT_FILE, "High-Confidence")
        save_data(med_conf, MEDIUM_CONF_OUTPUT_FILE, "Medium-Confidence")
    else:
        logger.error("Preprocessing stopped. Could not load input file.")

