#!/usr/bin/env python3
"""Filter bug reports based on labels and creation date."""

import json
import sys
from datetime import datetime, timezone

CUTOFF = datetime(2024, 8, 4, 23, 59, 59, tzinfo=timezone.utc)
BUG_LABEL_PREFIXES = ("type:bug", "type:security", "housekeeping:bugtoreview")

def is_bug(obj):
    # Accept either a list in "labels" or a single string "type"
    labels = obj.get("labels", [])
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        normalized = str(label).strip().lower()
        if any(normalized.startswith(prefix) for prefix in BUG_LABEL_PREFIXES):
            return True
    return str(obj.get("type", "")).lower() == "bug"

def created_ok(obj):
    ts = obj.get("created_at")
    if not ts:
        return False
    # GitHub style ISO → '2024-03-31T10:30:40+00:00'
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) <= CUTOFF

def load_objects(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for o in data:
                if isinstance(o, dict):
                    yield o
        else:  # NDJSON
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    if isinstance(o, dict):
                        yield o
                except json.JSONDecodeError:
                    continue

def main(inp, outp):
    bugs = [o for o in load_objects(inp) if is_bug(o) and created_ok(o)]
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(bugs, f, ensure_ascii=False, indent=2)
    print(f"Kept {len(bugs)} bug issues → {outp}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_bugs.py input.json output.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
