#!/usr/bin/env python3
"""Filter 3GPP specification dataset to a desired whitelist."""

import json
import logging

from janus.utils.paths import resolve_path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- 1.  Put the spec numbers you want to keep ------------------------------
SPEC_WHITELIST = {
    "23.501","23.502","23.503",
    "29.500","29.501","29.571","29.502","29.507","29.509","29.530","29.531",
    "33.501","33.514","33.515","33.516","33.517","33.518","33.926",
    "32.250","32.251","32.352",
    "28.531",
}

# --- 2.  Read in the original file ------------------------------------------
SOURCE_FILE = resolve_path("data/raw_data/spec_3gpp/ts_3gpp_dataset.json")
with SOURCE_FILE.open(encoding="utf-8") as fp:
    specs = json.load(fp)                  # => a list[dict]

# --- 3.  Filter by specnumber -----------------------------------------------
filtered_specs = [item for item in specs if item.get("specnumber") in SPEC_WHITELIST]

logging.info(f"Kept {len(filtered_specs)} of {len(specs)} specs.")

# --- 4.  (Optional) write them back out --------------------------------------
TARGET_FILE = resolve_path("data/raw_data/spec_3gpp/filtered_3gpp_specs.json")
with TARGET_FILE.open("w", encoding="utf-8") as fp:
    json.dump(filtered_specs, fp, ensure_ascii=False, indent=4)
