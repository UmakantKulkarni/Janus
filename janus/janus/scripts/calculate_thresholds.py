#!/usr/bin/env python3
import argparse
import json
import math
import re
import logging
from collections import defaultdict

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def sigmoid(x):
    """Calculates the sigmoid function."""
    return 1 / (1 + math.exp(-x))

def parse_log_file(log_file_path, all_nfs):
    """
    Parses the training log file to find the best step, calculate optimal
    probability thresholds, and apply a global fallback for missing NFs.

    Args:
        log_file_path (str): The path to the input log file.
        all_nfs (list): A list of all expected network function names.

    Returns:
        dict: A dictionary of probability thresholds for every NF.
    """
    # Regex to capture the global summary line with AUROC and S_norm/S_anom
    global_summary_re = re.compile(
        r"\[defects:\d+\] E\d+ S(\d+) \| .* AUC\(S\)=([\d\.]+) .* S_norm=([-\d\.]+) \| S_anom=([-\d\.]+)"
    )
    # Regex to capture the per-NF detail lines
    nf_detail_re = re.compile(
        r"\[PN-EVAL\] nf=(\w+) .* mean_margin_norm=([-\d\.]+) mean_margin_anom=([-\d\.]+)"
    )

    best_auroc = -1.0
    best_step = -1
    best_step_global_scores = {}
    per_step_nf_scores = defaultdict(dict)

    logger.info(f"Parsing log file: {log_file_path}...")

    try:
        with open(log_file_path, 'r') as f:
            current_step = -1
            for line in f:
                global_match = global_summary_re.search(line)
                if global_match:
                    step = int(global_match.group(1))
                    auroc = float(global_match.group(2))
                    s_norm = float(global_match.group(3))
                    s_anom = float(global_match.group(4))
                    current_step = step

                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_step = step
                        best_step_global_scores = {"norm": s_norm, "anom": s_anom}
                        logger.info(
                            f"New best step found: {best_step} with Global AUROC: {best_auroc:.4f}"
                        )
                    continue

                nf_match = nf_detail_re.search(line)
                if nf_match and current_step != -1:
                    nf_name = nf_match.group(1)
                    norm_score = float(nf_match.group(2))
                    anom_score = float(nf_match.group(3))
                    per_step_nf_scores[current_step][nf_name] = {
                        "norm": norm_score,
                        "anom": anom_score
                    }
    except FileNotFoundError:
        logger.error(f"The file '{log_file_path}' was not found.")
        return None

    if best_step == -1:
        logger.error("Could not find any valid evaluation summary lines in the log file.")
        return None

    logger.info("--- Calculation ---")
    logger.info(
        f"Best performance was at Step {best_step} with a Global AUROC of {best_auroc:.4f}."
    )

    # --- Global Fallback Calculation ---
    global_logit_threshold = (best_step_global_scores["norm"] + best_step_global_scores["anom"]) / 2
    global_prob_threshold = sigmoid(global_logit_threshold)
    logger.info(
        f"Calculated Global Fallback Threshold = {global_prob_threshold:.4f} (from global S_norm/S_anom)"
    )

    best_step_specific_scores = per_step_nf_scores.get(best_step, {})
    final_thresholds = {}

    logger.info("Assigning thresholds for each Network Function...")
    for nf in all_nfs:
        if nf in best_step_specific_scores:
            # Use specific scores if available
            scores = best_step_specific_scores[nf]
            logit_threshold = (scores["norm"] + scores["anom"]) / 2
            prob_threshold = sigmoid(logit_threshold)
            final_thresholds[nf] = prob_threshold
            logger.info(
                f"  - {nf.upper():<5}: Assigned SPECIFIC threshold = {prob_threshold:.4f}"
            )
        else:
            # Otherwise, use the global fallback
            final_thresholds[nf] = global_prob_threshold
            logger.warning(
                f"  - {nf.upper():<5}: Assigned FALLBACK threshold = {global_prob_threshold:.4f} (Not in best step batch)"
            )

    return final_thresholds

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Parse a training log file and calculate optimal probability thresholds for inference."
    )
    parser.add_argument(
        "--log-file",
        required=True,
        type=str,
        help="Path to the training log file."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=str,
        help="Path to save the calculated thresholds JSON file."
    )
    args = parser.parse_args()
    
    # --- Hardcoded list of all Network Functions ---
    # Includes NFs present in the final training phase and those that are not.
    ALL_NETWORK_FUNCTIONS = [
        "amf", "nrf", "pcf", "scp", "smf", "udm", "upf",  # Present in final phase
        "ausf", "udr", "nssf", "bsf"                       # Not present, will use fallback
    ]

    final_thresholds = parse_log_file(args.log_file, ALL_NETWORK_FUNCTIONS)

    if final_thresholds:
        with open(args.output_file, 'w') as f:
            json.dump(final_thresholds, f, indent=4)
        logger.info(f"Successfully calculated thresholds and saved them to '{args.output_file}'")

if __name__ == "__main__":
    main()