#!/usr/bin/env bash
# Sweep dual-pass training hyperparameters one at a time and capture logs/metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${ROOT_DIR}/janus/janus/config/dualpass.yml"

if [ ! -f "${CONFIG}" ]; then
    echo "Dual-pass configuration not found at ${CONFIG}" >&2
    exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "The 'accelerate' CLI is required but was not found in PATH." >&2
    exit 1
fi

TEMP_CONFIG="$(mktemp "${CONFIG%.yml}.sweep.XXXXXX.yml")"
trap 'rm -f "${TEMP_CONFIG}"' EXIT

TEMP_SAVE_ROOT="${ROOT_DIR}/artifacts/temp"
mkdir -p "${TEMP_SAVE_ROOT}"

DEFAULT_LAMBDA_CLM=0.3
DEFAULT_LAMBDA=1.0
DEFAULT_AUC_WEIGHT=0.5

format_value() {
    echo "$1" | sed 's/\./p/g'
}

update_config() {
    local base_config="$1"
    local out_config="$2"
    local lambda_clm="$3"
    local lambda_pn="$4"
    local auc_weight="$5"
    local save_path="$6"
    local loss_path="$7"

    python - <<'PY' "${base_config}" "${out_config}" "${lambda_clm}" "${lambda_pn}" "${auc_weight}" "${save_path}" "${loss_path}"
import sys
from pathlib import Path
import yaml

base_config, out_config, lambda_clm, lambda_pn, auc_weight, save_path, loss_path = sys.argv[1:8]

with open(base_config, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

cfg["save_path"] = Path(save_path).as_posix()
cfg["loss_log_path"] = Path(loss_path).as_posix()

training = cfg.setdefault("training", {})
training["lambda_clm"] = float(lambda_clm)
training["lambda"] = float(lambda_pn)
training["auc_weight"] = float(auc_weight)

with open(out_config, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
PY
}

run_experiment() {
    local param="$1"
    local lambda_clm="$2"
    local lambda_pn="$3"
    local auc_weight="$4"

    local value
    case "${param}" in
        lambda_clm) value="${lambda_clm}" ;;
        lambda) value="${lambda_pn}" ;;
        auc_weight) value="${auc_weight}" ;;
        *) echo "Unknown parameter ${param}" >&2; exit 1 ;;
    esac

    local value_tag
    value_tag="$(format_value "${value}")"
    local run_dir="${TEMP_SAVE_ROOT}/${param}_${value_tag}"
    local adapter_dir="${run_dir}/adapter"
    local metrics_dir="${run_dir}/metrics"
    local loss_path="${metrics_dir}/training_loss.csv"
    local final_loss_path="${metrics_dir}/training_loss_${param}_${value_tag}.csv"
    local logs_dir="${run_dir}/logs"
    local log_file="${logs_dir}/dualpass_${param}_${value_tag}.log"

    if [ -d "${run_dir}" ]; then
        echo "Removing existing directory ${run_dir}"
        rm -rf "${run_dir}"
    fi

    mkdir -p "${adapter_dir}" "${metrics_dir}" "${logs_dir}"

    update_config "${CONFIG}" "${TEMP_CONFIG}" "${lambda_clm}" "${lambda_pn}" "${auc_weight}" "${adapter_dir}" "${loss_path}"
    cp "${TEMP_CONFIG}" "${run_dir}/dualpass_${param}_${value_tag}.yml"

    echo "Running dual-pass training with ${param}=${value}" | tee "${log_file}"
    (
        cd "${ROOT_DIR}" && \
        PYTHONPATH="${ROOT_DIR}/janus" accelerate launch --num_processes=2 --num_machines=1 --machine_rank=0 -m janus.train.train --config "${TEMP_CONFIG}"
    ) | tee -a "${log_file}"

    if [ -f "${loss_path}" ]; then
        mv "${loss_path}" "${final_loss_path}"
        echo "Saved loss metrics to ${final_loss_path}" | tee -a "${log_file}"
    else
        echo "Warning: loss metrics not found at ${loss_path}" | tee -a "${log_file}"
    fi
}

# Sweep lambda_clm values
for value in 0.2 0.4 0.6 0.8 1.0; do
    run_experiment "lambda_clm" "${value}" "${DEFAULT_LAMBDA}" "${DEFAULT_AUC_WEIGHT}"

done

# Sweep lambda (PN loss weight) values
for value in 0.2 0.4 0.6 0.8; do
    run_experiment "lambda" "${DEFAULT_LAMBDA_CLM}" "${value}" "${DEFAULT_AUC_WEIGHT}"

done

# Sweep auc_weight values
for value in 0.2 0.4 0.6 0.8 1.0; do
    run_experiment "auc_weight" "${DEFAULT_LAMBDA_CLM}" "${DEFAULT_LAMBDA}" "${value}"

done

echo "Hyperparameter sweep completed. Results stored under ${TEMP_SAVE_ROOT}."