#!/usr/bin/env bash
# Monitor CPU%, Mem%, GPU Util% (avg by default), GPU Mem% (total used/total) and append to CSV.
# Usage: ./measure_resource_usage.sh /path/to/output.csv [interval_sec] [samples]
#   - interval_sec: seconds between samples (default 1; supports fractions like 0.5)
#   - samples: number of rows to write (0 = infinite; Ctrl+C to stop)
# Environment:
#   USE_MAX_GPU=1   -> report busiest GPU util instead of average across GPUs
#   GPU_DEBUG=1     -> write raw GPU queries to /tmp/gpu_metrics_debug.log

set -uo pipefail   # (no -e) avoid exiting on harmless non-zero reads/commands

CSV_OUT="${1:-}"
INTERVAL="${2:-1}"
SAMPLES="${3:-0}"   # 0 = run forever

if [[ -z "$CSV_OUT" ]]; then
  echo "Usage: $0 /path/to/output.csv [interval_sec] [samples]" >&2
  exit 1
fi

mkdir -p "$(dirname "$CSV_OUT")"
if [[ ! -f "$CSV_OUT" ]]; then
  echo "timestamp,cpu_percent,mem_percent,gpu_util_percent,gpu_mem_percent" > "$CSV_OUT"
fi

# ---------------- Helpers ----------------

read_cpu_totals() {
  # echo "total idle" from /proc/stat (idle includes iowait)
  awk '/^cpu[[:space:]]/ {
    idle=$5+$6; total=0;
    for(i=2;i<=NF;i++){ total+=$i }
    printf "%s %s\n", total, idle
    exit
  }' /proc/stat
}

get_mem_percent() {
  # Use MemAvailable (more realistic for "free" memory on Linux)
  awk '
    /MemTotal:/      { mt=$2 }
    /MemAvailable:/  { ma=$2 }
    END { if (mt>0) printf "%.2f", (100.0*(mt-ma)/mt); else printf "0.00"; }
  ' /proc/meminfo
}

get_gpu_metrics() {
  # Prints "GPU_UTIL_PCT,GPU_MEM_PCT"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf "0.00,0.00"; return
  fi
  export LC_ALL=C LANG=C

  # Query util, mem.used, mem.total for all GPUs in a single call.
  # Handle both "noheader" and "with header" CSV; strip units per column.
  local raw out su c maxu smu smt util memp
  raw="$(
    ( nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
      || nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv | tail -n +2 ) \
    2>/dev/null
  )"

  # Optional debug: store raw lines
  if [[ "${GPU_DEBUG:-0}" == "1" ]]; then
    {
      echo "---- $(date -Is) ----"
      echo "RAW (util,used,total):"
      echo "$raw"
    } >> /tmp/gpu_metrics_debug.log
  fi

  # If nothing came back, zero out
  if [[ -z "${raw// }" ]]; then
    printf "0.00,0.00"; return
  fi

  # Sum util, track max util, sum used and total memory; print: su c maxu smu smt
  out="$( awk -F',' '
      {
        u=$1; mu=$2; mt=$3;
        gsub(/[^0-9.]/,"",u);  # "100 %" -> "100"
        gsub(/[^0-9.]/,"",mu); # "4479 MiB" -> "4479"
        gsub(/[^0-9.]/,"",mt); # "16384 MiB" -> "16384"
        if(u!=""){su+=u; if(u>maxu) maxu=u; c++}
        if(mu!=""){smu+=mu}
        if(mt!=""){smt+=mt}
      }
      END {
        if(c==0){ print "0 0 0 0 0"; exit }
        printf "%s %s %s %s %s", su, c, maxu, smu, smt
      }' <<<"$raw" )"

  # Parse numbers from awk
  read -r su c maxu smu smt <<<"$out"
  # Compute util (avg or max) and mem percentage
  if [[ "${USE_MAX_GPU:-0}" == "1" ]]; then
    util=$(awk -v x="$maxu" 'BEGIN{ printf "%.2f", (x+0) }')
  else
    util=$(awk -v s="$su" -v n="$c" 'BEGIN{ if(n>0) printf "%.2f", s/n; else printf "0.00" }')
  fi
  memp=$(awk -v u="$smu" -v t="$smt" 'BEGIN{ if(t>0) printf "%.2f", 100.0*u/t; else printf "0.00" }')

  printf "%s,%s" "$util" "$memp"
}

# --------------- Main loop ---------------

# CPU baseline for delta
if ! read -r PREV_TOTAL PREV_IDLE < <(read_cpu_totals) ; then
  PREV_TOTAL=0 PREV_IDLE=0
fi

COUNT=0
trap 'exit 0' INT TERM

while :; do
  # Stop condition
  if [[ "$SAMPLES" -ne 0 && "$COUNT" -ge "$SAMPLES" ]]; then
    break
  fi

  sleep "$INTERVAL"

  # CPU%
  if read -r CUR_TOTAL CUR_IDLE < <(read_cpu_totals) ; then
    CPU_PCT=$(awk -v pt="$PREV_TOTAL" -v pi="$PREV_IDLE" -v t="$CUR_TOTAL" -v i="$CUR_IDLE" '
      BEGIN{
        dt=t-pt; di=i-pi;
        cpu=(dt>0 ? (100.0*(1.0 - di/dt)) : 0.0);
        printf "%.2f", cpu
      }')
    PREV_TOTAL="$CUR_TOTAL"; PREV_IDLE="$CUR_IDLE"
  else
    CPU_PCT="0.00"
  fi

  # Mem%
  MEM_PCT="$(get_mem_percent)"

  # GPU (avoid process substitution; capture string then split)
  gpu_line="$(get_gpu_metrics 2>/dev/null || echo '0.00,0.00')"
  GPU_UTIL_PCT="${gpu_line%%,*}"
  GPU_MEM_PCT="${gpu_line#*,}"
  # sanitize to two decimals (defensive)
  GPU_UTIL_PCT="$(awk -v x="$GPU_UTIL_PCT" 'BEGIN{gsub(/[^0-9.]/,"",x); printf("%.2f", (x==""?0:x)+0)}')"
  GPU_MEM_PCT="$(awk -v x="$GPU_MEM_PCT"  'BEGIN{gsub(/[^0-9.]/,"",x); printf("%.2f", (x==""?0:x)+0)}')"

  # Timestamp (ISO 8601)
  TS="$(date -Is)"

  # Append row
  echo "${TS},${CPU_PCT},${MEM_PCT},${GPU_UTIL_PCT},${GPU_MEM_PCT}" >> "$CSV_OUT"

  COUNT=$((COUNT+1))
done
