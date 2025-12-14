#!/usr/bin/env bash

# 3GPP Release 17 FTP Base URL
BASE_URL="https://www.3gpp.org/ftp/Specs/latest/Rel-17"

# Output directory for OpenAPI YAML specs
ROOT_DIR=$(PYTHONPATH="$(dirname "$0")/../.." python - <<'PY'
from janus.utils.paths import project_root
print(project_root())
PY
)
OUTPUT_DIR="$ROOT_DIR/data/raw_data/3gpp_openapi_yaml_rel17"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="openapi_download_log.txt"
echo "Downloading OpenAPI YAML files for 3GPP Release 17..." > "$LOG_FILE"

# Loop through series directories (21_series to 55_series)
for series in {21..55}_series; do
    SERIES_URL="${BASE_URL}/${series}/"
    SERIES_DIR="${OUTPUT_DIR}/${series}"
    mkdir -p "$SERIES_DIR"

    echo "Processing series: $series..." | tee -a "$LOG_FILE"

    # Check if the directory exists remotely
    wget -q --spider "$SERIES_URL"
    if [ $? -ne 0 ]; then
        echo "Skipping $series (Not Found)" | tee -a "$LOG_FILE"
        rm -rf "$SERIES_DIR"
        continue
    fi

    # Download ZIP files
    wget -r -np -nd -A "*.zip" -P "$SERIES_DIR" "$SERIES_URL"

    # Process each ZIP
    for zip_file in "$SERIES_DIR"/*.zip; do
        [ -f "$zip_file" ] || continue
        unzip_dir="${zip_file%.zip}"

        echo "Extracting $zip_file..." | tee -a "$LOG_FILE"
        mkdir -p "$unzip_dir"
        unzip -o "$zip_file" -d "$unzip_dir" >/dev/null

        # Search for YAML files inside the extracted directory
        yaml_files=$(find "$unzip_dir" -type f \( -iname "*.yaml" -o -iname "*.yml" \))

        if [[ -n "$yaml_files" ]]; then
            standard_name=$(basename "$zip_file" .zip)
            final_dir="${OUTPUT_DIR}/${standard_name}"
            mkdir -p "$final_dir"

            # Copy YAML files to the final directory
            echo "$yaml_files" | while read -r yf; do
                cp "$yf" "$final_dir/"
            done
            echo "✓ Found YAML(s), saved to: $final_dir" | tee -a "$LOG_FILE"
        fi

        # Clean up unneeded extracted content
        rm -rf "$unzip_dir"
    done

    # Clean up ZIPs and empty series folder
    rm -f "$SERIES_DIR"/*.zip
    rmdir --ignore-fail-on-non-empty "$SERIES_DIR"
done

echo "OpenAPI YAML download complete." | tee -a "$LOG_FILE"
