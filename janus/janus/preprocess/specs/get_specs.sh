#!/usr/bin/env bash

# 3GPP Release 17 FTP Base URL
BASE_URL="https://www.3gpp.org/ftp/Specs/latest/Rel-17"

# Output directory
ROOT_DIR=$(PYTHONPATH="$(dirname "$0")/../.." python - <<'PY'
from janus.utils.paths import project_root
print(project_root())
PY
)
OUTPUT_DIR="$ROOT_DIR/data/raw_data/spec_3gpp/3gpp_release_17"
WORD_FILE_DIR="$OUTPUT_DIR/word_docs"
PDF_DIR="$OUTPUT_DIR/pdf_docs"
mkdir -p $OUTPUT_DIR
mkdir -p $WORD_FILE_DIR
mkdir -p $PDF_DIR

# Log file
LOG_FILE="download_log.txt"
echo "Downloading 3GPP Release 17 specifications..." > $LOG_FILE

# Loop through series directories (21_series to 55_series)
for series in {21..55}_series; do
    SERIES_URL="${BASE_URL}/${series}/"
    SERIES_DIR="${OUTPUT_DIR}/${series}"
    mkdir -p $SERIES_DIR

    echo "Processing series: $series..." | tee -a $LOG_FILE

    # Get the list of ZIP files
    wget -q --spider ${SERIES_URL}
    if [ $? -ne 0 ]; then
        echo "Skipping $series (Not Found)" | tee -a $LOG_FILE
        continue
    fi

    # Download all zip files
    wget -r -np -nd -A "*.zip" -P $SERIES_DIR "${SERIES_URL}"

    # Extract ZIP files
    for zip_file in $SERIES_DIR/*.zip; do
        if [ -f "$zip_file" ]; then
            echo "Extracting: $zip_file" | tee -a $LOG_FILE
            unzip -o "$zip_file" -d "$SERIES_DIR"
        fi
    done

    # Move .docx files to the main directory
    find $SERIES_DIR -type f -name "*.docx" -exec mv {} $WORD_FILE_DIR/ \;
    find $SERIES_DIR -type f -name "*.doc" -exec mv {} $WORD_FILE_DIR/ \;
done

echo "All specifications downloaded and extracted." | tee -a $LOG_FILE

echo "Converting .doc and .docx files to .pdf..."
for file in $WORD_FILE_DIR/*.doc*; do
    if [[ -f "$file" ]]; then
        soffice --headless --convert-to pdf --outdir $PDF_DIR $file
        echo "Converted: $file -> $PDF_DIR"
    fi
done

echo "All files converted!" | tee -a $LOG_FILE
