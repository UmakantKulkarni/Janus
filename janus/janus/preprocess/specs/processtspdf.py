#!/usr/bin/env python3
"""Convert 3GPP specification PDFs into a structured JSON dataset."""

import os
import re
import fitz  # PyMuPDF
import json
import pandas as pd
import logging

from janus.utils.paths import resolve_path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

PDF_DIR = resolve_path("data/raw_data/spec_3gpp/3gpp_release_17/pdf_docs")
OUTPUT_FILE = resolve_path("data/raw_data/spec_3gpp/ts_3gpp_dataset.json")

EXCLUDE_SECTIONS = ["foreword", "references", "change history", "void"]

def extract_spec_number(filename):
    """Extracts the 3GPP specification number from filenames like '33210-h10.doc'."""
    match = re.match(r"(\d{5})", filename)  # Extracts first 5 digits as spec number
    #return f"3GPP TS {match.group(1)[:2]}.{match.group(1)[2:]}" if match else "Unknown"
    return f"{match.group(1)[:2]}.{match.group(1)[2:]}" if match else "Unknown"

def save_pdf_as_text_fixed(pdf_path):
    """Extracts text from a PDF, fixes line breaks, and returns cleaned text."""
    doc = fitz.open(pdf_path)
    txt_file = pdf_path.replace(".pdf", ".txt")
    full_text = []

    for page in doc:
        text = page.get_text("text")
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # Fix broken lines
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        full_text.append(text)

    all_txt = "\n".join(full_text)
    with open(txt_file, 'w') as output:
        output.write(all_txt)
    return all_txt

def extract_bookmarks(pdf_path):
    """Extracts section titles from PDF bookmarks."""
    doc = fitz.open(pdf_path)
    bookmarks = doc.get_toc(simple=False)

    sections = []
    for bookmark in bookmarks:
        if len(bookmark) >= 3:
            _, title, _ = bookmark[:3]
            title = title.strip()
            sections.append(title)

    return sections

def extract_text_between_sections(spec_number, df_specs, full_text, section_titles):
    """Extracts structured text from full_text using section titles."""

    specseries = spec_number.split(".")[0]
    specrelease = 17
    try:
        spectype, specwg, spectech, specinfo, specipr = df_specs.loc[df_specs['Spec No'] == spec_number, ['Type', 'Primary Resp Grp', 'Technology', 'Title', 'Initial planned Release']].values[0]
    except Exception:
        spectype, specwg, spectech, specinfo, specipr = (
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
        )
    
    spec_info = specinfo.split("; ")
    structured_data = []
    valid_sections = []
    section_positions = {}

    # Identify section positions
    for title in section_titles:
        first_pos = full_text.find(title)
        start_pos = full_text.find(title, first_pos + 1)  # Get second occurrence
        if start_pos == -1:
            continue

        valid_sections.append(title)
        section_positions[title] = start_pos

    # Process sections

    for i in range(len(valid_sections)):
        current_title = valid_sections[i]
        start_pos = section_positions[current_title]

        # Find the next valid section's start position
        end_pos = None
        for j in range(i + 1, len(valid_sections)):
            next_title = valid_sections[j]
            if next_title in section_positions:
                end_pos = section_positions[next_title]
                break

        if end_pos is None:
            end_pos = len(full_text)  # Last section gets text till end

        # Extract only the actual section text
        section_content = full_text[start_pos:end_pos].strip()

        # Prevent parent sections from getting subsection content
        if section_content.startswith(current_title):  
            section_content = section_content[len(current_title):].strip()  

        # Store section data
        if section_content:
            structured_data.append({
                "specnumber": spec_number,
                "series": specseries,
                "spectype": spectype,
                "specwg": specwg,
                "specipr": specipr,
                "technology": spectech.split(","),
                "topic": spec_info,
                "release": specrelease,
                "title": current_title,
                "content": section_content
            })


    return structured_data

def process_all_pdfs(pdf_dir):
    """Processes all PDFs and extracts structured text using section titles."""
    all_specs = []
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    df_specs = pd.read_csv('3gpp_specs_details.csv')

    for pdf_file in pdf_files:
        try:
            logging.info(f"Processing {pdf_file}...")
            spec_number = extract_spec_number(os.path.basename(pdf_file))
            text_file_data = save_pdf_as_text_fixed(pdf_file)
            json_sections_data = extract_bookmarks(pdf_file)
            structured_data = extract_text_between_sections(spec_number, df_specs, text_file_data, json_sections_data)
            structured_data = filter_excluded_sections(structured_data)
            for entry in structured_data:
                formatted_title = re.sub(r'^[A-Za-z]?\d+(\.\d+)*\s+', '', entry["title"])
                formatted_title = re.sub(r'^[A-Za-z]\.\d+(\.\d+)*\s+', '', formatted_title)
                entry["title"] = formatted_title
                # entry["title"] = f"3GPP TS {spec_number} - Section {entry['title']}"
            all_specs.extend(structured_data)
        except Exception as e:
            logging.error(f"Error processing {pdf_file}: {e}")

    return all_specs

def filter_excluded_sections(data):
    """Removes JSON objects where the title contains excluded words."""
    filtered_data = []
    for entry in data:
        title = entry["title"].lower()
        if any(exclude_word in title for exclude_word in EXCLUDE_SECTIONS) or title == "introduction":
            continue  # Skip this section
        filtered_data.append(entry)
    return filtered_data

def save_json(data, output_file):
    """Saves extracted and filtered data to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    logging.info("Extracting structured text using PDF bookmarks...")
    structured_data = process_all_pdfs(PDF_DIR)
    save_json(structured_data, OUTPUT_FILE)
    logging.info(f"Formatted text saved in {OUTPUT_FILE}")

    df = pd.DataFrame(structured_data)
    PROTOCOL_FEATURE_VOCAB = {
        "specnumber": df["specnumber"].unique().tolist(),
        "series": df["series"].unique().tolist(),
        "technology": df["technology"].explode().unique().tolist(),
        "topic": df["topic"].explode().unique().tolist(),
        "specwg": df["specwg"].unique().tolist(),
        "specipr": df["specipr"].unique().tolist(),
    }
    logging.info("Protocol feature vocabulary:")
    logging.info(json.dumps(PROTOCOL_FEATURE_VOCAB, indent=4))
    json_data = df.to_json(orient="records", indent=4)
    logging.info("Dataset saved in ts_3gpp_dataset.json")