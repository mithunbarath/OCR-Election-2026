"""
Tamil Electoral Roll — Box-Cut OCR Pipeline
============================================
Processes all 371 Tamil PDFs in the 119-eroll/ folder.

Strategy:
  1. Convert each PDF page to a high-res image (300 DPI) using PyMuPDF
  2. Use OpenCV to detect individual voter card boxes (black borders)
  3. OCR each card with Tesseract using tam+eng language
  4. Parse Tamil field labels with regex
  5. Export all results to a single CSV

Requirements:
  pip install pymupdf opencv-python pytesseract numpy

Usage:
  # Test on first 2 PDFs:
  python tamil_ocr_pipeline.py --test

  # Run on all 371 PDFs:
  python tamil_ocr_pipeline.py

  # Run on a specific PDF:
  python tamil_ocr_pipeline.py --file "119-eroll/2026-EROLLGEN-S22-119-SIR-DraftRoll-Revision1-TAM-1-WI.pdf"
"""

import os
import re
import csv
import sys
import time
import argparse
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import fitz          # PyMuPDF — no Poppler needed!
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
PDF_FOLDER      = r"C:\Users\navee\Downloads\OCR-Election-2026"
OUTPUT_CSV      = r"C:\Users\navee\Downloads\OCR-Election-2026\tamil_voters_extracted.csv"
TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DPI             = 300           # Higher = better quality, slower
MAX_WORKERS     = 3             # Parallel PDF workers (reduce if RAM is low)
RESUME          = True          # Skip already-processed PDFs

# Tesseract config for Tamil + English
TESS_CONFIG = r"--oem 1 --psm 6 -l tam+eng"

# ─────────────────────────────────────────────
#  TAMIL FIELD PATTERNS (compiled once)
# ─────────────────────────────────────────────
# These match printed Tamil labels on the voter card
P_NAME         = re.compile(r'பெயர்\s*[:\-]\s*(.+?)(?=\n|தந்தை|கணவர்|தாய்|வீட்டு|வயது|$)', re.DOTALL)
P_FATHER       = re.compile(r'தந்தை(?:யின்)?\s*(?:பெயர்)?\s*[:\-]\s*(.+?)(?=\n|வீட்டு|வயது|$)', re.DOTALL)
P_HUSBAND      = re.compile(r'கணவர்\s*(?:பெயர்)?\s*[:\-]\s*(.+?)(?=\n|வீட்டு|வயது|$)', re.DOTALL)
P_MOTHER       = re.compile(r'தாய(?:ின்)?\s*(?:பெயர்)?\s*[:\-]\s*(.+?)(?=\n|வீட்டு|வயது|$)', re.DOTALL)
P_HOUSE        = re.compile(r'வீட்டு\s*எண்\s*[:\-]\s*([A-Za-z0-9/\\\-\.]+)')
P_AGE          = re.compile(r'வயது\s*[:\-]\s*(\d+)')
P_EPIC         = re.compile(r'\b([A-Z]{2,4}[0-9]{6,8})\b')
P_SERIAL       = re.compile(r'^\s*(\d{1,5})\s*$', re.MULTILINE)
P_GENDER_F     = re.compile(r'பெண்')
P_GENDER_M     = re.compile(r'\bஆண்\b')

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

CSV_COLUMNS = [
    "serial_number", "epic_id", "name",
    "relation_type", "relation_name",
    "house_number", "age", "gender",
    "source_file", "page_number", "card_index"
]

# Thread-safe CSV writing
_csv_lock = threading.Lock()

# ─────────────────────────────────────────────
#  STEP 1: PDF → Page Images
# ─────────────────────────────────────────────
def pdf_to_page_images(pdf_path: str, dpi: int = 300):
    """
    Convert PDF pages to PIL images using PyMuPDF (no Poppler required).
    Yields (page_num, PIL_Image).
    """
    zoom = dpi / 72.0          # 72 is PyMuPDF's default DPI
    mat  = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        # Convert to PIL Image
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield page_num + 1, img
    doc.close()

# ─────────────────────────────────────────────
#  STEP 2: Detect Voter Card Boxes
# ─────────────────────────────────────────────
def detect_voter_cards(pil_image: Image.Image):
    """
    Find individual voter card boxes on a page using OpenCV contour detection.
    Returns a list of (x, y, w, h) bounding boxes sorted top-to-bottom, left-to-right.
    Falls back to a fixed 3-column grid if detection fails.
    """
    # Convert to OpenCV BGR
    img_np = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    page_h, page_w = gray.shape
    page_area = page_h * page_w
    min_card_area = page_area * 0.025   # Card must be at least 2.5% of page
    max_card_area = page_area * 0.45    # Card must be at most 45% of page

    # --- Thresholding to find dark borders ---
    # Use inverse binary: white cards on dark grid lines
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect border fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size and shape
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_card_area or area > max_card_area:
            continue
        aspect = w / h if h > 0 else 0
        if aspect < 0.3 or aspect > 3.5:
            continue
        # Must have reasonable minimum dimensions
        if w < page_w * 0.15 or h < page_h * 0.05:
            continue
        candidates.append((x, y, w, h))

    # Remove duplicates / very overlapping boxes
    candidates = _merge_overlapping_boxes(candidates)

    # Sort: top to bottom, then left to right (within same row)
    candidates = _sort_boxes(candidates)

    # If we got a reasonable number, use contour results
    if 10 <= len(candidates) <= 40:
        return candidates

    # ── FALLBACK: Fixed 3-column grid ───────────────────────
    print(f"    [WARN] Contour detection found {len(candidates)} boxes — using grid fallback")
    return _fixed_grid_fallback(page_h, page_w)


def _merge_overlapping_boxes(boxes, overlap_thresh=0.5):
    """Remove boxes that significantly overlap another larger box."""
    if not boxes:
        return boxes
    # Sort by area descending
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for kx, ky, kw, kh in kept:
            # Compute intersection
            ix = max(x1, kx)
            iy = max(y1, ky)
            iw = min(x1 + w1, kx + kw) - ix
            ih = min(y1 + h1, ky + kh) - iy
            if iw > 0 and ih > 0:
                inter_area = iw * ih
                small_area = min(w1 * h1, kw * kh)
                if inter_area / small_area > overlap_thresh:
                    dominated = True
                    break
        if not dominated:
            kept.append(box)
    return kept


def _sort_boxes(boxes, row_tolerance_pct=0.04):
    """Sort boxes in reading order: top→bottom, left→right within each row."""
    if not boxes:
        return boxes
    # Estimate row height to group cards in the same row
    heights = [h for (_, _, _, h) in boxes]
    avg_h = np.median(heights) if heights else 100
    tolerance = avg_h * row_tolerance_pct * 10  # group within ~40% of card height

    # Group into rows
    rows = []
    remaining = sorted(boxes, key=lambda b: b[1])  # sort by y
    while remaining:
        row_y = remaining[0][1]
        row = [b for b in remaining if abs(b[1] - row_y) < tolerance]
        rows.append(sorted(row, key=lambda b: b[0]))  # sort by x within row
        remaining = [b for b in remaining if b not in row]

    return [box for row in rows for box in row]


def _fixed_grid_fallback(page_h, page_w, cols=3):
    """
    Fallback: divide page into a 3-column fixed grid.
    Tries to auto-detect number of rows from 5 to 8.
    """
    header_pct = 0.10  # top 10% is header
    footer_pct = 0.05  # bottom 5% is footer

    y_start = int(page_h * header_pct)
    y_end   = int(page_h * (1 - footer_pct))
    grid_h  = y_end - y_start

    col_w = page_w // cols

    # Try to guess rows based on aspect ratio (standard card is ~landscape in a 3-col layout)
    # Each card is typically ~150-200px tall at 300 DPI → ~6-8 rows
    for rows in [6, 7, 8, 5]:
        row_h = grid_h // rows
        aspect = col_w / row_h
        if 1.0 < aspect < 2.5:  # reasonable landscape card
            break

    boxes = []
    for row in range(rows):
        for col in range(cols):
            x = col * col_w
            y = y_start + row * row_h
            boxes.append((x, y, col_w, row_h))

    return boxes


# ─────────────────────────────────────────────
#  STEP 3: Preprocess & OCR Each Card
# ─────────────────────────────────────────────
def preprocess_card(card_bgr):
    """Preprocess card image for Tamil OCR."""
    # Convert to grayscale
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale 2x for better OCR accuracy on small text
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold for clean black-on-white text
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10
    )
    return processed


def ocr_card(card_bgr):
    """Run Tesseract OCR on a single voter card image. Returns raw text."""
    try:
        processed = preprocess_card(card_bgr)
        text = pytesseract.image_to_string(processed, config=TESS_CONFIG)
        return text.strip()
    except Exception as e:
        return ""


# ─────────────────────────────────────────────
#  STEP 4: Parse Tamil Fields
# ─────────────────────────────────────────────
def _clean(text):
    """Strip whitespace and remove label residue."""
    if not text:
        return None
    text = re.sub(r'[\n\r]+', ' ', text).strip()
    text = re.sub(r'\s{2,}', ' ', text)
    return text if len(text) > 1 else None


def parse_tamil_card(raw_text: str, card_bgr=None) -> dict:
    """
    Extract structured voter fields from raw OCR text using Tamil regex patterns.
    """
    record = {
        "serial_number": None,
        "epic_id":        None,
        "name":           None,
        "relation_type":  None,
        "relation_name":  None,
        "house_number":   None,
        "age":            None,
        "gender":         None,
    }

    if not raw_text:
        return record

    # --- Serial Number ---
    m = P_SERIAL.search(raw_text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 9999:
            record["serial_number"] = str(val)

    # --- EPIC ID ---
    m = P_EPIC.search(raw_text)
    if m:
        record["epic_id"] = m.group(1)

    # --- Name ---
    m = P_NAME.search(raw_text)
    if m:
        record["name"] = _clean(m.group(1))

    # --- Relation (Father / Husband / Mother) ---
    m = P_FATHER.search(raw_text)
    if m:
        record["relation_type"] = "Father"
        record["relation_name"] = _clean(m.group(1))
    elif not record.get("relation_name"):
        m = P_HUSBAND.search(raw_text)
        if m:
            record["relation_type"] = "Husband"
            record["relation_name"] = _clean(m.group(1))
    if not record.get("relation_name"):
        m = P_MOTHER.search(raw_text)
        if m:
            record["relation_type"] = "Mother"
            record["relation_name"] = _clean(m.group(1))

    # --- House Number ---
    m = P_HOUSE.search(raw_text)
    if m:
        record["house_number"] = m.group(1).strip()

    # --- Age ---
    m = P_AGE.search(raw_text)
    if m:
        try:
            age = int(m.group(1))
            if 18 <= age <= 120:
                record["age"] = age
        except:
            pass

    # --- Gender ---
    if P_GENDER_F.search(raw_text):
        record["gender"] = "பெண்"       # Female
    elif P_GENDER_M.search(raw_text):
        record["gender"] = "ஆண்"        # Male

    return record


# ─────────────────────────────────────────────
#  STEP 5: Process One PDF
# ─────────────────────────────────────────────
def process_pdf(pdf_path: str, writer, processed_set: set):
    """Process one PDF file end-to-end and write records to CSV."""
    filename = os.path.basename(pdf_path)

    if RESUME and filename in processed_set:
        print(f"  [SKIP] {filename} (already done)")
        return 0

    print(f"  [START] {filename}")
    total_records = 0
    t0 = time.time()

    try:
        for page_num, pil_page in pdf_to_page_images(pdf_path, dpi=DPI):
            # Convert PIL → numpy BGR for OpenCV
            img_np  = np.array(pil_page.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Detect card boxes
            boxes = detect_voter_cards(pil_page)

            for card_idx, (x, y, w, h) in enumerate(boxes, start=1):
                # Crop card with small padding
                pad = 4
                cx = max(0, x + pad)
                cy = max(0, y + pad)
                cw = min(img_bgr.shape[1], x + w - pad)
                ch = min(img_bgr.shape[0], y + h - pad)

                card_bgr = img_bgr[cy:ch, cx:cw]
                if card_bgr.size == 0:
                    continue

                # OCR
                raw_text = ocr_card(card_bgr)

                # Parse
                record = parse_tamil_card(raw_text, card_bgr)

                # Add metadata
                record["source_file"] = filename
                record["page_number"] = page_num
                record["card_index"]  = card_idx

                # Write to CSV
                with _csv_lock:
                    writer.writerow([record.get(c, "") for c in CSV_COLUMNS])
                    total_records += 1

    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"  [DONE ] {filename} — {total_records} records in {elapsed:.1f}s")
    return total_records


# ─────────────────────────────────────────────
#  MAIN — Parallel batch processing
# ─────────────────────────────────────────────
def get_processed_files(output_csv):
    """Read the output CSV to get set of already-processed filenames (for resume)."""
    processed = set()
    if not os.path.exists(output_csv):
        return processed
    try:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) > 8:
                    processed.add(row[8])  # source_file column
    except:
        pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="Tamil Electoral Roll OCR Pipeline")
    parser.add_argument("--test",  action="store_true", help="Test on first 2 PDFs only")
    parser.add_argument("--file",  type=str,            help="Process a single PDF file")
    parser.add_argument("--dpi",   type=int, default=DPI, help=f"DPI for PDF render (default {DPI})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    dpi     = args.dpi
    workers = args.workers

    print("=" * 60)
    print("  Tamil Electoral Roll — Box-Cut OCR Pipeline")
    print("=" * 60)

    # Gather PDF files
    if args.file:
        pdf_files = [args.file]
    else:
        pdf_files = sorted(Path(PDF_FOLDER).glob("*.pdf"))
        pdf_files = [str(p) for p in pdf_files]

    if args.test:
        pdf_files = pdf_files[:2]
        print(f"[TEST MODE] Processing {len(pdf_files)} PDF(s)")
    else:
        print(f"[FULL MODE] Processing {len(pdf_files)} PDF(s)")

    if not pdf_files:
        print("No PDFs found!")
        return

    # Prepare output CSV
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    processed_set = get_processed_files(OUTPUT_CSV)
    print(f"Already processed: {len(processed_set)} files")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()

    total_records = 0
    start_time = time.time()

    # Open CSV once and use threading for parallel processing
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(CSV_COLUMNS)

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to share writer + lock
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_pdf, pdf, writer, processed_set): pdf
                for pdf in pdf_files
            }
            for future in as_completed(futures):
                try:
                    total_records += future.result()
                except Exception as e:
                    print(f"[FATAL] {futures[future]}: {e}")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  COMPLETE: {total_records} records extracted")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
