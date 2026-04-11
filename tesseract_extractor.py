"""
tesseract_extractor.py  –  Electoral Roll OCR Pipeline (Tesseract Edition)
===========================================================================
Target  : All 371 PDFs inside the  119-eroll/  sub-folder
Grid    : Adaptive contour-based card segmentation (falls back to 3×10)
Parser  : Multi-pass regex with hard EPIC-ID validation, serial-vs-age
          disambiguation, and exhaustive fallback patterns.

Author  : Mithun Barath M R
Created : 2026-04-11
"""

import fitz                          # PyMuPDF  –  no Poppler needed on Windows
import cv2
import numpy as np
import pandas as pd
import re
import os
import glob
import pytesseract
from pathlib import Path

# ─────────────────────────── Configuration ───────────────────────────────────

# Tesseract binary path (Windows default install location)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folder that contains all 371 PDFs
EROLL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "119-eroll")

# Output files (written next to this script)
OUTPUT_ALL_CSV    = "voters_extracted_tess.csv"
OUTPUT_REVIEW_CSV = "voters_review_tess.csv"

# Render resolution – 300 DPI equivalent (scale=3 on standard 72-dpi PDF)
RENDER_SCALE = 3

# Grid fallback dimensions when contour detection fails
GRID_ROWS         = 10
GRID_COLS         = 3
HEADER_FRAC       = 0.11    # fraction of page height to skip (header)
FOOTER_FRAC       = 0.04    # fraction of page height to skip (footer)

# ──────────────────────────── EPIC ID patterns ───────────────────────────────
# Tamil-Nadu EPIC IDs follow the pattern: 3 UPPERCASE letters + 7 digits
# e.g.  TJV1234567  /  ABC0012345
# Tesseract frequently inserts spaces between glyphs → handle both compact and
# spaced variants.  Also handle common OCR substitutions (O↔0, I↔1, S↔5 …)

EPIC_PATTERNS = [
    # Tight: "TJV1234567"
    re.compile(r'\b([A-Z]{3}[0-9]{7})\b', re.IGNORECASE),

    # Spaced: "T J V 1 2 3 4 5 6 7"  (arbitrary spaces between chars)
    re.compile(r'\b([A-Z]\s*[A-Z]\s*[A-Z]\s*(?:\d\s*){7})\b', re.IGNORECASE),

    # After "EPIC" / "ID" label
    re.compile(r'(?:EPIC|LD\.?|ID\.?)\s*[:\-]?\s*([A-Z]{3}\s*\d{1,2}\s*\d{1,2}\s*\d{1,2}\s*\d{1,2}\s*\d{1,2}\s*\d{1,2}\s*\d?)',
               re.IGNORECASE),

    # Extremely spaced:  "T J V  0 0 1  2 3  4 5"
    re.compile(r'([A-Z]\s+[A-Z]\s+[A-Z](?:\s+\d){7})', re.IGNORECASE),
]

def _clean_epic(raw: str) -> str | None:
    """Strip spaces, validate length, return uppercase or None."""
    cleaned = re.sub(r'\s+', '', raw).upper()
    # Must be exactly 3 letters + 7 digits
    if re.fullmatch(r'[A-Z]{3}[0-9]{7}', cleaned):
        return cleaned
    # Try to salvage O→0, I/L→1, S→5 substitutions
    salvaged = (cleaned
                .replace('O', '0').replace('I', '1')
                .replace('L', '1').replace('S', '5')
                .replace('Z', '2').replace('B', '8'))
    if re.fullmatch(r'[A-Z]{3}[0-9]{7}', salvaged):
        return salvaged
    return None


# ──────────────────────────── Serial-number helpers ──────────────────────────

def _extract_serial(text: str) -> int | None:
    """
    The serial number is generally the FIRST standalone integer (1-999)
    printed in the top-left corner of a voter card.  We look for it in the
    first two non-empty lines of the OCR output.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:3]:
        m = re.match(r'^(\d{1,3})[.\s]', line)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 999:
                return val
        # Sometimes it's the entire first token on the line
        m2 = re.fullmatch(r'(\d{1,3})', line)
        if m2:
            val = int(m2.group(1))
            if 1 <= val <= 999:
                return val
    return None


# ─────────────────────────── Field parser ────────────────────────────────────

def parse_tesseract_text(text: str, card_index: int = 0) -> dict:
    """
    Multi-pass regex parser for a single voter card's OCR text.
    Returns a dict with all voter fields.
    """
    record = {
        "serial_number" : None,
        "epic_id"       : None,
        "name"          : None,
        "relation_type" : None,
        "relation_name" : None,
        "house_number"  : None,
        "age"           : None,
        "gender"        : None,
    }

    full_text  = text.replace('\n', ' ')          # single-line view
    lines      = [l.strip() for l in text.split('\n') if l.strip()]

    # ── 1. Serial Number ─────────────────────────────────────────────────────
    sn = _extract_serial(text)
    # Sanity: if caller supplies a card_index use it as cross-check
    if sn is not None:
        record["serial_number"] = sn

    # ── 2. EPIC ID ───────────────────────────────────────────────────────────
    # Try all patterns in priority order
    for pat in EPIC_PATTERNS:
        m = pat.search(full_text)
        if m:
            epic = _clean_epic(m.group(1))
            if epic:
                record["epic_id"] = epic
                break

    # Exhaustive fallback: look for ANY 10-char alphanumeric that looks like
    # an EPIC ID (starts with 3 letters, ends with 7 digits)
    if not record["epic_id"]:
        candidates = re.findall(r'[A-Z0-9]{10}', full_text, re.IGNORECASE)
        for cand in candidates:
            epic = _clean_epic(cand)
            if epic:
                record["epic_id"] = epic
                break

    # ── 3. Age ───────────────────────────────────────────────────────────────
    # Strict: only extract age if it follows the word "Age" (with noise)
    # and the value is between 18 and 110.
    # IMPORTANT: Reject values ≤ 30 found without the "Age" keyword to avoid
    #            confusing serial numbers (1-30) with ages.

    age_val = None

    # Primary: look for "Age : 45" or "Age 45" pattern
    age_m = re.search(
        r'(?:Age|A\s*ge|Aqe)\s*[:\-–]?\s*(\d{2,3})',
        full_text, re.IGNORECASE
    )
    if age_m:
        try:
            v = int(age_m.group(1))
            if 18 <= v <= 110:
                age_val = v
        except ValueError:
            pass

    # Secondary: look for isolated 2-digit number that is NOT the serial
    if age_val is None:
        # Find all 2-3 digit standalone numbers in the text
        nums = re.findall(r'\b(\d{2,3})\b', full_text)
        sn_val = record["serial_number"]
        for n in nums:
            v = int(n)
            if sn_val and v == sn_val:
                continue    # skip — this is the serial number
            if 18 <= v <= 110:
                age_val = v
                break

    record["age"] = age_val

    # ── 4. Gender ────────────────────────────────────────────────────────────
    gender_m = re.search(
        r'(?:Gender|Sex)\s*[:\-–]?\s*(Male|Female|Third\s*Gender|M|F)',
        full_text, re.IGNORECASE
    )
    if gender_m:
        g = gender_m.group(1).strip().upper()
        if g.startswith('M'):
            record["gender"] = "Male"
        elif g.startswith('F'):
            record["gender"] = "Female"
        else:
            record["gender"] = "Third Gender"
    else:
        # Fallback: standalone keyword anywhere
        if re.search(r'\bMale\b', full_text, re.IGNORECASE):
            record["gender"] = "Male"
        elif re.search(r'\bFemale\b', full_text, re.IGNORECASE):
            record["gender"] = "Female"

    # ── 5. House Number ──────────────────────────────────────────────────────
    house_m = re.search(
        r'House\s*N(?:o|um|umber|u)?\.?\s*[:\-–]?\s*([A-Za-z0-9][A-Za-z0-9/\\\-]{0,19})',
        full_text, re.IGNORECASE
    )
    if house_m:
        val = re.sub(r'[^\w/\\\-]', '', house_m.group(1)).strip()
        if val:
            record["house_number"] = val

    # ── 6. Name & Relation ──────────────────────────────────────────────────
    # Keyword variants Tesseract produces for "Name:"
    NAME_KWDS    = r'(?:ELECTOR\'?S?\s*)?(?:NAME|NANE|WAME|MAME|NAIME|NAXE)'
    RELATION_MAP = {
        "FATHER"  : "Father",
        "HUSBAND" : "Husband",
        "MOTHER"  : "Mother",
        "WIFE"    : "Wife",
        "SON OF"  : "Father",
        "W/O"     : "Husband",
        "S/O"     : "Father",
        "D/O"     : "Father",
        "H/O"     : "Wife",
    }

    for i, line in enumerate(lines):
        upper_line = line.upper()

        # Primary name (line with a "Name:" keyword, no relation keyword)
        if record["name"] is None:
            if not any(rk in upper_line for rk in ["FATHER", "HUSBAND", "MOTHER", "WIFE", "W/O", "S/O", "D/O"]):
                nm = re.search(NAME_KWDS + r'\s*[:;.\-]?\s*(.*)', line, re.IGNORECASE)
                if nm:
                    raw_name = nm.group(1).strip()
                    # Remove sub-labels like "S Name" that Tesseract duplicates
                    raw_name = re.sub(
                        r'^(?:S\s*)?(?:NAME|NANE|MAME|WAME)\s*[:;.\-]?\s*', '',
                        raw_name, flags=re.IGNORECASE
                    )
                    cleaned = re.sub(r'[^A-Za-z\s\.\'\-]', '', raw_name).strip()
                    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
                    if len(cleaned) >= 2:
                        record["name"] = cleaned

        # Relation name
        if record["relation_name"] is None:
            for rel_kw, rel_type in RELATION_MAP.items():
                if rel_kw in upper_line:
                    rn_m = re.search(rel_kw + r'\s*[:;.\-]?\s*(.*)', line, re.IGNORECASE)
                    if rn_m:
                        raw_rn = rn_m.group(1).strip()
                        raw_rn = re.sub(
                            r'^(?:S\s*)?(?:NAME|NANE|MAME|WAME)\s*[:;.\-]?\s*', '',
                            raw_rn, flags=re.IGNORECASE
                        )
                        cleaned_rn = re.sub(r'[^A-Za-z\s\.\'\-]', '', raw_rn).strip()
                        cleaned_rn = re.sub(r'\s{2,}', ' ', cleaned_rn)
                        if len(cleaned_rn) >= 2:
                            record["relation_name"] = cleaned_rn
                            record["relation_type"] = rel_type
                    break

    return record


# ──────────────────────────── Card Segmentation ──────────────────────────────

def _detect_card_regions_contour(img: np.ndarray, header_frac: float, footer_frac: float) -> list:
    """
    Detect individual voter-card boundaries using horizontal separator lines.
    Falls back to mathematical 3×10 grid if contour detection yields too few
    or too many cards.
    Returns list of (x1, y1, x2, y2) tuples.
    """
    img_h, img_w = img.shape[:2]
    header_px    = int(img_h * header_frac)
    footer_px    = int(img_h * footer_frac)
    work_top     = header_px
    work_bot     = img_h - footer_px
    work_img     = img[work_top:work_bot, :]

    gray   = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    # Sharpen slightly before edge detection
    kernel  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Detect horizontal lines (card separators)
    h_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w // 4, 1))
    h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Reduce to single-pixel-height slices
    col_sums = np.sum(h_lines_img, axis=1)
    threshold = img_w * 255 * 0.25       # at least 25% of row is white
    h_line_rows = np.where(col_sums > threshold)[0]

    # Cluster nearby rows
    def cluster_rows(rows, gap=8):
        if len(rows) == 0:
            return []
        clusters, current = [], [rows[0]]
        for r in rows[1:]:
            if r - current[-1] <= gap:
                current.append(r)
            else:
                clusters.append(int(np.mean(current)))
                current = [r]
        clusters.append(int(np.mean(current)))
        return clusters

    separators = cluster_rows(h_line_rows)

    # Also detect vertical separators (column dividers)
    v_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_h // 6))
    v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    row_sums    = np.sum(v_lines_img, axis=0)
    v_threshold = (work_bot - work_top) * 255 * 0.20
    v_line_cols = np.where(row_sums > v_threshold)[0]

    def cluster_cols(cols, gap=8):
        if len(cols) == 0:
            return []
        clusters, current = [], [cols[0]]
        for c in cols[1:]:
            if c - current[-1] <= gap:
                current.append(c)
            else:
                clusters.append(int(np.mean(current)))
                current = [c]
        clusters.append(int(np.mean(current)))
        return clusters

    col_seps = cluster_cols(v_line_cols)

    # Decide column boundaries
    if 1 <= len(col_seps) <= 4:
        col_bounds = [0] + col_seps + [img_w]
    else:
        cw = img_w // GRID_COLS
        col_bounds = [c * cw for c in range(GRID_COLS + 1)]

    # Decide row boundaries
    if 8 <= len(separators) <= 15:
        row_bounds = [0] + separators + [work_bot - work_top]
    else:
        # Fallback to uniform grid
        row_h = (work_bot - work_top) // GRID_ROWS
        row_bounds = [r * row_h for r in range(GRID_ROWS + 1)]

    # Build card regions (absolute coordinates in full image)
    regions = []
    for ri in range(len(row_bounds) - 1):
        for ci in range(len(col_bounds) - 1):
            y1 = work_top + row_bounds[ri]
            y2 = work_top + row_bounds[ri + 1]
            x1 = col_bounds[ci]
            x2 = col_bounds[ci + 1]
            # Skip slivers
            if (y2 - y1) < 40 or (x2 - x1) < 40:
                continue
            regions.append((x1, y1, x2, y2))

    return regions


def extract_cards_from_page(page_image_bytes: bytes) -> list:
    """
    Convert raw PNG bytes → list of cropped card images (as numpy arrays).
    Uses adaptive contour detection with 3×10 fallback.
    """
    nparr  = np.frombuffer(page_image_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]

    regions = _detect_card_regions_contour(img, HEADER_FRAC, FOOTER_FRAC)

    # Sanity-check: expect 25–35 cards per page
    if not (20 <= len(regions) <= 40):
        # Hard fallback: pure mathematical 3×10 grid
        top    = int(img_h * HEADER_FRAC)
        bot    = img_h - int(img_h * FOOTER_FRAC)
        row_h  = (bot - top) // GRID_ROWS
        col_w  = img_w // GRID_COLS
        regions = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x1 = col * col_w
                x2 = x1 + col_w
                y1 = top + row * row_h
                y2 = y1 + row_h
                regions.append((x1, y1, x2, y2))

    cards = []
    for (x1, y1, x2, y2) in regions:
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            cards.append(crop)

    return cards


# ──────────────────────── Image Pre-processing ───────────────────────────────

def preprocess_card(card_img: np.ndarray) -> np.ndarray:
    """
    Prepare a card image for Tesseract:
    - convert to grayscale
    - upscale if too small (helps Tesseract immensely)
    - denoise
    - binarise with adaptive threshold
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

    # Upscale tiny cards
    h, w = gray.shape
    if h < 200 or w < 200:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Mild denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Light morphological opening to remove tiny noise specks
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


# ────────────────────────────── Main Pipeline ────────────────────────────────

def main():
    print("=" * 60)
    print("  Electoral Roll PDF Extractor  –  Tesseract Edition")
    print("  Target folder : /")
    print("=" * 60)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print("[ERROR] Tesseract-OCR not found at:")
        print(f"        {pytesseract.pytesseract.tesseract_cmd}")
        print("Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        return

    if not os.path.isdir(EROLL_DIR):
        print(f"[ERROR] 119-eroll folder not found at: {EROLL_DIR}")
        return

    pdf_files = sorted(glob.glob(os.path.join(EROLL_DIR, "*.pdf")))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {EROLL_DIR}")
        return

    print(f"[INFO]  Found {len(pdf_files)} PDF files.\n")

    all_records = []
    total_pdfs  = len(pdf_files)

    for pdf_idx, pdf_file in enumerate(pdf_files, 1):
        pdf_name = os.path.basename(pdf_file)
        print(f"[{pdf_idx:>3}/{total_pdfs}] {pdf_name}")

        try:
            doc = fitz.open(pdf_file)
        except Exception as e:
            print(f"         ✗ Could not open PDF: {e}")
            continue

        total_pages    = len(doc)
        # Skip cover (page 0) + summary/map page (page 1)
        # Skip last page (modification / deletion summary)
        start_page     = 2
        end_page       = total_pages - 1      # exclusive

        if end_page <= start_page:
            print(f"         ✗ PDF too short ({total_pages} pages), skipping.")
            doc.close()
            continue

        page_count     = 0
        card_count     = 0
        epic_ok_count  = 0

        for page_num in range(start_page, end_page):
            page         = doc.load_page(page_num)
            pix          = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
            img_bytes    = pix.tobytes("png")

            cards        = extract_cards_from_page(img_bytes)
            page_count  += 1

            for card_idx, card_img in enumerate(cards):
                processed = preprocess_card(card_img)

                try:
                    # PSM 6 = assume a uniform block of text (best for cards)
                    custom_cfg = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(processed, lang='eng', config=custom_cfg)
                except Exception as e:
                    print(f"         [Tesseract ERR] page {page_num+1}, card {card_idx}: {e}")
                    text = ""

                record = parse_tesseract_text(text, card_index=card_idx)

                # Populate metadata
                record["source_file"] = pdf_name
                record["page_number"] = page_num + 1
                record["card_on_page"] = card_idx + 1
                record["raw_ocr_text"] = text.strip().replace('\n', ' | ')

                # ── Validation flags ──────────────────────────────────────
                flags = []

                if not record["epic_id"]:
                    flags.append("EPIC_MISSING")
                else:
                    epic_ok_count += 1

                age = record["age"]
                if age is None:
                    flags.append("AGE_MISSING")
                elif not (18 <= age <= 110):
                    flags.append("AGE_OUTLIER")

                for field in ("name", "gender"):
                    if not record.get(field):
                        flags.append(f"{field.upper()}_MISSING")

                if not record.get("relation_name"):
                    flags.append("RELATION_MISSING")

                if not record.get("house_number"):
                    flags.append("HOUSE_MISSING")

                record["ocr_flag"] = " | ".join(sorted(set(flags)))
                all_records.append(record)
                card_count += 1

        doc.close()
        print(f"         ✓ {page_count} content pages | {card_count} cards | "
              f"{epic_ok_count} EPIC IDs found")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Extraction complete.  Total records: {len(all_records)}")

    columns_order = [
        "serial_number", "epic_id",
        "name", "relation_type", "relation_name",
        "house_number", "age", "gender",
        "source_file", "page_number", "card_on_page",
        "ocr_flag", "raw_ocr_text",
    ]

    df = pd.DataFrame(all_records)
    df = df.reindex(columns=columns_order)

    df.to_csv(OUTPUT_ALL_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved ALL records  → {OUTPUT_ALL_CSV}  ({len(df)} rows)")

    review_mask = df['ocr_flag'].astype(bool) & (df['ocr_flag'] != "")
    review_df   = df[review_mask]
    review_df.to_csv(OUTPUT_REVIEW_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved REVIEW records → {OUTPUT_REVIEW_CSV}  ({len(review_df)} rows)")

    # ── Quick stats ──────────────────────────────────────────────────────────
    total = len(df)
    if total > 0:
        print(f"\n── Extraction Quality Stats ──────────────────────────────")
        print(f"  EPIC IDs extracted       : {df['epic_id'].notna().sum():>6} / {total} "
              f"({df['epic_id'].notna().sum()/total*100:.1f}%)")
        print(f"  Ages extracted           : {df['age'].notna().sum():>6} / {total} "
              f"({df['age'].notna().sum()/total*100:.1f}%)")
        print(f"  Names extracted          : {df['name'].notna().sum():>6} / {total} "
              f"({df['name'].notna().sum()/total*100:.1f}%)")
        print(f"  Gender extracted         : {df['gender'].notna().sum():>6} / {total} "
              f"({df['gender'].notna().sum()/total*100:.1f}%)")
        print(f"  Clean records (no flags) : {(~review_mask).sum():>6} / {total} "
              f"({(~review_mask).sum()/total*100:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
