import fitz  # We use PyMuPDF because it avoids installing Poppler on Windows!
import cv2
import numpy as np
import pandas as pd
import re
import os
import glob
import pytesseract
import time

# ================= Configuration =================
# VERY IMPORTANT: You MUST install Tesseract-OCR on Windows.
# Download it from: https://github.com/UB-Mannheim/tesseract/wiki
# And ensure you check the "Tamil" language pack during installation.
# If you install it in the standard location, the path below will work.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OUTPUT_ALL_CSV = "voters_extracted_tess.csv"
OUTPUT_REVIEW_CSV = "voters_review_tess.csv"

# Regex Parsing Patterns (Tesseract output can be messy so we make them flexible)
EPIC_PATTERN = re.compile(r'([A-Z]{3}[0-9]{7})')
AGE_PATTERN = re.compile(r'Age\s*[:;-]?\s*(\d{2,3})', re.IGNORECASE)
GENDER_PATTERN = re.compile(r'Gender\s*[:;-]?\s*(Male|Female|Third\s*Gender|M|F)', re.IGNORECASE)
HOUSE_PATTERN = re.compile(r'House\s*N(?:o|umber)?\s*[:;-]?\s*([A-Za-z0-9/\\\-]+)', re.IGNORECASE)

# Keywords to find Names
RELATION_TYPES = ["Father", "Husband", "Mother"]

def parse_tesseract_text(text):
    """
    Takes the raw string from Tesseract and uses Regex heuristics to pull out fields.
    """
    record = {
        "epic_id": None,
        "name": None,
        "relation_type": None,
        "relation_name": None,
        "house_number": None,
        "age": None,
        "gender": None
    }
    
    # Clean up empty lines
    lines = [L.strip() for L in text.split('\n') if L.strip()]
    full_text = " ".join(lines)
    
    # 1. EPIC ID
    epic_match = EPIC_PATTERN.search(full_text)
    if epic_match:
        record["epic_id"] = epic_match.group(1)
        
    # 2. Age
    age_match = AGE_PATTERN.search(full_text)
    if age_match:
        try:
            record["age"] = int(age_match.group(1))
        except:
            pass
            
    # 3. Gender
    gender_match = GENDER_PATTERN.search(full_text)
    if gender_match:
        g = gender_match.group(1).upper()
        if g.startswith('M'): record["gender"] = "Male"
        elif g.startswith('F'): record["gender"] = "Female"
        else: record["gender"] = "Third Gender"
        
    # 4. House Number
    house_match = HOUSE_PATTERN.search(full_text)
    if house_match:
        record["house_number"] = house_match.group(1)
        
    # 5. Name & Relation Logic
    # Usually format is:
    # Name : John Doe
    # Father's Name : Richard Doe
    for i, line in enumerate(lines):
        upper_line = line.upper()
        
        # Primary Name
        if "NAME" in upper_line and not any(r.upper() in upper_line for r in RELATION_TYPES):
            parts = re.split(r'[:;-]', line, maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                record["name"] = parts[1].strip()
                
        # Relation
        for rel in RELATION_TYPES:
            if rel.upper() in upper_line:
                record["relation_type"] = rel
                parts = re.split(r'[:;-]', line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    record["relation_name"] = parts[1].strip()
                    
    return record


def extract_cards_from_page(page_image_bytes):
    """
    Uses mathematical slicing (3x10 grid) to crop 30 cards per page.
    """
    nparr = np.frombuffer(page_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]
    
    header_offset = int(img_h * 0.11)
    footer_offset = int(img_h * 0.04)
    grid_h = img_h - header_offset - footer_offset
    
    row_height = grid_h // 10
    col_width = img_w // 3
    
    cropped_images = []
    
    for row in range(10):
        for col in range(3):
            y_start = header_offset + (row * row_height)
            y_end = y_start + row_height
            x_start = col * col_width
            x_end = x_start + col_width
            
            crop = img[y_start:y_end, x_start:x_end]
            cropped_images.append(crop)  # keep as CV2 image for Tesseract
            
    return cropped_images

def main():
    print("=======================================")
    print(" Electoral Roll PDF Extractor (Tesseract) ")
    print("=======================================\n")
    
    # Check if Tesseract is installed where we expect it
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print("[!] ERROR: Tesseract OCR is not found at:")
        print(f"    {pytesseract.pytesseract.tesseract_cmd}")
        print("Please install Tesseract-OCR for Windows and make sure you check the Tamil Language pack!")
        return
        
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
        
    all_records = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        
        doc = fitz.open(pdf_file)
        
        for page_num in range(len(doc)):
            print(f"  -> Page {page_num+1}/{len(doc)}")
            page = doc.load_page(page_num)
            
            # High res render (300 DPI) is extremely critical for classic OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_bytes = pix.tobytes("png")
            
            cards = extract_cards_from_page(img_bytes)
            
            for i, card_img in enumerate(cards):
                # We do some quick image preprocessing before sending to Tesseract
                gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # Denoise / thresholding to help Tesseract read Tamil + English
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Run PyTesseract (eng+tam language)
                try:
                    text = pytesseract.image_to_string(gray, lang='eng+tam')
                except Exception as e:
                    print(f"    [!] Tesseract Error: Did you install the 'tam' language pack? {e}")
                    text = ""
                    
                record = parse_tesseract_text(text)
                
                # Metadata
                record["serial_number"] = None # Tesseract usually scrambles the small corner number
                record["source_file"] = pdf_file
                record["page_number"] = page_num + 1
                record["ocr_flag"] = []
                
                # Validation Logic
                if not record["epic_id"]:
                    record["ocr_flag"].append("EPIC_MISSING")
                    
                age = record["age"]
                if not isinstance(age, int) or age < 18 or age > 120:
                    record["ocr_flag"].append("AGE_OUTLIER")
                    
                # NULL checks
                for field in ["name", "relation_name", "house_number", "gender"]:
                    if not record.get(field):
                        record["ocr_flag"].append("NEEDS_REVIEW")
                        break
                        
                record["ocr_flag"] = " | ".join(set(record["ocr_flag"]))
                all_records.append(record)

        doc.close()
        
    print("\nExtraction Complete! Saving datasets...")
    
    df = pd.DataFrame(all_records)
    columns_order = [
         "serial_number", "epic_id", "name", "relation_type", "relation_name", 
         "house_number", "age", "gender", "source_file", "page_number", "ocr_flag"
    ]
    df = df.reindex(columns=columns_order)
    
    df.to_csv(OUTPUT_ALL_CSV, index=False)
    print(f"Saved all records to {OUTPUT_ALL_CSV} ({len(df)} rows)")
    
    review_df = df[df['ocr_flag'].astype(bool) & (df['ocr_flag'] != "")]
    review_df.to_csv(OUTPUT_REVIEW_CSV, index=False)
    print(f"Saved {len(review_df)} anomalous records to {OUTPUT_REVIEW_CSV}")

if __name__ == "__main__":
    main()
