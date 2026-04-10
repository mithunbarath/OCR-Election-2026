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
# And ensure you install the standard English pack (default).
# If you install it in the standard location, the path below will work.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OUTPUT_ALL_CSV = "voters_extracted_tess.csv"
OUTPUT_REVIEW_CSV = "voters_review_tess.csv"

# Regex Parsing Patterns (Tesseract output can be messy so we make them flexible)
# We will use inline regexes inside the function for more robust filtering.

def parse_tesseract_text(text):
    """
    Takes the raw string from Tesseract and uses robust Regex heuristics to pull out fields.
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
    
    # Clean text to single line for global regexes
    full_text = text.replace('\n', ' ')
    
    # 1. EPIC ID (Account for wide spacing like 'A B C 1 2 3 4 5 6 7')
    epic_match = re.search(r'([A-Z]{3}\s*(?:\d\s*){7})', full_text, re.IGNORECASE)
    if epic_match:
        record["epic_id"] = epic_match.group(1).replace(" ", "").upper()
    else:
        epic_match2 = re.search(r'([A-Z]{3}[0-9]{7})', full_text, re.IGNORECASE)
        if epic_match2:
            record["epic_id"] = epic_match2.group(1).upper()
            
    # 2. Age
    age_match = re.search(r'Age\s*[^0-9]*(\d{2,3})', full_text, re.IGNORECASE)
    if age_match:
        try:
            record["age"] = int(age_match.group(1))
        except:
            pass
            
    # 3. Gender
    gender_match = re.search(r'Gender\s*[^A-Za-z]*(Male|Female|Third\s*Gender|M|F)', full_text, re.IGNORECASE)
    if gender_match:
        g = gender_match.group(1).upper()
        if g.startswith('M'): record["gender"] = "Male"
        elif g.startswith('F'): record["gender"] = "Female"
        else: record["gender"] = "Third Gender"
        
    # 4. House Number
    house_match = re.search(r'House\s*N[ou]?[a-z]*\s*[^A-Za-z0-9]*([A-Za-z0-9/\\\-]+)', full_text, re.IGNORECASE)
    if house_match:
        val = house_match.group(1).strip()
        val = re.sub(r'[^\w/\\-]', '', val) # Keep alphanumeric, /, \ and -
        if val:
            record["house_number"] = val
            
    # 5. Name & Relation Logic
    lines = [L.strip() for L in text.split('\n') if L.strip()]
    
    relation_keywords = ["FATHER", "HUSBAND", "MOTHER", "WIFE"]
    
    for i, line in enumerate(lines):
        upper_line = line.upper()
        
        # Primary Name
        if not any(r in upper_line for r in relation_keywords):
            name_match = re.search(r'(?:NAME|NANE|WAME|MAME|ELECTOR)\s*[:;-]?\s*(.*)', line, re.IGNORECASE)
            if name_match:
                val = name_match.group(1)
                clean_name = re.sub(r'[^A-Za-z\s\.]', '', val).strip()
                clean_name = re.sub(r'^(S\s*NAME|S\s*NANE|NAME|NANE|MAME|WAME)\s*', '', clean_name, flags=re.IGNORECASE).strip()
                clean_name = re.sub(r'\s+', ' ', clean_name)
                if len(clean_name) >= 2:
                    record["name"] = clean_name
                    
        # Relation
        for rel in relation_keywords:
            if rel in upper_line:
                rel_match = re.search(rel + r'\s*[:;-]?\s*(.*)', line, re.IGNORECASE)
                if rel_match:
                    val = rel_match.group(1)
                    clean_rel_name = re.sub(r'[^A-Za-z\s\.]', '', val).strip()
                    clean_rel_name = re.sub(r'^(S\s*NAME|S\s*NANE|NAME|NANE|MAME|WAME|S\s*MAME|S\s*WAME)\s*', '', clean_rel_name, flags=re.IGNORECASE).strip()
                    clean_rel_name = re.sub(r'\s+', ' ', clean_rel_name)
                    
                    if len(clean_rel_name) >= 2:
                        record["relation_name"] = clean_rel_name
                        if rel == "WIFE":
                            record["relation_type"] = "Husband"
                        else:
                            record["relation_type"] = rel.capitalize()
                            
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
        print("Please install Tesseract-OCR for Windows!")
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
                # Denoise / thresholding to help Tesseract read English text
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Run PyTesseract (English language)
                try:
                    text = pytesseract.image_to_string(gray, lang='eng')
                except Exception as e:
                    print(f"    [!] Tesseract Error: {e}")
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
