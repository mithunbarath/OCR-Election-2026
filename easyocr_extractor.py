import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import re
import os
import glob
import easyocr
import time

# ================= Configuration =================
OUTPUT_ALL_CSV = "voters_extracted_easyocr.csv"
OUTPUT_REVIEW_CSV = "voters_review_easyocr.csv"

def parse_easyocr_text(lines):
    """
    Takes the raw list of strings from EasyOCR and uses robust Regex heuristics to pull out fields.
    EasyOCR usually reads top-to-bottom, left-to-right.
    """
    record = {
        "serial_number": None,
        "epic_id": None,
        "name": None,
        "relation_type": None,
        "relation_name": None,
        "house_number": None,
        "age": None,
        "gender": None
    }
    
    full_text = " ".join(lines)
    
    # 1. Serial Number 
    # Usually the very first or second isolated text blurb is the serial number (1-4 digits)
    for text in lines[:3]:
        # If it's a pure number and <= 4 digits
        clean_num = re.sub(r'[^0-9]', '', text)
        if clean_num and len(clean_num) <= 4:
            record["serial_number"] = clean_num
            break

    # 1. EPIC ID
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
        val = re.sub(r'[^\w/\\-]', '', val)
        if val:
            record["house_number"] = val
            
    # 5. Name & Relation Logic
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
                    
        # Relation Name
        for rel in relation_keywords:
            if rel in upper_line:
                rel_match = re.search(rel + r'\s*[:;-]?\s*(.*)', line, re.IGNORECASE)
                if rel_match:
                    val = rel_match.group(1)
                    clean_rel_name = re.sub(r'[^A-Za-z\s\.]', '', val).strip()
                    clean_rel_name = re.sub(r'^(S\s*NAME|S\s*NANE|NAME|NANE|MAME|WAME|S\s*MAME|S\s*WAME|S\s*HAME)\s*', '', clean_rel_name, flags=re.IGNORECASE).strip()
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
    Mathematical grid slicing for ECI rolls. 30 cards per page.
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
            cropped_images.append(crop)
            
    return cropped_images

def main():
    print("=======================================")
    print(" Electoral Roll PDF Extractor (EasyOCR) ")
    print("=======================================\n")
    
    print("Initializing EasyOCR Model (This may take a minute on first run)...")
    # gpu=True will utilize Cuda if available, drastically improving speeds.
    # Otherwise it comfortably falls back to cpu.
    reader = easyocr.Reader(['en'], gpu=True)
        
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
        
    all_records = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        
        doc = fitz.open(pdf_file)
        
        for page_num in range(2, len(doc) - 1):
            print(f"  -> Page {page_num+1}/{len(doc)}")
            page = doc.load_page(page_num)
            
            # High res render (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_bytes = pix.tobytes("png")
            
            cards = extract_cards_from_page(img_bytes)
            
            for i, card_img in enumerate(cards):
                # We can feed the RGB image directly into EasyOCR
                try:
                    # detail=0 returns a simple list of text strings found in the image.
                    # paragraph=False means it keeps tight horizontal bounding boxes distinct!
                    lines = reader.readtext(card_img, detail=0, paragraph=False)
                except Exception as e:
                    print(f"    [!] EasyOCR Error: {e}")
                    lines = []
                    
                record = parse_easyocr_text(lines)
                
                # Metadata
                record["source_file"] = pdf_file
                record["page_number"] = page_num + 1
                record["ocr_flag"] = []
                
                # Validation Logic
                if not record["epic_id"]:
                    record["ocr_flag"].append("EPIC_MISSING")
                    
                age = record["age"]
                if not isinstance(age, int) or age < 18 or age > 120:
                    record["ocr_flag"].append("AGE_OUTLIER")
                    
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
