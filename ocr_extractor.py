import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import re
import requests
import json
import base64
import os
import glob
import time

# ================= Configuration =================
# Set your OpenRouter API Key here
OPENROUTER_API_KEY = "sk-or-v1-..."  # <-- PASTE YOUR OPENROUTER API KEY HERE

# Recommend free vision models on OpenRouter
OPENROUTER_MODEL_NAME = "meta-llama/llama-3.2-11b-vision-instruct:free"

# Output files
OUTPUT_ALL_CSV = "voters_extracted.csv"
OUTPUT_REVIEW_CSV = "voters_review.csv"

# Regex for EPIC ID
EPIC_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{7}$')

PROMPT = """You are an OCR extraction assistant processing a single Indian Electoral Roll voter card.
The card contains bilingual text (Tamil and English). Ignore the Tamil text and extract only the English text.
Return ONLY a valid JSON object with the following schema. Do NOT include any markdown formatting or tags like ```json.
{
  "serial_number": 12,
  "epic_id": "ABC1234567",
  "name": "Voter Name",
  "relation_type": "Father", 
  "relation_name": "Father Name",
  "house_number": "12/A",
  "age": 35,
  "gender": "Male"
}
If a field is completely unreadable or missing, set its value to null.
For relation_type, ensure it is strictly "Father", "Husband", or "Mother".
"""

def extract_cards_from_page(page_image_bytes):
    """
    Given a Full A4 page image from the Electoral Roll,
    slice the image into a 3x10 grid of exactly 30 cards. 
    This robust mathematical slicing avoids OpenCV contour failure on faint PDF grid lines.
    """
    nparr = np.frombuffer(page_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]
    
    # ECI Rolls typically have a header (top ~11%) and footer (bottom ~4%).
    # We will slice the middle 85% into 10 rows and 3 columns.
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
            _, buffer = cv2.imencode('.jpg', crop)
            cropped_images.append(buffer.tobytes())
            
    return cropped_images

def extract_data_openrouter(image_bytes):
    """
    Sends the cropped card to OpenRouter API and returns parsed JSON.
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{base64_image}"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENROUTER_MODEL_NAME,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0
    }
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        reply = result['choices'][0]['message']['content']
        # Clean up if model still outputted markdown
        reply = reply.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(reply)
        return data
    except Exception as e:
        print(f"    [!] OpenRouter Extraction Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"    [!] OpenRouter Response: {response.text}")
        return None

def main():
    print("=======================================")
    print(" Electoral Roll PDF Extractor (OpenRouter) ")
    print("=======================================\n")
    
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
        
    all_records = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        
        # Open PDF
        doc = fitz.open(pdf_file)
        
        for page_num in range(len(doc)):
            print(f"  -> Page {page_num+1}/{len(doc)}")
            page = doc.load_page(page_num)
            
            # High res render (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_bytes = pix.tobytes("png")
            
            cards = extract_cards_from_page(img_bytes)
            print(f"     Found {len(cards)} individual voter cards.")
            
            for i, card_img in enumerate(cards):
                data = extract_data_openrouter(card_img)
                
                # Default empty record if failure occurs
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
                
                if isinstance(data, dict):
                    record.update(data)
                
                # Metadata
                record["source_file"] = pdf_file
                record["page_number"] = page_num + 1
                record["ocr_flag"] = []
                
                # Validation Logic
                if not record["epic_id"] or not EPIC_PATTERN.match(str(record["epic_id"])):
                    record["ocr_flag"].append("EPIC_INVALID")
                    
                age = record["age"]
                if not isinstance(age, int) or age < 18 or age > 120:
                    record["ocr_flag"].append("AGE_OUTLIER")
                    
                # NULL checks
                for field in ["name", "relation_name", "house_number", "gender"]:
                    if not record[field]:
                        record["ocr_flag"].append("NEEDS_REVIEW")
                        break
                        
                record["ocr_flag"] = " | ".join(set(record["ocr_flag"]))
                all_records.append(record)
                
                # Tiny sleep to let local GPU breathe
                time.sleep(0.5)

        doc.close()
        
    print("\nExtraction Complete! Saving datasets...")
    
    # Save to CSV
    df = pd.DataFrame(all_records)
    
    # Reorder columns explicitly per requirements
    columns_order = [
         "serial_number", "epic_id", "name", "relation_type", "relation_name", 
         "house_number", "age", "gender", "source_file", "page_number", "ocr_flag"
    ]
    df = df.reindex(columns=columns_order)
    
    df.to_csv(OUTPUT_ALL_CSV, index=False)
    print(f"Saved all records to {OUTPUT_ALL_CSV} ({len(df)} rows)")
    
    # Review CSV
    review_df = df[df['ocr_flag'].astype(bool) & (df['ocr_flag'] != "")]
    review_df.to_csv(OUTPUT_REVIEW_CSV, index=False)
    print(f"Saved {len(review_df)} anomalous records to {OUTPUT_REVIEW_CSV}")

if __name__ == "__main__":
    main()
