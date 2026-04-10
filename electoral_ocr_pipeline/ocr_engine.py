import cv2
import pytesseract
import easyocr
import re
import os

# Initialize EasyOCR reader (Tamil and English). 
# Note: gpu=False ensures it runs everywhere, but set to True if a CUDA GPU is available.
reader = easyocr.Reader(['en', 'ta'], gpu=False)

# Configure Tesseract path for Windows
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def preprocess_image(img_bgr, threshold_type='adaptive', enlarge=True):
    """
    General image preprocessing to enhance text logic for OCR engines.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if enlarge:
        # Upscale to assist in reading small text
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
    if threshold_type == 'adaptive':
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif threshold_type == 'otsu':
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        processed = gray
        
    # Light morphological transformations could be added if noise is high, 
    # but simple adaptive thresholding works best for ECI pdfs.
    return processed

def run_hybrid_ocr(rois):
    """
    Takes the Regions Of Interest (ROIs), processes them, and runs the optimal OCR strategy.
    """
    extracted_data = {
        "raw_serial": "",
        "raw_epic": "",
        "raw_name_rel": "",
        "raw_bottom": ""
    }
    
    # --- 1. Serial Number (Tesseract) ---
    serial_img = preprocess_image(rois["serial_no"], threshold_type='otsu', enlarge=True)
    # Whitelist to only digits, since serials are usually purely numeric or small alphanumeric
    custom_config_serial = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    extracted_data["raw_serial"] = pytesseract.image_to_string(serial_img, config=custom_config_serial).strip()
    
    # --- 2. EPIC ID (Tesseract) ---
    epic_img = preprocess_image(rois["epic_id"], threshold_type='otsu', enlarge=True)
    custom_config_epic = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    extracted_data["raw_epic"] = pytesseract.image_to_string(epic_img, config=custom_config_epic).strip()
    
    # --- 3. Name & Relation (EasyOCR) ---
    # EasyOCR takes BGR/RGB array directly and handles its own processing, but enlarging helps.
    name_img = cv2.resize(rois["name_rel"], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    name_results = reader.readtext(name_img, detail=0)
    extracted_data["raw_name_rel"] = " \n ".join(name_results)
    
    # --- 4. Bottom Details (EasyOCR/Tesseract Hybrid fallback) ---
    bot_img = cv2.resize(rois["bottom_details"], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bot_results = reader.readtext(bot_img, detail=0)
    extracted_data["raw_bottom"] = " \n ".join(bot_results)
    
    return extracted_data
