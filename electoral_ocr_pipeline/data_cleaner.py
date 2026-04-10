import re

def clean_ocr_data(raw_data):
    """
    Takes the raw OCR text outputs and applies heuristics to extract structured fields.
    """
    structured = {
        "serial_number": None,
        "epic_id": None,
        "name": None,
        "relation_type": None,
        "relation_name": None,
        "house_number": None,
        "age": None,
        "gender": None
    }
    
    # 1. Serial Number
    serial_str = re.sub(r'[^0-9]', '', raw_data["raw_serial"])
    if serial_str:
        structured["serial_number"] = serial_str
        
    # 2. EPIC ID
    epic_raw = raw_data["raw_epic"].upper().replace(" ", "")
    # Fix common OCR mistakes in EPIC (e.g., O->0, I->1) for the standard format XXX1234567
    epic_match = re.search(r'([A-Z]{3}[0-9]{7})', epic_raw)
    if epic_match:
        structured["epic_id"] = epic_match.group(1)
    else:
        # Fallback loose match
        loose = re.sub(r'[^A-Z0-9]', '', epic_raw)
        if len(loose) >= 8:
            structured["epic_id"] = loose
            
    # 3. Name & Relation (from raw_name_rel)
    lines_nr = [line.strip() for line in raw_data["raw_name_rel"].split('\n') if line.strip()]
    
    # English and Tamil relation keywords
    relation_keys = ["FATHER", "HUSBAND", "MOTHER", "WIFE", "தந்தை", "கணவர்", "தாய்", "கணவன்"]
    
    for i, line in enumerate(lines_nr):
        upper_line = line.upper()
        
        # Name detection
        if not any(k in upper_line for k in relation_keys):
            if "NAME" in upper_line or "பெயர்" in upper_line:
                clean_name = re.sub(r'^(?:NAME|NANE|S NAME|பெயர்|பெயர்:)\s*[:;-]?\s*', '', line, flags=re.IGNORECASE)
                if len(clean_name) > 2 and not structured["name"]:
                    structured["name"] = clean_name.strip()
            elif i == 0 and not structured["name"]: # Fallback to first line if keyword missed
                structured["name"] = line.strip()
                
        # Relation detection
        else:
            for k in relation_keys:
                if k in upper_line:
                    clean_rel = re.sub(k + r'\s*[:;-]?\s*NAME\s*[:;-]?\s*', '', line, flags=re.IGNORECASE)
                    clean_rel = re.sub(k + r'\s*[:;-]?\s*பெயர்\s*[:;-]?\s*', '', clean_rel, flags=re.IGNORECASE)
                    clean_rel = re.sub(k + r'\s*[:;-]?\s*', '', clean_rel, flags=re.IGNORECASE)
                    
                    if len(clean_rel.strip()) > 1:
                        structured["relation_name"] = clean_rel.strip()
                        
                        # Map relation type
                        if k in ["FATHER", "தந்தை"]: structured["relation_type"] = "Father"
                        elif k in ["HUSBAND", "WIFE", "கணவர்", "கணவன்"]: structured["relation_type"] = "Husband"
                        elif k in ["MOTHER", "தாய்"]: structured["relation_type"] = "Mother"
                        break

    # 4. Bottom details (House, Age, Gender)
    bot_text = raw_data["raw_bottom"].replace('\n', ' ')
    
    # House
    # Matches "House No" or "வீட்டு எண்"
    house_match = re.search(r'(?:House\s*N[ou]|வீட்டு\s*எண்)\s*[:;-]?\s*([A-Za-z0-9/\\\-]+)', bot_text, re.IGNORECASE)
    if house_match:
        structured["house_number"] = house_match.group(1).strip()
        
    # Age
    # Matches "Age" or "வயது"
    age_match = re.search(r'(?:Age|வயது)\s*[^0-9]*(\d{2,3})', bot_text, re.IGNORECASE)
    if age_match:
        try:
            val = int(age_match.group(1))
            if 18 <= val <= 120:
                structured["age"] = val
        except:
            pass
            
    # Gender
    # Matches "Gender" or "பாலினம்"
    gen_text = bot_text.upper()
    if "FEMALE" in gen_text or "பெண்" in gen_text:
        structured["gender"] = "Female"
    elif "MALE" in gen_text or "ஆண்" in gen_text:
        structured["gender"] = "Male"
    elif "THIRD" in gen_text or "மூன்றாம்" in gen_text:
        structured["gender"] = "Third Gender"

    return structured
