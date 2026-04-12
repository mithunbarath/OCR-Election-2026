"""
vision_pipeline.py — v6 (Google Cloud Vision API)
===================================================
Architecture: Page-level Vision API (1 API call per page, NOT 1 per card)
This script processes multiple FULL PDFs.

Setup (do this once):
  1. pip install google-cloud-vision
  2. Create a Google Cloud project, enable "Cloud Vision API"
  3. Create a Service Account, download JSON key
  4. Set CREDENTIALS_JSON below to your JSON file path
  OR run: gcloud auth application-default login

Run:
    python vision_pipeline.py
"""

import fitz
import cv2
import numpy as np
import re
import os
import sys
import pandas as pd
import glob

try:
    from google.cloud import vision as gvision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("[ERROR] google-cloud-vision not installed.")
    print("  Run: pip install google-cloud-vision")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
# Set to the path of your Google Cloud service account JSON key.
# Leave empty ("") to use Application Default Credentials.
CREDENTIALS_JSON = r"C:\Users\navee\Downloads\gcp_vision_key.json"

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
EROLL_DIR      = os.path.join(SCRIPT_DIR, "119-eroll")
RENDER_SCALE   = 2          # 2× is fine for Vision API (it handles lower DPI well)
MAX_TEST_PAGES = 0          # 0 for processing ALL pages in the PDF
MAX_PDFS       = 2          # Process the first 2 PDFs
OUT_CSV        = os.path.join(SCRIPT_DIR, "vision_2_pdfs_result.csv")

GRID_ROWS      = 10
GRID_COLS      = 3
HEADER_FRAC    = 0.11
FOOTER_FRAC    = 0.04


# ─── Vision API client ────────────────────────────────────────────────────────
def get_vision_client():
    if CREDENTIALS_JSON and os.path.exists(CREDENTIALS_JSON):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_JSON)
        return gvision.ImageAnnotatorClient(credentials=creds)
    return gvision.ImageAnnotatorClient()   # uses ADC / env var


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE-LEVEL OCR  (1 API call per full page image)
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_page_vision(page_img: np.ndarray, client) -> list[dict]:
    """
    Send a full page image to Google Vision API.
    Returns a list of word dicts:
      { 'text': str, 'cx': float, 'cy': float,
        'x0': int, 'y0': int, 'x1': int, 'y1': int }
    Pixel coordinates match the coordinate space of page_img.
    """
    _, png_bytes = cv2.imencode('.png', page_img)
    image   = gvision.Image(content=png_bytes.tobytes())
    resp    = client.document_text_detection(image=image)

    if resp.error.message:
        print(f"[Vision API Error] {resp.error.message}")
        return []

    words = []
    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    text  = ''.join(s.text for s in word.symbols)
                    verts = word.bounding_box.vertices
                    xs    = [v.x for v in verts]
                    ys    = [v.y for v in verts]
                    words.append({
                        'text': text,
                        'cx': sum(xs) / len(xs),
                        'cy': sum(ys) / len(ys),
                        'x0': min(xs), 'y0': min(ys),
                        'x1': max(xs), 'y1': max(ys),
                    })
    return words


# ═══════════════════════════════════════════════════════════════════════════════
#  CARD SEGMENTATION  (returns images + coordinates)
# ═══════════════════════════════════════════════════════════════════════════════

def segment_page(img: np.ndarray):
    """
    Returns:
      cards      – list of card images (cropped from page)
      coords     – list of (x1, y1, x2, y2) tuples in page pixel space
      region_cnt – number of detected regions
    """
    img_h, img_w = img.shape[:2]
    top  = int(img_h * HEADER_FRAC)
    bot  = img_h - int(img_h * FOOTER_FRAC)
    work = img[top:bot, :]

    gray   = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    sharp  = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32))
    binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)

    def cluster(arr, gap=8):
        if not len(arr): return []
        out, cur = [], [arr[0]]
        for v in arr[1:]:
            if v - cur[-1] <= gap: cur.append(v)
            else: out.append(int(np.mean(cur))); cur = [v]
        out.append(int(np.mean(cur)))
        return out

    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w//4, 1))
    rsep = cluster(np.where(np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk), axis=1)
                            > img_w*255*0.25)[0])
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, (bot-top)//6))
    csep = cluster(np.where(np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk), axis=0)
                            > (bot-top)*255*0.20)[0])

    rh = (bot-top)//GRID_ROWS;  cw = img_w//GRID_COLS
    rb = ([0]+rsep+[bot-top]) if 8<=len(rsep)<=15 else [r*rh for r in range(GRID_ROWS+1)]
    cb = ([0]+csep+[img_w])   if 1<=len(csep)<=4  else [c*cw for c in range(GRID_COLS+1)]

    coords = [(cb[ci], top+rb[ri], cb[ci+1], top+rb[ri+1])
              for ri in range(len(rb)-1)
              for ci in range(len(cb)-1)
              if (rb[ri+1]-rb[ri])>=40 and (cb[ci+1]-cb[ci])>=40]

    if not (20 <= len(coords) <= 40):
        coords = [(c*cw, top+r*rh, (c+1)*cw, top+(r+1)*rh)
                  for r in range(GRID_ROWS) for c in range(GRID_COLS)]

    cards = [img[y1:y2, x1:x2]
             for x1,y1,x2,y2 in coords
             if img[y1:y2,x1:x2].size > 0]

    return cards, coords, len(coords)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORD → CARD ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assign_words_to_cards(all_words: list[dict],
                          coords: list[tuple]) -> list[list[dict]]:
    """
    Each word is assigned to the card whose bounding box contains the word center.
    """
    per_card = [[] for _ in coords]
    for w in all_words:
        for i, (x1, y1, x2, y2) in enumerate(coords):
            if x1 <= w['cx'] <= x2 and y1 <= w['cy'] <= y2:
                per_card[i].append(w)
                break
    return per_card


def card_text(words: list[dict]) -> str:
    """Reconstruct readable text from a card's words, sorted top-to-bottom then left-to-right."""
    row_h   = 10    # group words within 10px into the same "row"
    sorted_ = sorted(words, key=lambda w: (round(w['cy'] / row_h) * row_h, w['cx']))
    return ' '.join(w['text'] for w in sorted_)


# ═══════════════════════════════════════════════════════════════════════════════
#  EPIC ID EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

_CORR_CONS = str.maketrans('OILSZB',   '011528')
_CORR_AGG  = str.maketrans('OILSZBDG', '01152806')
# Reverse: correct digits → letters in the 3-char prefix position
_REV_PREFIX = str.maketrans('015892', 'OISBN2')   # for R1V → RIV


def _soft_validate_epic(raw: str) -> str | None:
    """
    Soft EPIC validation:
      • 2-3 leading uppercase letters (1→I, 0→O corrected in prefix)
      • 5-7 recoverable digits following
    """
    s = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
    if len(s) < 7:
        return None

    # Normalize the first 3 chars: digits that look like letters get corrected
    normalized = s[:3].translate(_REV_PREFIX) + s[3:]

    # Count leading letters
    prefix = ''
    for c in normalized:
        if c.isalpha() and len(prefix) < 3:
            prefix += c
        else:
            break

    if len(prefix) < 2:
        return None

    rest = normalized[len(prefix):]
    if not rest:
        return None

    for table in (_CORR_CONS, _CORR_AGG):
        corrected   = rest[:7].translate(table)
        digit_count = sum(1 for c in corrected if c.isdigit())

        if digit_count == 7 and re.fullmatch(r'[0-9]{7}', corrected):
            return prefix.ljust(3, prefix[-1])[:3] + corrected

        if digit_count >= 5:    # soft accept
            return prefix.ljust(3, prefix[-1])[:3] + corrected[:7].ljust(7, '0')

    return None


def extract_epic(card_words: list[dict],
                 cx1: int, cy1: int, cx2: int, cy2: int) -> str | None:
    """
    Strategy 1: positional – look at words in top-right of card (y<30%, x>50%)
    Strategy 2: regex scan through all card text
    """
    card_h = cy2 - cy1
    card_w = cx2 - cx1
    epic_y_max = cy1 + card_h * 0.30
    epic_x_min = cx1 + card_w * 0.50

    # Strategy 1: positional
    epic_region_words = [w['text'] for w in card_words
                         if w['cx'] >= epic_x_min and w['cy'] <= epic_y_max]
    for candidate in epic_region_words + [''.join(epic_region_words)]:
        epic = _soft_validate_epic(candidate)
        if epic:
            return epic

    # Strategy 2: regex over full card text
    full = card_text(card_words)
    for m in re.finditer(r'[A-Za-z0-9]{9,12}', full):
        epic = _soft_validate_epic(m.group())
        if epic:
            return epic

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  FIELD PARSING FROM CARD TEXT
# ═══════════════════════════════════════════════════════════════════════════════

NAME_RE      = r"(?:ELECTOR'?S?\s*)?(?:NAME|NANE|WAME|MAME|NAIME|NAXE|RAME)"
NAME_SEP     = r'[\s:;.\-*+=?|!]+'
AGE_SEP      = r'[\s:=\-–+*?{}|¢·,;\]\[]+'
GENDER_LABEL = r'(?:Gender|G[ae]nd[ae]r|Gander|Gendar|Sex)'
GENDER_VALUE = (r'(?P<gv>Female|Femata|Femate|Femala|Famala|Femta|'
                r'Male|Mala|Mela|Third\s*Gender|F(?:emale)?|M(?:ale)?)')
GENDER_SEP   = r'[\s:=\-–+*?{}|¢·,;\]\[]+'

RELATION_MAP = [
    ("SON OF","Father"),("S/O","Father"),("D/O","Father"),
    ("W/O","Husband"),("H/O","Wife"),
    ("HUSBAND","Husband"),("FATNER","Father"),("FETHER","Father"),
    ("FATHER","Father"),("MOTHER","Mother"),("WIFE","Wife"),
]


def _clean_name(raw: str) -> str | None:
    raw  = re.sub(r'[^\x00-\x7F]', ' ', raw)
    raw  = re.sub(r'^[^A-Za-z]+', '', raw)
    kept = re.sub(r'[^A-Za-z\s.\'\-]', '', raw).strip(" .|][!lt\n")
    kept = re.sub(r'\s{2,}', ' ', kept).strip()
    return kept if len(kept) >= 2 else None


def _serial(text: str) -> int | None:
    for line in [l.strip() for l in text.split(' ') if l.strip()][:4]:
        m = re.fullmatch(r'(\d{1,3})', line)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 999:
                return v
    return None


def parse_fields(text: str, serial: int | None, card_words: list[dict],
                 cx1: int, cy1: int, cx2: int, cy2: int) -> dict:
    """
    Parse all fields. Vision API text is much cleaner → regex hits higher.
    """
    rec = dict(name=None, relation_type=None, relation_name=None,
               house_number=None, age=None, gender=None)

    full  = text
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # also try space-split pseudo-lines from Vision API word ordering
    if not lines:
        lines = [text]

    # ── House Number ──────────────────────────────────────────────────────────
    hm = re.search(
        r'House\s*(?:Number|Num|No|N\.?)[^a-zA-Z0-9]{0,6}(\d[\dA-Za-z/\\\-]*)',
        full, re.IGNORECASE)
    if hm:
        val = re.split(r'(?i)photo|avail', hm.group(1).rstrip('.,; '))[0].strip().rstrip('.,;')
        rec["house_number"] = val or None
    house_digits = set(re.findall(r'\d+', rec["house_number"] or ""))

    # ── Age ───────────────────────────────────────────────────────────────────
    age_kw = bool(re.search(r'\b(?:Age|A\s*ge|Aqe|Aae)\b', full, re.IGNORECASE))
    age_m  = re.search(r'(?:Age|A\s*ge|Aqe|Aae)' + AGE_SEP + r'(\d{2,3})',
                       full, re.IGNORECASE)
    if age_m:
        raw_n = age_m.group(1); v = int(raw_n)
        if 18 <= v <= 110 and not (serial and v==serial) and str(v) not in house_digits:
            rec["age"] = v
        elif len(raw_n)==3 and v > 110:
            for sub in (int(raw_n[1:]), int(raw_n[:2])):
                if 18 <= sub <= 110 and not (serial and sub==serial) and str(sub) not in house_digits:
                    rec["age"] = sub; break

    if rec["age"] is None and not age_kw:
        for n in re.findall(r'\b(\d{2,3})\b', full):
            v = int(n)
            if (serial and v==serial) or str(v) in house_digits: continue
            if 18 <= v <= 110:
                rec["age"] = v; break

    # ── Gender ────────────────────────────────────────────────────────────────
    gm = re.search(GENDER_LABEL + GENDER_SEP + GENDER_VALUE, full, re.IGNORECASE)
    if gm:
        g = gm.group("gv").upper()
        rec["gender"] = "Female" if g.startswith('F') else \
                        "Male"   if g.startswith('M') else "Third Gender"
    elif re.search(r'\bFe?mata?\b|\bFemala\b|\bFamala\b|\bFemate\b', full, re.IGNORECASE):
        rec["gender"] = "Female"
    elif re.search(r'\bMale\b|\bMala\b|\bMela\b', full, re.IGNORECASE):
        rec["gender"] = "Male"

    # ── Name & Relation ───────────────────────────────────────────────────────
    m1 = re.search(r'(?:^|\s)(?:Name|Nama|Nane)\s*[:;\-]?\s*(.*?)(?=\s+(?:Husband|Father|Mother|Wife|Other)\b|\s+House\b|\s+Photo|\s+Age|$)', full, re.IGNORECASE)
    if m1:
        n = _clean_name(m1.group(1))
        if n and len(n) > 1:
            rec["name"] = n

    m2 = re.search(r'\b(Husband|Father|Mother|Wife|Other)\b\s*(?:Name|Nama|Nane)?\s*[:;\-\?]?\s*(.*?)(?=\s+House\b|\s+Photo|\s+Age|$)', full, re.IGNORECASE)
    if m2:
        rec["relation_type"] = m2.group(1).capitalize()
        rn = _clean_name(m2.group(2))
        if rn and len(rn) > 1:
            rec["relation_name"] = rn

    return rec


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print(f"  ELECTION ROLL OCR — Vision API (Processing {MAX_PDFS} PDFs)")
    print("=" * 65)

    if not os.path.exists(EROLL_DIR):
        print(f"[ERROR] PDF directory not found: {EROLL_DIR}")
        return

    pdf_files = sorted(glob.glob(os.path.join(EROLL_DIR, "*.pdf")))
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {EROLL_DIR}")
        return
        
    pdf_files = pdf_files[:MAX_PDFS]

    # Init Vision client
    try:
        client = get_vision_client()
        print("[OK] Vision API client initialized")
    except Exception as e:
        print(f"[ERROR] Could not initialize Vision API client: {e}")
        print("  Make sure CREDENTIALS_JSON is set or ADC is configured.")
        return

    all_records = []
    grand_total = 0

    for pdf_idx, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[PDF {pdf_idx}/{len(pdf_files)}] {os.path.basename(pdf_file)}")
        doc = fitz.open(pdf_file)
        total_pages = len(doc)

        start = 2
        end   = (min(total_pages-1, start+MAX_TEST_PAGES) if MAX_TEST_PAGES > 0
                 else total_pages-1)
                 
        if end <= start:
            print(f"         ✗ PDF too short ({total_pages} pages), skipping.")
            doc.close()
            continue

        print(f"[TEST] Pages {start+1} to {end}  (of {total_pages} total)")
        print(f"       = {end-start} Vision API calls total for this PDF\n")

        for page_num in range(start, end):
            print()
            print("-" * 65)
            print(f"  PAGE {page_num+1}  (Vision API call {page_num-start+1}/{end-start})")
            print("-" * 65)

            page = doc.load_page(page_num)
            pix  = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
            img  = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)

            # ── Segment page into card regions ────────────────────────────────────
            cards, coords, region_count = segment_page(img)
            print(f"  Grid: {region_count} regions -> {len(cards)} cards")

            # ── ONE Vision API call for the ENTIRE page ───────────────────────────
            print(f"  Calling Vision API...", end=' ', flush=True)
            all_words = ocr_page_vision(img, client)
            print(f"got {len(all_words)} words")

            # ── Assign each word to its card ──────────────────────────────────────
            per_card_words = assign_words_to_cards(all_words, coords)

            # ── Parse each card ───────────────────────────────────────────────────
            for idx, (card_img, (cx1,cy1,cx2,cy2)) in enumerate(zip(cards, coords)):
                grand_total += 1
                cwords = per_card_words[idx]
                ctext  = card_text(cwords)

                # --- Serial
                serial = _serial(ctext)

                # --- EPIC (positional + regex fallback)
                epic = extract_epic(cwords, cx1, cy1, cx2, cy2)

                # --- Other fields via regex
                fields = parse_fields(ctext, serial, cwords, cx1, cy1, cx2, cy2)

                flags = []
                if not epic:                        flags.append("EPIC_MISSING")
                if fields["age"] is None:           flags.append("AGE_MISSING")
                if not fields["name"]:              flags.append("NAME_MISSING")
                if not fields["gender"]:            flags.append("GENDER_MISSING")
                if not fields["house_number"]:      flags.append("HOUSE_MISSING")
                if not fields["relation_name"]:     flags.append("RELATION_MISSING")

                ok       = "OK " if not flags else "ERR"
                flag_str = ("  [" + " | ".join(flags) + "]") if flags else ""
                
                # Consolidate log line to reduce terminal clutter
                print(f"  Card {idx+1:>2}  (#{grand_total:>4})  {ok}{flag_str}  | EPIC: {epic} | Name: {fields['name']} | Rel: {fields['relation_name']}")

                all_records.append(dict(
                    serial_number = serial,
                    epic_id       = epic,
                    name          = fields["name"],
                    relation_type = fields["relation_type"],
                    relation_name = fields["relation_name"],
                    house_number  = fields["house_number"],
                    age           = fields["age"],
                    gender        = fields["gender"],
                    source_file   = os.path.basename(pdf_file),
                    page_number   = page_num+1,
                    card_on_page  = idx+1,
                    ocr_flag      = " | ".join(flags),
                    raw_vision_text = ctext,
                ))

        doc.close()

    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    total = len(all_records)
    print(f"  Total cards : {total}")

    if total > 0:
        df   = pd.DataFrame(all_records)
        cols = ["serial_number","epic_id","name","relation_type","relation_name",
                "house_number","age","gender","source_file","page_number",
                "card_on_page","ocr_flag","raw_vision_text"]
        df   = df.reindex(columns=cols)

        def pct(col, flag_mode=False):
            n = (df[col]=="").sum() if flag_mode else df[col].notna().sum()
            return f"{n}/{total}  ({n/total*100:.1f}%)"

        print(f"  EPIC ID found   : {pct('epic_id')}")
        print(f"  Serial found    : {pct('serial_number')}")
        print(f"  Age found       : {pct('age')}")
        print(f"  Name found      : {pct('name')}")
        print(f"  Gender found    : {pct('gender')}")
        print(f"  House # found   : {pct('house_number')}")
        print(f"  Clean records   : {pct('ocr_flag', flag_mode=True)}")

        df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n  CSV -> {OUT_CSV}")

    print("=" * 65)


if __name__ == "__main__":
    main()
