# 🗳️ Electoral Roll OCR Extraction Pipeline

A robust, scalable hybrid OCR pipeline to extract structured voter data from multi-page PDF Electoral Roll documents (ECI format). Combines **Tesseract** and **EasyOCR** with layout-aware image segmentation for high-accuracy bilingual (Tamil + English) field extraction.

---

## 📌 Features

- 🔍 **Hybrid OCR** — Tesseract for structured alphanumeric fields + EasyOCR for Tamil/English names
- 📄 **PDF to Image** conversion at 300+ DPI using PyMuPDF (no Poppler needed)
- 🗂️ **Grid-based card segmentation** — fixed 3×10 layout with proportional sub-region isolation
- 🖼️ **Per-field preprocessing** — adaptive thresholding, upscaling, denoising per ROI
- 🔤 **Bilingual post-processing** — regex + Tamil keyword normalization
- ⚡ **Multiprocessing** — parallel page processing via `ProcessPoolExecutor`
- 📊 **Structured CSV output** — 8 fields per voter card

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      main.py  (Orchestrator)                    │
│                                                                 │
│  PDF Files ──► pdf_processor.py ──► [Page Images @ 300 DPI]    │
│                                            │                    │
│                                   image_segmentation.py         │
│                                            │                    │
│                        ┌──────────────────────────────┐         │
│                        │  30 Voter Cards per Page     │         │
│                        │  (3 cols × 10 rows grid)     │         │
│                        └──────────────────────────────┘         │
│                                            │                    │
│                                   Sub-Region Isolation          │
│                        ┌──────────────────────────────┐         │
│                        │  serial_no │ epic_id         │         │
│                        │  name_rel  │ bottom_details  │         │
│                        └──────────────────────────────┘         │
│                                            │                    │
│                                     ocr_engine.py               │
│                         ┌──────────────────────────┐            │
│                         │  Tesseract  │  EasyOCR   │            │
│                         │  (EPIC/Age) │ (Names/Rel)│            │
│                         └──────────────────────────┘            │
│                                            │                    │
│                                    data_cleaner.py              │
│                                            │                    │
│                              extracted_voters_hybrid.csv        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
OCR-Election-2026/
│
├── electoral_ocr_pipeline/          # Core pipeline package
│   ├── __init__.py
│   ├── main.py                      # Entry point + multiprocessing orchestrator
│   ├── pdf_processor.py             # PDF → high-res OpenCV images
│   ├── image_segmentation.py        # Page → cards → sub-regions (ROIs)
│   ├── ocr_engine.py                # Hybrid OCR (Tesseract + EasyOCR)
│   └── data_cleaner.py              # Text post-processing + regex field extraction
│
├── tesseract_extractor.py           # Standalone Tesseract-only extractor (legacy)
├── easyocr_extractor.py             # Standalone EasyOCR extractor (legacy)
├── ocr_extractor.py                 # Vision API extractor (legacy)
│
├── extracted_voters_hybrid.csv      # Main output (hybrid pipeline)
├── voters_extracted_tess.csv        # Output from Tesseract-only run
├── voters_extracted_easyocr.csv     # Output from EasyOCR-only run
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Module Details

### `pdf_processor.py`
Converts each PDF page to a high-resolution OpenCV image using PyMuPDF.

- Uses `fitz.Matrix(4.0, 4.0)` for ~288 DPI rendering (no Poppler dependency)
- Automatically skips the cover page and deletion-summary last page
- Yields `(page_number, cv2_bgr_image)` as a memory-efficient generator

### `image_segmentation.py`
Splits each page image into voter cards and each card into field regions.

**Page → 30 Cards (3×10 grid):**
```
┌──────────────┬──────────────┬──────────────┐  ▲ 11% header offset
│  Card [1,1]  │  Card [1,2]  │  Card [1,3]  │
├──────────────┼──────────────┼──────────────┤
│     ...      │     ...      │     ...      │  10 rows
├──────────────┼──────────────┼──────────────┤
│  Card[10,1]  │  Card[10,2]  │  Card[10,3]  │
└──────────────┴──────────────┴──────────────┘  ▼ 4% footer offset
```

**Card → Sub-Regions (ROIs):**

| Region         | X Range         | Y Range           | Fields Extracted        |
|----------------|-----------------|-------------------|-------------------------|
| `serial_no`    | 0% → 35%        | 0% → 15%          | Serial Number           |
| `epic_id`      | 35% → 100%      | 0% → 15%          | EPIC ID                 |
| `name_rel`     | 25% → 100%      | 15% → 65%         | Name, Relation Type/Name|
| `bottom_details`| 25% → 100%    | 65% → 95%         | House No, Age, Gender   |

### `ocr_engine.py`
Applies optimal OCR strategy per field region.

| Region           | OCR Engine  | Config                                        |
|------------------|-------------|-----------------------------------------------|
| `serial_no`      | Tesseract   | PSM 7, whitelist: `A-Z0-9`, Otsu threshold   |
| `epic_id`        | Tesseract   | PSM 7, whitelist: `A-Z0-9`, Otsu threshold   |
| `name_rel`       | EasyOCR     | `['en', 'ta']`, 2× upscale                   |
| `bottom_details` | EasyOCR     | `['en', 'ta']`, 2× upscale                   |

**Preprocessing per ROI:**
1. BGR → Grayscale
2. 2× bicubic upscaling
3. Adaptive Gaussian thresholding (for Tesseract) or raw BGR (for EasyOCR)

### `data_cleaner.py`
Parses raw OCR strings into structured fields using regex and bilingual heuristics.

| Field           | Strategy                                                      |
|-----------------|---------------------------------------------------------------|
| `epic_id`       | Regex: `[A-Z]{3}[0-9]{7}` with OCR noise cleanup             |
| `serial_number` | Strip non-numeric characters                                  |
| `name`          | Keyword detection: `NAME` / `பெயர்`                          |
| `relation_type` | Keyword match: `FATHER/HUSBAND/MOTHER` or Tamil equivalents   |
| `relation_name` | Text after relation keyword                                   |
| `house_number`  | Regex: after `House No` / `வீட்டு எண்`                       |
| `age`           | Regex: integer 18–120 after `Age` / `வயது`                   |
| `gender`        | Keyword: `MALE/FEMALE/THIRD` or `ஆண்/பெண்/மூன்றாம்`          |

### `main.py`
Parallel orchestrator using `ProcessPoolExecutor`.

```
1. Load all PDFs → extract page images sequentially (memory safe)
2. Submit each page as a parallel worker task
3. Each worker: segment cards → isolate ROIs → run hybrid OCR → clean data
4. Aggregate all records
5. Export to CSV
```

- Default workers: `min(cpu_count, 4)` — reduce to `2` if OOM errors occur
- All exceptions caught per-page; failed pages are logged and skipped

---

## 📤 Output CSV Schema

| Column          | Type    | Description                            |
|-----------------|---------|----------------------------------------|
| `serial_number` | string  | Card serial number within the roll     |
| `epic_id`       | string  | Voter EPIC ID (format: `XXX1234567`)   |
| `name`          | string  | Voter's full name                      |
| `relation_type` | string  | `Father` / `Husband` / `Mother`        |
| `relation_name` | string  | Name of the relation                   |
| `house_number`  | string  | House/door number                      |
| `age`           | integer | Voter's age (18–120)                   |
| `gender`        | string  | `Male` / `Female` / `Third Gender`     |
| `source_file`   | string  | Source PDF filename                    |
| `page_number`   | integer | Page number within the PDF             |
| `card_index`    | integer | Card index on the page (1–30)          |

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR for Windows**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default path: `C:\Program Files\Tesseract-OCR\`

### Installation

```bash
pip install PyMuPDF>=1.23.0 opencv-python>=4.8.0 pytesseract easyocr pandas
```

Or via requirements file:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Place your ECI Electoral Roll PDF(s) in the project root directory, then run:

```bash
python -m electoral_ocr_pipeline.main
```

Output will be saved as `extracted_voters_hybrid.csv`.

---

## 🔧 Troubleshooting

### EasyOCR Tamil model size mismatch
If you encounter:
```
RuntimeError: size mismatch for Prediction.weight ...
```
This is a known bug in recent EasyOCR releases where the server-side `tamil.pth` has more characters than the installed code expects. The pipeline includes a patch to EasyOCR's `recognition.py` to **filter shape-mismatched layers** before loading, allowing the model to load without crashing.

### Low OCR accuracy for Tamil names
- The Tamil sub-regions use EasyOCR in bilingual mode (`en` + `ta`)
- Ensure the `tamil.pth` model is downloaded (first run requires internet)
- For better accuracy, tweak ROI proportions in `image_segmentation.py`

### OOM / Memory errors
Reduce the worker count in `main.py`:
```python
workers = 2  # line 63
```

---

## 📊 Benchmark

Tested on a 300-page ECI Electoral Roll PDF (9,000 voter cards):

| Metric              | Value         |
|---------------------|---------------|
| Pages Processed     | 300           |
| Cards per Page      | 30            |
| Total Cards         | 9,000         |
| Avg Time per Page   | ~8–12 seconds |
| EPIC ID Accuracy    | ~92%          |
| Name Accuracy (en)  | ~85%          |

---

## 📄 License

This project is developed for electoral data research purposes. Ensure data handling complies with applicable privacy laws and ECI guidelines.

---

## 👤 Author

**Mithun Barath M R**
- 📧 barathmithun1548@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/mithunbarathmr13/)
- 🐙 [GitHub](https://github.com/mithunbarath)
- 📍 Tiruppur, India
