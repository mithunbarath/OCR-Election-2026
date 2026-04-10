#!/bin/bash
# =============================================================
#  GCP VM Startup Script for Electoral Roll OCR Pipeline
#  Run this ONCE after you SSH into the VM to set everything up.
# =============================================================

set -e

PROJECT_ID="YOUR_PROJECT_ID"         # ← Replace
BUCKET_NAME="YOUR_BUCKET_NAME"       # ← Replace (no gs:// prefix)
PDF_PREFIX="pdfs/"                   # ← GCS folder where your PDFs are
OUTPUT_PREFIX="results/"             # ← GCS folder for output CSVs
WORKERS=8                            # ← Match to vCPU count (n1-highmem-8 → 8)
REPO_URL="https://github.com/mithunbarath/OCR-Election-2026.git"

echo "=================================================="
echo " Electoral Roll OCR — GCP VM Setup"
echo "=================================================="

# 1. System dependencies
sudo apt-get update -y
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-tam \
    python3-pip python3-venv git libgl1-mesa-glx libglib2.0-0

# 2. Clone the repo
cd /home
git clone $REPO_URL ocr-pipeline || (cd ocr-pipeline && git pull)
cd /home/ocr-pipeline

# 3. Python environment
python3 -m venv venv
source venv/bin/activate

# 4. Install pip dependencies
pip install --upgrade pip
pip install PyMuPDF>=1.23.0 \
            opencv-python-headless>=4.8.0 \
            pytesseract>=0.3.10 \
            easyocr>=1.7.0 \
            pandas>=2.0.0 \
            google-cloud-storage>=2.10.0 \
            tqdm>=4.66.0

# 5. Patch EasyOCR Tamil model size mismatch
python3 -c "
import site, os
sp = site.getsitepackages()[0]
rec_path = os.path.join(sp, 'easyocr', 'recognition.py')
with open(rec_path, 'r') as f:
    content = f.read()

old1 = '        model.load_state_dict(new_state_dict)'
new1 = '''        from collections import OrderedDict as _OD
        _ms = model.state_dict()
        _cs = _OD({k: v for k, v in new_state_dict.items() if k in _ms and _ms[k].shape == v.shape})
        model.load_state_dict(_cs, strict=False)'''
old2 = '        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))'
new2 = '''        _raw = torch.load(model_path, map_location=device, weights_only=False)
        from collections import OrderedDict as _OD
        _ms = model.state_dict()
        _cs = _OD({k: v for k, v in _raw.items() if k in _ms and _ms[k].shape == v.shape})
        model.load_state_dict(_cs, strict=False)'''
content = content.replace(old1, new1).replace(old2, new2)
with open(rec_path, 'w') as f:
    f.write(content)
print('EasyOCR patched OK')
"

# 6. Warm-up EasyOCR model (downloads Tamil + English models once)
echo "Downloading EasyOCR models (one-time)..."
python3 -c "import easyocr; easyocr.Reader(['en', 'ta'], gpu=False)" || true

# 7. Launch the pipeline!
echo ""
echo "=================================================="
echo " Starting OCR pipeline on $BUCKET_NAME..."
echo "=================================================="

python3 gcp_runner.py \
    --bucket "$BUCKET_NAME" \
    --prefix "$PDF_PREFIX" \
    --output-prefix "$OUTPUT_PREFIX" \
    --workers "$WORKERS"

echo ""
echo "✅ Pipeline complete. Results in gs://$BUCKET_NAME/$OUTPUT_PREFIX"
