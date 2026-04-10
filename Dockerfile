FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tam \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    PyMuPDF>=1.23.0 \
    opencv-python-headless>=4.8.0 \
    pytesseract>=0.3.10 \
    easyocr>=1.7.0 \
    pandas>=2.0.0 \
    google-cloud-storage>=2.10.0 \
    tqdm>=4.66.0

# Patch EasyOCR to handle Tamil model shape mismatch
RUN python -c "
import site, os, re
sp = site.getsitepackages()[0]
rec_path = os.path.join(sp, 'easyocr', 'recognition.py')
with open(rec_path, 'r') as f:
    content = f.read()

# Patch 1: CPU path
old1 = '        model.load_state_dict(new_state_dict)'
new1 = '''        from collections import OrderedDict as _OD
        _ms = model.state_dict()
        _cs = _OD({k: v for k, v in new_state_dict.items() if k in _ms and _ms[k].shape == v.shape})
        model.load_state_dict(_cs, strict=False)'''
# Patch 2: GPU path
old2 = '        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))'
new2 = '''        _raw = torch.load(model_path, map_location=device, weights_only=False)
        from collections import OrderedDict as _OD
        _ms = model.state_dict()
        _cs = _OD({k: v for k, v in _raw.items() if k in _ms and _ms[k].shape == v.shape})
        model.load_state_dict(_cs, strict=False)'''
content = content.replace(old1, new1).replace(old2, new2)
with open(rec_path, 'w') as f:
    f.write(content)
print('EasyOCR patched successfully')
"

# Copy the pipeline code
COPY electoral_ocr_pipeline/ ./electoral_ocr_pipeline/
COPY gcp_runner.py .

# Warm up EasyOCR model downloads at build time so containers start fast
RUN python -c "import easyocr; easyocr.Reader(['en', 'ta'], gpu=False)" || true

ENTRYPOINT ["python", "gcp_runner.py"]
