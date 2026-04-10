import fitz
import cv2
import numpy as np
import os

def pdf_to_images(pdf_path, dpi=300):
    """
    Generator that converts a multi-page PDF into high-res OpenCV images.
    Yields (page_number, cv2_image).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF {pdf_path} not found.")

    doc = fitz.open(pdf_path)
    
    # Render with higher resolution (300 DPI equivalent)
    # 300 DPI is ~4.16 zoom (72 * 4.16 = ~300)
    zoom = 4.0
    mat = fitz.Matrix(zoom, zoom)

    # We skip cover pages and summary pages. Standard ECI roll has dataset from page 3 to len-1
    start_page = 2
    end_page = len(doc) - 1

    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array and OpenCV format (BGR)
        img_array = np.frombuffer(pix.tobytes("png"), np.uint8)
        cv2_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        yield page_num + 1, cv2_img
        
    doc.close()
