"""Check if the PDF has an embedded text layer (no OCR needed if it does)."""
import fitz, os

PDF = "2026-FC-EROLLGEN-S22-119-SIR-FinalRoll-Revision2-ENG-9-WI.pdf"
doc = fitz.open(PDF)
page = doc.load_page(2)

print("=== RAW TEXT LAYER (page 3) ===")
text = page.get_text()
print(text[:3000])
print()
print("=== BLOCKS with bounding boxes (first 10) ===")
for b in page.get_text("blocks")[:10]:
    # (x0, y0, x1, y1, text, block_no, block_type)
    print(f"  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]  {repr(b[4][:80])}")
