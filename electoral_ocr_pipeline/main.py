import glob
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Import our modular pipeline pieces
from electoral_ocr_pipeline.pdf_processor import pdf_to_images
from electoral_ocr_pipeline.image_segmentation import extract_cards_from_page, get_sub_regions
from electoral_ocr_pipeline.ocr_engine import run_hybrid_ocr
from electoral_ocr_pipeline.data_cleaner import clean_ocr_data

OUTPUT_CSV = "extracted_voters_hybrid.csv"

def process_page(args):
    """
    Task function to run in a separate process for a given page.
    """
    page_num, cv_img, source_file = args
    page_records = []
    
    print(f"  [Worker] Starting {source_file} - Page {page_num}")
    
    # 1. Break page into cards
    cards = extract_cards_from_page(cv_img)
    
    for i, card_img in enumerate(cards):
        # 2. Extract Sub regions
        rois = get_sub_regions(card_img)
        
        # 3. OCR (Hybrid Tesseract + EasyOCR)
        raw_ocr_data = run_hybrid_ocr(rois)
        
        # 4. Clean & Structure
        record = clean_ocr_data(raw_ocr_data)
        
        # Add metadata
        record["source_file"] = source_file
        record["page_number"] = page_num
        record["card_index"] = i + 1
        
        page_records.append(record)
        
    print(f"  [Worker] Finished {source_file} - Page {page_num} ({len(page_records)} records)")
    return page_records

def main():
    print("==================================================")
    print(" Hybrid OCR Voter Roll Extraction Pipeline")
    print(" (Tesseract + EasyOCR with Region Segmentation) ")
    print("==================================================\n")
    
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
        
    # We will accumulate all tasks (pages) here
    tasks = []
    
    # Step A: Parse PDFs and extract images (Done sequentially to avoid massive RAM usage)
    print("Step 1: Extracting page images from PDFs...")
    for pdf_file in pdf_files:
        print(f"  -> Loading {pdf_file}")
        # The generator yields (page_number, cv2_img)
        for page_num, cv_img in pdf_to_images(pdf_file, dpi=300):
            tasks.append((page_num, cv_img, os.path.basename(pdf_file)))
            
    total_pages = len(tasks)
    print(f"Total pages to process: {total_pages}\n")
    
    if total_pages == 0:
        print("No pages extracted. Exiting.")
        return

    all_records = []
    start_time = time.time()
    
    # Step B: Process pages in parallel
    print("Step 2: Running parallel Hybrid OCR...")
    # Adjust max_workers depending on CPU/RAM limits. 
    # EasyOCR takes a lot of memory per worker.
    workers = min(os.cpu_count() or 4, 4) 
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_page, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                page_results = future.result()
                all_records.extend(page_results)
            except Exception as e:
                task_info = futures[future]
                print(f"  [!] Extraction failed on {task_info[2]} Page {task_info[0]}: {e}")
                
    end_time = time.time()
    print(f"\nExtraction complete in {end_time - start_time:.2f} seconds.")
    
    # Step C: Export
    print("Step 3: Exporting to CSV...")
    df = pd.DataFrame(all_records)
    
    # Arrange columns
    cols = [
        "serial_number", "epic_id", "name", "relation_type", "relation_name",
        "house_number", "age", "gender", "source_file", "page_number", "card_index"
    ]
    df = df.reindex(columns=cols)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved {len(df)} records to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
