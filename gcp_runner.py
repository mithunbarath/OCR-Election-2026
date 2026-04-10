"""
GCP Runner for Electoral Roll OCR Pipeline
==========================================
Reads PDFs from a GCS bucket, processes them with the hybrid OCR pipeline,
and writes per-PDF CSV results + a final merged CSV back to GCS.

Usage:
    python gcp_runner.py --bucket my-bucket --prefix pdfs/ --output-prefix results/

Environment variables:
    GOOGLE_APPLICATION_CREDENTIALS - path to service account JSON (auto-set on GCP VMs)
"""

import os
import sys
import argparse
import tempfile
import traceback
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

# Add local package to path
sys.path.insert(0, os.path.dirname(__file__))

from electoral_ocr_pipeline.pdf_processor import pdf_to_images
from electoral_ocr_pipeline.image_segmentation import extract_cards_from_page, get_sub_regions
from electoral_ocr_pipeline.ocr_engine import run_hybrid_ocr
from electoral_ocr_pipeline.data_cleaner import clean_ocr_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def process_single_pdf(pdf_local_path: str, source_name: str) -> list[dict]:
    """
    Runs the full OCR pipeline on a single local PDF file.
    Returns a list of extracted record dicts.
    """
    records = []
    for page_num, cv_img in pdf_to_images(pdf_local_path, dpi=300):
        try:
            cards = extract_cards_from_page(cv_img)
            for i, card_img in enumerate(cards):
                rois = get_sub_regions(card_img)
                raw = run_hybrid_ocr(rois)
                record = clean_ocr_data(raw)
                record["source_file"] = source_name
                record["page_number"] = page_num
                record["card_index"] = i + 1
                records.append(record)
        except Exception as e:
            log.warning(f"  Page {page_num} error in {source_name}: {e}")

    return records


def process_pdf_gcs(args):
    """
    Worker function: downloads a PDF from GCS, runs OCR, returns records.
    args = (bucket_name, gcs_blob_name, worker_id)
    """
    bucket_name, blob_name, worker_id = args
    pdf_name = Path(blob_name).name

    log.info(f"[Worker {worker_id}] Starting: {pdf_name}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        blob.download_to_filename(tmp_path)
        log.info(f"[Worker {worker_id}] Downloaded {pdf_name} ({os.path.getsize(tmp_path) // 1024} KB)")

        records = process_single_pdf(tmp_path, pdf_name)
        log.info(f"[Worker {worker_id}] Finished {pdf_name}: {len(records)} records")
        return pdf_name, records

    except Exception as e:
        log.error(f"[Worker {worker_id}] FAILED {pdf_name}: {e}\n{traceback.format_exc()}")
        return pdf_name, []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def upload_to_gcs(local_path: str, bucket_name: str, gcs_dest: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_dest)
    blob.upload_from_filename(local_path)
    log.info(f"Uploaded → gs://{bucket_name}/{gcs_dest}")


def list_pdfs_in_bucket(bucket_name: str, prefix: str) -> list[str]:
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [b.name for b in blobs if b.name.lower().endswith(".pdf")]


def main():
    parser = argparse.ArgumentParser(description="GCP Electoral Roll OCR Runner")
    parser.add_argument("--bucket",         required=True,  help="GCS bucket name (no gs:// prefix)")
    parser.add_argument("--prefix",         default="",     help="GCS prefix/folder where PDFs are stored")
    parser.add_argument("--output-prefix",  default="results/", help="GCS prefix/folder for output CSVs")
    parser.add_argument("--workers",        type=int, default=4, help="Parallel worker count (default: 4)")
    parser.add_argument("--limit",          type=int, default=0,  help="Process only N PDFs (0 = all, for testing)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(" Electoral Roll Hybrid OCR — GCP Runner")
    log.info("=" * 60)
    log.info(f"Bucket:   gs://{args.bucket}/{args.prefix}")
    log.info(f"Output:   gs://{args.bucket}/{args.output_prefix}")
    log.info(f"Workers:  {args.workers}")

    # List all PDFs
    log.info("\nScanning bucket for PDFs...")
    pdf_blobs = list_pdfs_in_bucket(args.bucket, args.prefix)
    if not pdf_blobs:
        log.error("No PDFs found. Check --bucket and --prefix arguments.")
        sys.exit(1)

    if args.limit > 0:
        pdf_blobs = pdf_blobs[:args.limit]

    log.info(f"Found {len(pdf_blobs)} PDFs to process.\n")

    # Build task list
    tasks = [(args.bucket, blob_name, i + 1) for i, blob_name in enumerate(pdf_blobs)]

    # Process in parallel
    all_records = []
    failed_pdfs = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdf_gcs, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Processing PDFs", unit="pdf") as pbar:
            for future in as_completed(futures):
                pdf_name, records = future.result()
                pbar.update(1)

                if records:
                    all_records.extend(records)

                    # Save per-PDF CSV immediately to GCS
                    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False, newline='', encoding='utf-8') as tmp_csv:
                        tmp_csv_path = tmp_csv.name

                    per_pdf_df = pd.DataFrame(records)
                    per_pdf_df = per_pdf_df.reindex(columns=[
                        "serial_number", "epic_id", "name", "relation_type", "relation_name",
                        "house_number", "age", "gender", "source_file", "page_number", "card_index"
                    ])
                    per_pdf_df.to_csv(tmp_csv_path, index=False)

                    gcs_csv_dest = args.output_prefix + pdf_name.replace(".pdf", "_extracted.csv")
                    upload_to_gcs(tmp_csv_path, args.bucket, gcs_csv_dest)
                    os.remove(tmp_csv_path)
                else:
                    failed_pdfs.append(pdf_name)

    # Write final merged CSV
    log.info(f"\nAll workers done. Total records: {len(all_records)}")
    log.info(f"Failed PDFs: {len(failed_pdfs)}")

    if all_records:
        merged_df = pd.DataFrame(all_records)
        merged_df = merged_df.reindex(columns=[
            "serial_number", "epic_id", "name", "relation_type", "relation_name",
            "house_number", "age", "gender", "source_file", "page_number", "card_index"
        ])

        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False, newline='', encoding='utf-8') as tmp_merged:
            tmp_merged_path = tmp_merged.name

        merged_df.to_csv(tmp_merged_path, index=False)
        upload_to_gcs(tmp_merged_path, args.bucket, args.output_prefix + "ALL_VOTERS_MERGED.csv")
        os.remove(tmp_merged_path)
        log.info(f"\n✅ Done! Merged CSV with {len(merged_df)} rows saved to GCS.")

    if failed_pdfs:
        log.warning(f"\nFailed PDFs:\n" + "\n".join(f"  - {p}" for p in failed_pdfs))


if __name__ == "__main__":
    main()
