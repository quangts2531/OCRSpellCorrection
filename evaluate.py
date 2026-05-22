"""
evaluate.py — Standalone evaluation script for the OCRSpellCorrection pipeline.

Usage:
    python evaluate.py --image-dir ./test_images --ground-truth ./ground_truth.txt
    python evaluate.py --image-dir ./test_images --ground-truth ./ground_truth.txt --output-csv results.csv
"""

import argparse
import csv
import os
import sys
import logging
import time

# pyrefly: ignore [missing-import]
from jiwer import wer, cer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}


def collect_images(image_dir: str) -> list:
    """Collect and sort image file paths from a directory."""
    images = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() in ALLOWED_EXTENSIONS:
            images.append(os.path.join(image_dir, fname))
    return images


def load_ground_truth(gt_path: str) -> list:
    """Load ground-truth lines from a text file."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def run_evaluation(image_dir: str, ground_truth_path: str, output_csv: str = None):
    """Run the full evaluation: raw EasyOCR vs. full pipeline (with spelling correction)."""
    # Import here so the heavy model loading only happens when actually evaluating
    # pyrefly: ignore [missing-import]
    import easyocr
    # pyrefly: ignore [missing-import]
    import cv2 as cv
    from image_to_text import ImageToText

    images = collect_images(image_dir)
    gt_lines = load_ground_truth(ground_truth_path)

    if len(images) != len(gt_lines):
        logger.error(
            "Mismatch: found %d images but %d ground-truth lines.",
            len(images), len(gt_lines)
        )
        sys.exit(1)

    if len(images) == 0:
        logger.error("No images found in %s", image_dir)
        sys.exit(1)

    logger.info("Evaluating %d image(s)...", len(images))

    # Initialize models
    logger.info("Loading OCR pipeline (this may take a while on first run)...")
    engine = ImageToText()
    raw_reader = easyocr.Reader(['vi', 'en'])

    raw_hypotheses = []
    full_hypotheses = []
    per_image_results = []

    for idx, (img_path, gt_text) in enumerate(zip(images, gt_lines)):
        basename = os.path.basename(img_path)
        logger.info("[%d/%d] Processing: %s", idx + 1, len(images), basename)

        # --- Raw EasyOCR (no layout, no spelling correction) ---
        img = cv.imread(img_path)
        raw_results = raw_reader.readtext(img)
        raw_text = " ".join([text for _, text, _ in raw_results]).strip()
        raw_hypotheses.append(raw_text)

        # --- Full pipeline (layout + spelling correction) ---
        start = time.time()
        full_text = engine.image_to_text(img_path).replace("\n", " ").strip()
        elapsed = time.time() - start
        full_hypotheses.append(full_text)

        # Per-image metrics
        img_raw_cer = cer([gt_text], [raw_text])
        img_raw_wer = wer([gt_text], [raw_text])
        img_full_cer = cer([gt_text], [full_text])
        img_full_wer = wer([gt_text], [full_text])

        per_image_results.append({
            "image": basename,
            "ground_truth": gt_text,
            "raw_ocr_text": raw_text,
            "full_pipeline_text": full_text,
            "raw_cer": img_raw_cer,
            "raw_wer": img_raw_wer,
            "full_cer": img_full_cer,
            "full_wer": img_full_wer,
            "time_s": elapsed,
        })

        logger.info("  GT:   %s", gt_text[:80])
        logger.info("  RAW:  %s", raw_text[:80])
        logger.info("  FULL: %s", full_text[:80])
        logger.info("  Time: %.2fs", elapsed)

    # --- Compute aggregate metrics ---
    raw_cer_agg = cer(gt_lines, raw_hypotheses)
    raw_wer_agg = wer(gt_lines, raw_hypotheses)

    full_cer_agg = cer(gt_lines, full_hypotheses)
    full_wer_agg = wer(gt_lines, full_hypotheses)

    # --- Print results table ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Pipeline Stage':<35} {'CER':>10} {'WER':>10}")
    print("-" * 60)
    print(f"{'Raw EasyOCR':<35} {raw_cer_agg:>10.2%} {raw_wer_agg:>10.2%}")
    print(f"{'+ Spelling Correction':<35} {full_cer_agg:>10.2%} {full_wer_agg:>10.2%}")
    print("-" * 60)
    print(f"{'CER Improvement':<35} {(raw_cer_agg - full_cer_agg):>+10.2%}")
    print(f"{'WER Improvement':<35} {(raw_wer_agg - full_wer_agg):>+10.2%}")
    print("=" * 60)
    print(f"\nImages evaluated: {len(images)}")

    # --- Write per-image CSV if requested ---
    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "image", "ground_truth", "raw_ocr_text", "full_pipeline_text",
                "raw_cer", "raw_wer", "full_cer", "full_wer", "time_s"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_image_results:
                writer.writerow(row)
        logger.info("Per-image results saved to %s", output_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCRSpellCorrection pipeline using CER and WER metrics."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Path to folder containing test images."
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to .txt file with one ground-truth line per image (same order)."
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save per-image results as CSV."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        logger.error("Image directory does not exist: %s", args.image_dir)
        sys.exit(1)
    if not os.path.isfile(args.ground_truth):
        logger.error("Ground truth file does not exist: %s", args.ground_truth)
        sys.exit(1)

    run_evaluation(args.image_dir, args.ground_truth, args.output_csv)


if __name__ == "__main__":
    main()
