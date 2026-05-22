import os
import easyocr
from jiwer import wer, cer
from image_to_text import ImageToText

IMAGE_PATH = "mau-cv-xin-viec-don-gian-image-1.jpg"

# Ground truth: manually transcribed from mau-cv-xin-viec-don-gian-image-1.jpg
# Fill this in ONLY if you can read the image content — otherwise leave as empty string
# and the script will print raw vs corrected output for manual inspection
GROUND_TRUTH = ""

def raw_ocr(reader, image_path):
    results = reader.readtext(image_path)
    return " ".join([text for _, text, _ in results])

def main():
    assert os.path.exists(IMAGE_PATH), f"Image not found: {IMAGE_PATH}"

    print(f"Loading models...")
    reader = easyocr.Reader(['vi', 'en'])
    engine = ImageToText()

    print(f"\nRunning Raw EasyOCR on {IMAGE_PATH}...")
    raw = raw_ocr(reader, IMAGE_PATH)

    print(f"Running Full Pipeline (+ Spelling Correction)...")
    full = engine.image_to_text(IMAGE_PATH).strip()

    print("\n══════════════════════════════════════")
    print("RAW EasyOCR OUTPUT:")
    print("══════════════════════════════════════")
    print(raw)

    print("\n══════════════════════════════════════")
    print("FULL PIPELINE OUTPUT (+ Spell Correction):")
    print("══════════════════════════════════════")
    print(full)

    if GROUND_TRUTH:
        raw_cer_score  = cer([GROUND_TRUTH], [raw])
        raw_wer_score  = wer([GROUND_TRUTH], [raw])
        full_cer_score = cer([GROUND_TRUTH], [full])
        full_wer_score = wer([GROUND_TRUTH], [full])

        print("\n┌─────────────────────────┬────────┬────────┐")
        print(  "│ Stage                   │  CER   │  WER   │")
        print(  "├─────────────────────────┼────────┼────────┤")
        print(f"│ Raw EasyOCR             │{raw_cer_score*100:5.2f}% │{raw_wer_score*100:5.2f}% │")
        print(f"│ + Spelling Correction   │{full_cer_score*100:5.2f}% │{full_wer_score*100:5.2f}% │")
        print(  "└─────────────────────────┴────────┴────────┘")
    else:
        print("\n[INFO] GROUND_TRUTH is empty — skipping CER/WER calculation.")
        print("[INFO] Review the two outputs above, then fill in GROUND_TRUTH in evaluate.py to get metrics.")

if __name__ == "__main__":
    main()
