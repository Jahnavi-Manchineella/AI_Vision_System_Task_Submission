import os
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ─────────────────────────────────────────────
#  Model Configuration
# ─────────────────────────────────────────────
TROCR_MODEL_ID   = "microsoft/trocr-base-printed"
TROCR_SAVE_PATH  = "trocr_model"
EASYOCR_LANGS    = ['en']

def download_easyocr() -> easyocr.Reader:
    """Initialize and return the EasyOCR reader with specified languages."""
    print("\n[1/2] Initializing EasyOCR model...")
    reader = easyocr.Reader(EASYOCR_LANGS)
    print("       EasyOCR ready.")
    return reader

def download_trocr() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Download and return the TrOCR processor and model."""
    print("\n[2/2] Fetching TrOCR model from pretrained weights...")
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
    print("       TrOCR loaded.")
    return processor, model

def save_trocr(processor: TrOCRProcessor,
               model: VisionEncoderDecoderModel,
               save_path: str) -> None:
    """Persist the TrOCR processor and model to a local directory."""
    os.makedirs(save_path, exist_ok=True)
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"\n       TrOCR saved locally → ./{save_path}")

def main() -> None:
    print("=" * 48)
    print("        OCR Model Download Pipeline")
    print("=" * 48)

    reader  = download_easyocr()
    processor, model = download_trocr()
    save_trocr(processor, model, TROCR_SAVE_PATH)

    print("\n" + "=" * 48)
    print(" All models downloaded successfully!")
    print("=" * 48 + "\n")

if __name__ == "__main__":
    main()
