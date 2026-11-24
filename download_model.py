"""
Model Download & Verification Utility for AABAR CV Service.

Downloads and validates the official MediaPipe Face Landmarker pretrained task model:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"
DEFAULT_MODELS_DIR = Path(__file__).parent / "models"
MIN_FILE_SIZE_BYTES = 3_000_000  # Expected size ~3.6 MB


def download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(
            f"\rDownloading model: {percent:.1f}% ({downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB)"
        )
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading model: {downloaded / (1024 * 1024):.2f} MB")
        sys.stdout.flush()


def ensure_model(dest_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """
    Ensure the MediaPipe face_landmarker.task model exists and is valid.

    Args:
        dest_dir: Directory where the model should reside.

    Returns:
        Path to the verified model file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / MODEL_FILENAME

    if model_path.exists():
        size = model_path.stat().st_size
        if size >= MIN_FILE_SIZE_BYTES:
            print(f"[OK] Model already exists: {model_path} ({size / (1024 * 1024):.2f} MB)")
            return model_path
        else:
            print(f"[WARNING] Model file exists but appears truncated ({size} bytes). Re-downloading...")
            model_path.unlink()

    print(f"Downloading MediaPipe Face Landmarker model from:\n  {MODEL_URL}\nTo:\n  {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(model_path), reporthook=download_progress_hook)
        print("\n[OK] Model download completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}", file=sys.stderr)
        if model_path.exists():
            model_path.unlink()
        raise

    # Verify downloaded file
    size = model_path.stat().st_size
    if size < MIN_FILE_SIZE_BYTES:
        raise ValueError(f"Downloaded model size ({size} bytes) is suspiciously small. Download may have failed.")

    print(f"[OK] Verified model at {model_path} ({size / (1024 * 1024):.2f} MB)")
    return model_path


if __name__ == "__main__":
    ensure_model()
