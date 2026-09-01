from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel

# TODO:: integration and testing
def load_python_script(file_path: str) -> str:
    """
    Load a Python script and return its contents as a string.

    Args:
        file_path: Path to the .py file.

    Returns:
        The entire source code as a string.
    """
    path_ = f"artifacts/{file_path}"
    return Path(path_).read_text(encoding="utf-8")


def load_excel(file_path: str):
    """Add Description"""
    path_ = f"artifacts/{file_path}"
    try:
        sheets = pd.read_excel(path_, sheet_name=None)  # dict of all sheets
    except Exception as e:
        return f"Error reading Excel file: {e}"
    
    return sheets

def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """Tier 1 baseline — no manual preprocessing. Library handles
    decode -> mono -> 16kHz -> float32 -> 30s chunking internally."""
    path_ = f"artifacts/{file_path}"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(path_, beam_size=5)
    transcript = " ".join(seg.text.strip() for seg in segments)
    transcript = f"Text transcript: {transcript}"
    return transcript