"""Configuration for OutLoud."""

import platform
import subprocess
from pathlib import Path


# Default output directory
OUTPUT_DIR = Path.home() / "Desktop"

# Vosk models
VOSK_MODELS = {
    "small": {
        "name": "vosk-model-small-ru-0.22",
        "size": "70MB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
    },
    "medium": {
        "name": "vosk-model-ru-0.42",
        "size": "800MB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
    },
    "large": {
        "name": "vosk-model-large-ru-0.42",
        "size": "2.3GB",
        "url": "https://alphacephei.com/vosk/models/vosk-model-large-ru-0.42.zip",
    },
}

# Summary models
SUMMARY_MODELS = {
    "extractive": "extractive",  # Fast, no ML
    "qwen": "qwen",              # Qwen 0.8B 4-bit via MLX
}

# Grammar models
GRAMMAR_MODELS = {
    "qwen": {
        "name": "Qwen 0.8B 4-bit",
        "size": "500MB",
    },
}

# Recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def get_models_dir() -> Path:
    """Get the models directory."""
    return Path(__file__).parent.parent / "models"


def get_output_dir() -> Path:
    """Get the default output directory."""
    return OUTPUT_DIR


def detect_hardware() -> dict:
    """Detect hardware capabilities."""
    chip = platform.processor() or "unknown"
    try:
        mem = subprocess.check_output(
            "sysctl -n hw.memsize 2>/dev/null || echo 4294967296", shell=True
        ).decode().strip()
        mem_gb = int(mem) // (1024**3)
    except Exception:
        mem_gb = 4

    # Choose best model for hardware
    if mem_gb >= 16:
        vosk_size = "medium"
        ai_model = "qwen4b"
    elif mem_gb >= 8:
        vosk_size = "small"
        ai_model = "qwen"
    else:
        vosk_size = "small"
        ai_model = "qwen"

    return {
        "chip": chip,
        "ram_gb": mem_gb,
        "vosk_size": vosk_size,
        "ai_model": ai_model,
    }
