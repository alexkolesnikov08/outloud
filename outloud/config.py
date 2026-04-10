"""Конфигурация проекта."""

import os
from pathlib import Path


# Пути
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "outloud-logs"
OUTPUT_DIR = Path.home() / "Desktop"

# Модели Vosk
VOSK_MODELS = {
    "small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        "name": "vosk-model-small-ru-0.22",
        "size": "70MB",
    },
    "medium": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
        "name": "vosk-model-ru-0.42",
        "size": "800MB",
    },
    "large": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42-large.zip",
        "name": "vosk-model-ru-0.42-large",
        "size": "2.3GB",
    },
}

# Модели Whisper
WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
}

# Модели суммаризации
SUMMARY_MODELS = {
    "extractive": "extractive",  # Быстрая, без ML
    "qwen": "qwen",              # Qwen3.5-0.8B, качество
}

# Модель коррекции грамматики
GRAMMAR_MODELS = {
    "mbart": {
        "name": "MRNH/mbart-russian-grammar-corrector",
        "size": "2.4GB",
        "src_lang": "ru_RU",
        "tgt_lang": "ru_RU",
    },
    "qwen": {
        "name": "Qwen/Qwen3.5-0.8B",
        "size": "1.8GB",
    },
}

# Параметры записи
SAMPLE_RATE = 16000
CHANNELS = 1


def get_output_dir() -> Path:
    """Получить директорию для вывода."""
    output_env = os.environ.get("OUTLOUD_OUTPUT_DIR")
    if output_env:
        return Path(output_env)
    return OUTPUT_DIR


def get_models_dir() -> Path:
    """Получить директорию для моделей."""
    models_env = os.environ.get("OUTLOUD_MODELS_DIR")
    if models_env:
        return Path(models_env)
    return MODELS_DIR
