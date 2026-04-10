"""Транскрибация аудио."""

import json
import os
from pathlib import Path

import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from outloud.config import MODELS_DIR, VOSK_MODELS, WHISPER_MODELS, SAMPLE_RATE
from outloud.utils import to_wav, cleanup_temp_wav
from outloud.logger import get_logger

log = get_logger("transcriber")

VOSK_CHUNK = 8000  # ~0.5 сек при 16kHz


def check_vosk_model(variant: str = "small") -> bool:
    """Проверить наличие модели Vosk."""
    model_info = VOSK_MODELS.get(variant)
    if not model_info:
        return False
    model_path = MODELS_DIR / model_info["name"]
    return model_path.exists()


def download_vosk_model(variant: str = "small"):
    """Скачать и распаковать модель Vosk."""
    import urllib.request
    import zipfile

    model_info = VOSK_MODELS[variant]
    url = model_info["url"]
    name = model_info["name"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / f"{name}.zip"

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded / total_size * 100, 100)
            bar_len = 30
            filled = int(bar_len * percent / 100)
            bar = "#" * filled + "." * (bar_len - filled)
            mb_down = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  [{bar}] {percent:.0f}% ({mb_down:.0f}/{mb_total:.0f}MB)", end="", flush=True)

    print(f"Downloading Vosk {variant}...")
    log.info("Downloading Vosk %s", variant)
    urllib.request.urlretrieve(url, zip_path, reporthook)
    print()

    print(f"Extracting {name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(MODELS_DIR)

    os.remove(zip_path)
    log.info("Vosk %s downloaded", variant)
    print(f"Vosk {variant} ready")


def check_whisper_model(variant: str = "small") -> bool:
    """Проверить наличие модели Whisper."""
    try:
        import whisper
        whisper.load_model(variant, device="cpu")
        return True
    except Exception:
        return False


def download_whisper_model(variant: str = "small"):
    """Скачать модель Whisper."""
    import whisper
    log.info("Downloading Whisper %s", variant)
    print(f"Downloading Whisper {variant}...")

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    model = whisper.load_model(variant, device="cpu")

    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(devnull)
    os.close(old_out)
    os.close(old_err)

    del model
    log.info("Whisper %s downloaded", variant)
    print(f"Whisper {variant} ready")


def transcribe_vosk(audio_data: np.ndarray | str, variant: str = "small",
                   sample_rate: int = SAMPLE_RATE) -> str:
    """Транскрибировать аудио через Vosk с прогресс-баром."""
    import logging
    logging.getLogger("vosk").setLevel(logging.ERROR)

    model_info = VOSK_MODELS[variant]
    model_path = MODELS_DIR / model_info["name"]

    if not model_path.exists():
        raise RuntimeError(
            f"Vosk {variant} not found. Run: outloud install-models -m vosk -v {variant}"
        )

    # Подготовка аудио
    temp_wav = None
    if isinstance(audio_data, np.ndarray):
        audio_bytes = audio_data.tobytes()
    else:
        temp_wav = to_wav(audio_data, sample_rate)
        import wave
        with wave.open(temp_wav, 'rb') as wf:
            audio_bytes = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()

    log.info("Loading Vosk %s", variant)

    # Глушим логи загрузки
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    from vosk import Model, KaldiRecognizer

    model = Model(str(model_path))
    recognizer = KaldiRecognizer(model, sample_rate)

    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(devnull)
    os.close(old_out)
    os.close(old_err)

    log.info("Transcribing with Vosk %s, %d bytes", variant, len(audio_bytes))

    # Батчим по кускам для прогресс-бара
    text_parts = []
    total = len(audio_bytes)
    processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[white]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("transcribing", total=total)

        for i in range(0, total, VOSK_CHUNK * 2):  # int16 = 2 байта
            chunk = audio_bytes[i:i + VOSK_CHUNK * 2]
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                t = result.get('text', '')
                if t:
                    text_parts.append(t)
            processed += len(chunk)
            prog.update(task, completed=min(processed, total))

        # Финальный результат
        result = json.loads(recognizer.FinalResult())
        t = result.get('text', '')
        if t:
            text_parts.append(t)

    if temp_wav:
        cleanup_temp_wav(temp_wav)

    text = ' '.join(filter(None, text_parts)).strip()
    log.info("Vosk done: %d chars", len(text))
    return text


def transcribe_whisper(audio_path: str | Path, variant: str = "small") -> str:
    """Транскрибировать аудио через Whisper."""
    import whisper

    audio_path = str(audio_path)

    temp_wav = None
    if not audio_path.lower().endswith('.wav'):
        temp_wav = to_wav(audio_path)
        audio_path = temp_wav

    log.info("Loading Whisper %s", variant)

    # Глушим логи
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    model = whisper.load_model(variant, device="cpu")

    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(devnull)
    os.close(old_out)
    os.close(old_err)

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]transcribing with Whisper"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        prog.add_task("work", total=None)  # Indeterminate
        result = model.transcribe(audio_path, language="ru")

    del model
    if temp_wav:
        cleanup_temp_wav(temp_wav)

    text = result.get('text', '')
    log.info("Whisper done: %d chars", len(text))
    return text
