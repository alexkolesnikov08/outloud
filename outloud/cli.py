"""OutLoud CLI — просто и понятно."""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from outloud import __version__
from outloud.config import get_output_dir
from outloud.recorder import record_audio, save_audio
from outloud.transcriber import (
    transcribe_vosk,
    check_vosk_model,
    download_vosk_model,
)
from outloud.summarizer import summarize_text
from outloud.qwen_llm import check_qwen_model, get_pipeline as get_qwen
from outloud.cloud import (
    transcribe_cloud,
    summarize_cloud,
    correct_grammar_cloud,
    save_api_keys,
    check_keys,
    verify_keys,
)
from outloud.youtube import download_audio, get_video_info
from outloud.logger import get_logger


log = get_logger("cli")


def _save(filename: str, content: str, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding="utf-8")


def _stats(start: float, text: str, summary: str, folder: Path):
    secs = time.time() - start
    words = len(text.split())
    print()
    print(f"words: {words} | time: {secs:.0f}s | saved: {folder}")


@click.group()
@click.version_option(version=__version__)
def main():
    """OutLoud — запись, транскрибация и краткое содержание."""
    pass


# ─── setup ───────────────────────────────────────────────────────────────

@main.command()
def setup():
    """Подготовить всё для работы (один раз)."""
    print("Setting up OutLoud...")
    print()

    # Определяем железо
    import platform
    chip = platform.processor() or "unknown"
    try:
        import subprocess as _sp
        mem = _sp.check_output(
            "sysctl -n hw.memsize 2>/dev/null || echo 4294967296", shell=True
        ).decode().strip()
        mem_gb = int(mem) // (1024**3)
    except Exception:
        mem_gb = 4

    print(f"  Chip: {chip}")
    print(f"  RAM: {mem_gb}GB")
    print()

    # Транскрибация — Vosk small (всегда)
    print("Voice model: downloading (~70MB)...")
    if not check_vosk_model("small"):
        download_vosk_model("small")
    print("  Voice model: ready")

    # AI модель — под железо
    if mem_gb >= 16:
        # Могли бы запустить Qwen 4B или даже 7B
        print("  AI model: downloading Qwen 4B...")
    elif mem_gb >= 8:
        print("  AI model: downloading Qwen 0.8B 4-bit (~500MB)...")
        from outloud.qwen_llm import download_qwen_model
        download_qwen_model()
    else:
        # 4GB — только Qwen 0.8B 4-bit
        print("  AI model: downloading Qwen 0.8B 4-bit (~500MB)...")
        from outloud.qwen_llm import download_qwen_model
        download_qwen_model()

    # ffmpeg проверка
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("  Audio converter: ready")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Audio converter: NOT FOUND — brew install ffmpeg")

    print()
    print("Ready. Type: outloud record")


# ─── record ──────────────────────────────────────────────────────────────

@main.command()
@click.option('--cloud', is_flag=True, default=False, help='Облачные модели')
@click.option('--grammar', is_flag=True, default=False, help='Исправить ошибки')
def record(cloud: bool, grammar: bool):
    """Записать голос → текст → краткое содержание."""
    if cloud and not check_keys():
        print("Cloud not configured. Type: outloud cloud-setup")
        return

    folder = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = folder / f"outloud_{ts}"
    sess.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Запись
    print()
    print("Recording... (Ctrl+C to stop)")
    audio = record_audio()
    if len(audio) == 0:
        print("Recording failed")
        return

    save_audio(audio, str(sess / "audio.wav"))
    print("Audio saved")

    # Транскрибация
    if cloud:
        text = transcribe_cloud(str(sess / "audio.wav"))
    else:
        text = transcribe_vosk(audio)
    _save("transcription.txt", text, sess)
    print("Transcription done")

    # Суммаризация
    if cloud:
        summary = summarize_cloud(text)
    else:
        summary = summarize_text(text, engine="qwen")
    _save("summary.txt", summary, sess)
    print("Summary done")

    # Грамматика
    if grammar:
        if cloud:
            fixed = correct_grammar_cloud(summary)
        else:
            qwen = get_qwen()
            fixed = qwen.correct_grammar(summary)
            qwen.cleanup()
        final = fixed
        _save("corrected.txt", fixed, sess)
        print("Grammar done")
    else:
        final = summary

    _stats(start, text, final, sess)


# ─── file ────────────────────────────────────────────────────────────────

@main.command("file")
@click.argument("filepath", type=click.Path(exists=True))
@click.option('--cloud', is_flag=True, default=False, help='Облачные модели')
@click.option('--grammar', is_flag=True, default=False, help='Исправить ошибки')
def transcribe_file(filepath: str, cloud: bool, grammar: bool):
    """Транскрибировать аудиофайл."""
    if cloud and not check_keys():
        print("Cloud not configured. Type: outloud cloud-setup")
        return

    folder = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = folder / f"outloud_{ts}"
    sess.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Транскрибация
    if cloud:
        text = transcribe_cloud(filepath)
    else:
        text = transcribe_vosk(filepath)
    _save("transcription.txt", text, sess)
    print("Transcription done")

    # Суммаризация
    if cloud:
        summary = summarize_cloud(text)
    else:
        summary = summarize_text(text, engine="qwen")
    _save("summary.txt", summary, sess)
    print("Summary done")

    # Грамматика
    if grammar:
        if cloud:
            fixed = correct_grammar_cloud(summary)
        else:
            qwen = get_qwen()
            fixed = qwen.correct_grammar(summary)
            qwen.cleanup()
        final = fixed
        _save("corrected.txt", fixed, sess)
        print("Grammar done")
    else:
        final = summary

    _stats(start, text, final, sess)


# ─── yt ──────────────────────────────────────────────────────────────────

@main.command("yt")
@click.argument("url")
@click.option('--cloud', is_flag=True, default=False, help='Облачные модели')
@click.option('--grammar', is_flag=True, default=False, help='Исправить ошибки')
def youtube(url: str, cloud: bool, grammar: bool):
    """Транскрибировать YouTube видео."""
    if cloud and not check_keys():
        print("Cloud not configured. Type: outloud cloud-setup")
        return

    folder = get_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = folder / f"outloud_{ts}"
    sess.mkdir(parents=True, exist_ok=True)
    start = time.time()

    info = get_video_info(url)
    print(f"Video: {info['title']}")
    print(f"Duration: {info['duration_min']} min")

    audio_path, _ = download_audio(url, sess)
    print("Audio downloaded")

    # Транскрибация
    if cloud:
        text = transcribe_cloud(audio_path)
    else:
        text = transcribe_vosk(audio_path)
    _save("transcription.txt", text, sess)
    print("Transcription done")

    # Суммаризация
    if cloud:
        summary = summarize_cloud(text)
    else:
        summary = summarize_text(text, engine="qwen")
    _save("summary.txt", summary, sess)
    print("Summary done")

    # Грамматика
    if grammar:
        if cloud:
            fixed = correct_grammar_cloud(summary)
        else:
            qwen = get_qwen()
            fixed = qwen.correct_grammar(summary)
            qwen.cleanup()
        final = fixed
        _save("corrected.txt", fixed, sess)
        print("Grammar done")
    else:
        final = summary

    _stats(start, text, final, sess)


# ─── cloud-setup ──────────────────────────────────────────────────────────

@main.command("cloud-setup")
def cloud_setup():
    """Подключить облачные модели (Whisper + GPT-OSS + Llama)."""
    print("OutLoud Cloud Setup")
    print()
    print("Нужен ключ Groq (бесплатно): https://console.groq.com/keys")
    print()
    print("Облачные модели:")
    print("  Расшифровка: Whisper Large v3 (H100)")
    print("  Суммаризация: GPT-OSS 20B")
    print("  Грамматика: Llama 3.1 8B")
    print()

    key = input("Groq API key: ").strip()
    if not key:
        print("Key is required")
        return

    print()
    print("Saving...")
    save_api_keys(key)

    print("Checking...")
    if verify_keys():
        print("Ready. Use: outloud record --cloud")
    else:
        print("Warning: verification failed. Check key and try again.")


# ─── cloud-status ──────────────────────────────────────────────────────────

@main.command("cloud-status")
def cloud_status():
    """Проверить облачные ключи."""
    if not check_keys():
        print("Cloud not configured. Run: outloud cloud-setup")
        return
    print("Keys: configured")
    if verify_keys():
        print("Status: OK")
    else:
        print("Status: error (check https://console.groq.com/settings/project/limits)")


if __name__ == "__main__":
    main()
