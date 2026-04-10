"""YouTube audio download via yt-dlp."""

import os
import subprocess
from pathlib import Path

import yt_dlp

from outloud.logger import get_logger

log = get_logger("youtube")


def get_video_info(url: str) -> dict:
    """Get video metadata without downloading.

    Returns:
        dict with title, duration_min, etc.
    """
    ydl_opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Unknown"),
            "duration_min": round(info.get("duration", 0) / 60),
            "duration_sec": info.get("duration", 0),
        }


def download_audio(url: str, output_dir: Path) -> tuple[str, str]:
    """Download audio from YouTube.

    Args:
        url: YouTube URL
        output_dir: Directory to save the file

    Returns:
        (audio_path, title)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "yt_audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    # Suppress yt-dlp console output
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown")
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)

    # Find the downloaded WAV
    wav_path = str(output_dir / "yt_audio.wav")
    if os.path.exists(wav_path):
        log.info("YouTube audio downloaded: %s", wav_path)
        return wav_path, title

    log.warning("Downloaded file not found at %s", wav_path)
    return "", title
