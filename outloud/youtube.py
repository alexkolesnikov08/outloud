"""YouTube - скачивание аудио из видео с прогресс-баром."""

import os
import sys
import tempfile
from pathlib import Path

from outloud.logger import get_logger

log = get_logger("youtube")


class ProgressHook:
    """Хук для отображения прогресса скачивания."""

    def __init__(self):
        self.current_status = ""
        self.last_printed = ""

    def __call__(self, d):
        if d['status'] == 'downloading':
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            speed = d.get('speed', 0)
            eta = d.get('eta', 0)

            if total > 0:
                pct = min(downloaded / total * 100, 100)
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "#" * filled + "." * (bar_len - filled)

                # Скорость
                if speed:
                    if speed > 1024 * 1024:
                        spd = f"{speed / 1024 / 1024:.1f}MB/s"
                    else:
                        spd = f"{speed / 1024:.0f}KB/s"
                else:
                    spd = "??"

                # ETA
                if eta:
                    eta_str = f"{eta // 60}m{eta % 60}s"
                else:
                    eta_str = "??"

                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024

                status = f"\r  [{bar}] {pct:.0f}% ({mb:.0f}/{total_mb:.0f}MB) {spd} ETA:{eta_str}"
                if status != self.last_printed:
                    print(status, end="", flush=True)
                    self.last_printed = status

        elif d['status'] == 'finished':
            filename = d.get('filename', '')
            size = os.path.getsize(filename) if os.path.exists(filename) else 0
            mb = size / 1024 / 1024
            print(f"\r  [##############################] 100% ({mb:.0f}MB)              ")


def download_audio(url: str, output_dir: Path | None = None) -> tuple[str, str]:
    """
    Скачать аудио из YouTube видео с прогресс-баром.

    Args:
        url: URL YouTube видео
        output_dir: Куда сохранить

    Returns:
        (путь к файлу, название видео)
    """
    import yt_dlp

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="outloud_yt_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    hook = ProgressHook()

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'retries': 10,
        'fragment_retries': 10,
        'socket_timeout': 30,
        'progress_hooks': [hook],
    }

    log.info("Downloading audio from: %s", url)
    print("Downloading video audio...")

    # Глушим stderr (логи yt-dlp), stdout оставляем для прогресс-бара
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'unknown')
            video_id = info.get('id', 'unknown')

            # Ищем скачанный файл
            audio_path = None
            for ext in ['.m4a', '.mp3', '.webm', '.opus', '.wav']:
                candidate = output_dir / f"{video_id}{ext}"
                if candidate.exists():
                    audio_path = str(candidate)
                    break

            # Если не нашли по ID — ищем любой аудиофайл
            if audio_path is None:
                for f in sorted(output_dir.iterdir()):
                    if f.suffix in ('.m4a', '.mp3', '.webm', '.opus', '.wav'):
                        audio_path = str(f)
                        break

            if audio_path is None:
                raise RuntimeError("Audio file not found after download")

    except Exception as e:
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_err)
        print()  # Новая строка после прогресс-бара
        raise RuntimeError(f"Failed to download: {e}")

    os.dup2(old_err, 2)
    os.close(devnull)
    os.close(old_err)

    print()  # Новая строка после прогресс-бара
    log.info("Audio downloaded: %s (%.1fMB)", audio_path,
             os.path.getsize(audio_path) / 1024 / 1024)
    print(f"Downloaded: {title}")

    return audio_path, title


def get_video_info(url: str) -> dict:
    """Получить информацию о видео без скачивания."""
    import yt_dlp

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    finally:
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_err)

    duration = info.get('duration', 0)
    return {
        'title': info.get('title', 'unknown'),
        'duration': duration,
        'duration_min': duration // 60 if duration else 0,
        'channel': info.get('channel', info.get('uploader', 'unknown')),
    }
