"""Утилиты для работы с аудио."""

from pathlib import Path
import tempfile
import os

from pydub import AudioSegment


def to_wav(audio_path: str | Path, sample_rate: int = 16000) -> str:
    """Конвертировать любой аудиоформат в WAV."""
    audio_path = Path(audio_path)

    if audio_path.suffix.lower() == '.wav':
        return str(audio_path)

    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)

    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(wav_file.name, format="wav")

    return wav_file.name


def cleanup_temp_wav(wav_path: str):
    """Удалить временный WAV файл."""
    if wav_path and os.path.exists(wav_path):
        try:
            os.unlink(wav_path)
        except OSError:
            pass
