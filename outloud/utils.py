"""Audio format conversion utilities."""

import os
from pathlib import Path

from pydub import AudioSegment

from outloud.logger import get_logger

log = get_logger("utils")


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """Convert audio file to WAV format.

    Args:
        input_path: Path to input file (m4a, mp3, ogg, wav)
        output_path: Path to output WAV (defaults to same name + .wav)

    Returns:
        Path to the WAV file
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".wav"))

    log.info("Converting %s -> %s", input_path, output_path)

    ext = Path(input_path).suffix.lower()
    if ext == ".mp3":
        audio = AudioSegment.from_mp3(input_path)
    elif ext == ".m4a":
        audio = AudioSegment.from_file(input_path, format="m4a")
    elif ext == ".ogg":
        audio = AudioSegment.from_ogg(input_path)
    elif ext == ".wav":
        return input_path  # Already WAV
    else:
        audio = AudioSegment.from_file(input_path)

    # Convert to mono, 16kHz for Vosk
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(output_path, format="wav")

    log.info("Conversion done: %s", output_path)
    return output_path
