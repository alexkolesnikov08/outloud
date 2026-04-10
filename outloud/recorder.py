"""Запись аудио с микрофона + VU-метр."""

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from outloud.config import SAMPLE_RATE, CHANNELS


VU_CHARS = " _-~=+*#%@"
VU_WIDTH = 40


def _rms_level(data: np.ndarray) -> float:
    """RMS уровень громкости (0-1)."""
    if len(data) == 0:
        return 0.0
    normalized = data.astype(np.float32) / 32767.0
    rms = np.sqrt(np.mean(normalized ** 2))
    return min(rms * 5, 1.0)


def _render_vu(level: float) -> str:
    """VU-метр строкой."""
    filled = int(level * VU_WIDTH)
    bar = ''.join(VU_CHARS[-1] if i < filled else ' ' for i in range(VU_WIDTH))
    db = int(level * 100)
    return f"[{bar}] {db:3d}%"


def record_audio(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Записать аудио с микрофона до Ctrl+C с VU-метром."""
    frames = []
    recording = False
    current_vu = ""

    def callback(indata, frames_count, time_info, status):
        nonlocal current_vu
        if recording:
            frames.append(indata.copy())
            level = _rms_level(indata)
            current_vu = "\r" + _render_vu(level)

    print("Recording... (Ctrl+C to stop)")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype='int16',
            callback=callback
        ):
            recording = True
            while True:
                sd.sleep(100)
                if current_vu:
                    print(current_vu, end="", flush=True)
    except KeyboardInterrupt:
        print()
        print("Recording stopped")
        recording = False

    if frames:
        return np.concatenate(frames, axis=0)
    return np.array([], dtype='int16')


def save_audio(data: np.ndarray, filepath: str, sample_rate: int = SAMPLE_RATE) -> str:
    """Сохранить аудио в WAV файл."""
    write(filepath, sample_rate, data)
    return filepath
