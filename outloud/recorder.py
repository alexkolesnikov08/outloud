"""Audio recording with VU meter."""

import numpy as np
import sounddevice as sd

from outloud.config import SAMPLE_RATE, CHANNELS, DTYPE
from outloud.logger import get_logger

log = get_logger("recorder")


def _vu_meter(indata: np.ndarray, frames: int, time_info, status):
    """Callback that shows volume level during recording."""
    if status:
        pass  # Ignore status messages
    rms = np.sqrt(np.mean(indata ** 2))
    level = int(rms * 1000)
    bar = "█" * min(level, 30)
    print(f"\r[{bar:<30}] {level:3d}", end="", flush=True)


def record_audio() -> np.ndarray:
    """Record audio from microphone with VU meter.

    Returns:
        numpy array of int16 audio samples
    """
    log.info("Starting recording...")

    print("Recording... (Ctrl+C to stop)")
    recording = []

    def callback(indata, frames, time_info, status):
        if status:
            pass
        rms = np.sqrt(np.mean(indata ** 2))
        level = int(rms * 1000)
        bar_len = min(level // 3, 30)
        bar = "█" * bar_len
        print(f"\r[{bar:<30}] {level:3d}", end="", flush=True)
        recording.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback,
        ):
            sd.sleep(1000)  # Keep alive
    except KeyboardInterrupt:
        print()
        log.info("Recording stopped by user")

    if not recording:
        return np.array([], dtype=np.int16)

    audio = np.concatenate(recording, axis=0)
    log.info("Recording done: %d samples (%.1fs)", len(audio), len(audio) / SAMPLE_RATE)
    return audio


def save_audio(audio: np.ndarray, filepath: str):
    """Save audio to WAV file.

    Args:
        audio: numpy array of int16 samples
        filepath: Output path
    """
    import wave

    log.info("Saving audio to %s", filepath)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    log.info("Audio saved: %s", filepath)
