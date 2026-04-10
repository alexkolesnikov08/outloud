"""OutLoud Cloud — Groq + fallback на локальные модели."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from outloud.logger import get_logger

log = get_logger("cloud")

KEYS_FILE = Path.home() / ".outloud" / "api_keys.json"

# Модели — Groq
WHISPER_MODEL = "whisper-large-v3-turbo"
# Цепочка fallback для суммаризации
SUMMARY_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
# Цепочка fallback для грамматики
GRAMMAR_MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
LOCAL_SUMMARY_MODEL = "llama-3.1-8b-instant"
LOCAL_GRAMMAR_MODEL = "llama-3.1-8b-instant"


def _ensure_keys_dir():
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)


def save_api_keys(groq_key: str):
    _ensure_keys_dir()
    data = {
        "groq_api_key": groq_key,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(KEYS_FILE, 0o600)


def load_api_keys() -> Optional[dict]:
    if not KEYS_FILE.exists():
        return None
    with open(KEYS_FILE) as f:
        return json.load(f)


def check_keys() -> bool:
    return load_api_keys() is not None


def _get_client() -> OpenAI:
    keys = load_api_keys()
    if not keys:
        raise RuntimeError("API keys not configured. Run: outloud cloud-setup")
    return OpenAI(
        api_key=keys["groq_api_key"],
        base_url="https://api.groq.com/openai/v1"
    )


def _is_rate_limit(err) -> bool:
    """Проверить, что ошибка — лимиты."""
    msg = str(err).lower()
    return any(k in msg for k in [
        "rate_limit", "quota", "limit", "rate limit",
        "insufficient_quota", "too_many", "429"
    ])


def _chat_with_fallback(client: OpenAI, model_list: list[str],
                        messages: list[dict], **kwargs) -> str:
    """Попробовать модели по очереди, при лимитах — следующую."""
    last_err = None

    for model in model_list:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, **kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if _is_rate_limit(e):
                log.warning("Rate limit on %s, trying next model", model)
                continue
            raise

    # Все модели Groq исчерпаны — последняя строка
    print("⚠ Groq лимиты исчерпаны — используем локальные модели")
    return ""


def _chunk_audio(audio_path: str, chunk_sec: int = 600) -> list[str]:
    """Разбить аудио на чанки по chunk_sec секунд."""
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_sec * 1000):
        chunk = audio[i:i + chunk_sec * 1000]
        tmp = audio_path + f".chunk_{i}.wav"
        chunk.export(tmp, format="wav")
        chunks.append(tmp)
    return chunks


def transcribe_cloud(audio_path: str) -> str:
    """Расшифровка: Whisper Large v3 Turbo. Fallback → локальный Vosk."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    log.info("Cloud transcription: %s", audio_path)
    client = _get_client()

    # Проверяем размер — если > 20MB, чанкуем
    file_size = os.path.getsize(audio_path)
    is_large = file_size > 20 * 1024 * 1024

    if is_large:
        return _transcribe_chunks(client, audio_path)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[white]transcribing (whisper-turbo)"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            transient=True,
        ) as prog:
            prog.add_task("work", total=None)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="text",
                    language="ru",
                )

        text = result.strip() if isinstance(result, str) else result.text.strip()
        log.info("Cloud transcription done: %d words", len(text.split()))
        return text

    except Exception as e:
        if "blocked" in str(e).lower() or "permission" in str(e).lower():
            log.warning("Whisper blocked, fallback to local Vosk")
            from outloud.transcriber import transcribe_vosk
            return transcribe_vosk(audio_path)
        if _is_rate_limit(e):
            print("⚠ Groq лимиты — транскрибация локально")
            from outloud.transcriber import transcribe_vosk
            return transcribe_vosk(audio_path)
        if "413" in str(e) or "too large" in str(e).lower():
            return _transcribe_chunks(client, audio_path)
        raise


def _transcribe_chunks(client: OpenAI, audio_path: str) -> str:
    """Транскрибация большого файла по частям."""
    from pydub import AudioSegment
    import tempfile

    audio = AudioSegment.from_mp3(audio_path)
    chunk_ms = 300_000  # 5 минут — mp3 будет ~5MB
    total_chunks = max(1, len(audio) // chunk_ms + (1 if len(audio) % chunk_ms else 0))

    print(f"File is large — splitting into {total_chunks} chunks")

    parts = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        chunk_num = i // chunk_ms + 1

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[white]transcribing chunk {chunk_num}/{total_chunks}"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            transient=True,
        ) as prog:
            prog.add_task("work", total=None)

            # Экспортируем в mp3 для экономии места
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                chunk.export(tmp.name, format="mp3", bitrate="64k")
                with open(tmp.name, "rb") as f:
                    result = client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=f,
                        response_format="text",
                        language="ru",
                    )
                os.unlink(tmp.name)

            text = result.strip() if isinstance(result, str) else result.text.strip()
            parts.append(text)

    full_text = " ".join(parts)
    log.info("Cloud transcription (chunked) done: %d words", len(full_text.split()))
    return full_text


def summarize_cloud(text: str) -> str:
    """Суммаризация: GPT-OSS 20B → Qwen 32B → Llama 70B → Llama 8B."""
    if not text.strip():
        return ""
    words = text.split()
    if len(words) < 20:
        return text

    log.info("Cloud summarizing: %d words", len(words))
    client = _get_client()

    messages = [
        {"role": "system", "content": (
            "Кратко перескажи текст. 2-4 предложения. "
            "Только результат, без вводных слов."
        )},
        {"role": "user", "content": f"Перескажи кратко:\n\n{text}"}
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]summarizing (gpt-oss-20b)"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        prog.add_task("work", total=None)
        result = _chat_with_fallback(
            client, SUMMARY_MODELS, messages,
            temperature=0.3, top_p=0.9, max_tokens=512)

    if result:
        log.info("Cloud summary done: %d words", len(result.split()))
        return result

    # Fallback на локальную модель
    from outloud.summarizer import summarize_text
    return summarize_text(text, engine="qwen")


def correct_grammar_cloud(text: str) -> str:
    """Грамматика: Llama 3.1 8B → Llama 4 Scout."""
    if not text.strip():
        return text

    log.info("Cloud grammar correction: %d chars", len(text))
    client = _get_client()

    messages = [
        {"role": "system", "content": (
            "Исправь грамматические и пунктуационные ошибки. "
            "НЕ меняй смысл. НЕ добавляй от себя. "
            "Выведи только исправленный текст."
        )},
        {"role": "user", "content": f"Исправь ошибки:\n\n{text}"}
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]correcting grammar (llama-3.1-8b)"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        prog.add_task("work", total=None)
        result = _chat_with_fallback(
            client, GRAMMAR_MODELS, messages,
            temperature=0.2, top_p=0.9, max_tokens=len(text.split()) * 2)

    if result:
        log.info("Cloud grammar done: %d chars", len(result))
        return result

    # Fallback на локальную модель
    from outloud.qwen_llm import get_pipeline
    qwen = get_pipeline()
    fixed = qwen.correct_grammar(text)
    qwen.cleanup()
    return fixed


def verify_keys() -> bool:
    try:
        client = _get_client()
        models = client.models.list()
        return len(models.data) > 0
    except Exception as e:
        log.error("Key verification failed: %s", e)
        return False
