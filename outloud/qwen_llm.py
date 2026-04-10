"""Qwen3.5-0.8B 4-bit через MLX — суммаризация и коррекция грамматики."""

import gc
import re

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

from outloud.logger import get_logger

log = get_logger("qwen")

MODEL_NAME = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
MODEL_SIZE_MB = 500  # 4-bit квантование


class QwenPipeline:
    """Qwen3.5 4-bit через MLX — нативно для Apple Silicon."""

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Ленивая загрузка модели."""
        if self._model is not None:
            return

        log.info("Loading %s (4-bit MLX)", MODEL_NAME)
        self._model, self._tokenizer = load(MODEL_NAME)
        log.info("Qwen loaded (~%dMB 4-bit)", MODEL_SIZE_MB)

    def _run(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Запустить генерацию."""
        self._load()

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            enable_thinking=False)

        sampler = make_sampler(temp=0.3, top_p=0.9)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

        mx.clear_cache()
        gc.collect()

        return response.strip()

    def summarize(self, text: str) -> str:
        """Суммаризировать текст."""
        if not text.strip():
            return ""

        words = text.split()
        if len(words) < 20:
            return text

        log.info("Qwen summarizing: %d words", len(words))

        if len(text) > 6000:
            return self._summarize_long(text)

        messages = [
            {"role": "system", "content": (
                "Ты помогаешь кратко пересказать текст. "
                "Выдели главную мысль в 2-4 предложениях. "
                "Пиши только результат, без вводных слов."
            )},
            {"role": "user", "content": f"Перескажи кратко:\n\n{text}"}
        ]

        return self._run(messages, max_tokens=256)

    def _summarize_long(self, text: str) -> str:
        """Суммаризация длинного текста батчами."""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        batches = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 6000 and current:
                batches.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        if current:
            batches.append(current)

        log.info("Long text: %d batches", len(batches))

        partials = []
        for batch in batches:
            messages = [
                {"role": "system", "content": (
                    "Кратко перескажи суть текста. 2-3 предложения. "
                    "Только результат."
                )},
                {"role": "user", "content": f"Перескажи:\n\n{batch}"}
            ]
            partials.append(self._run(messages, max_tokens=200))

        if len(partials) > 1:
            combined = "\n".join(partials)
            messages = [
                {"role": "system", "content": (
                    "Объедини в один краткий текст. 2-4 предложения. "
                    "Только результат."
                )},
                {"role": "user", "content": f"Объедини:\n\n{combined}"}
            ]
            return self._run(messages, max_tokens=256)

        return partials[0] if partials else ""

    def correct_grammar(self, text: str) -> str:
        """Исправить грамматику и пунктуацию."""
        if not text.strip():
            return text

        log.info("Qwen grammar correction: %d chars", len(text))

        sentences = re.split(r'(?<=[.!?])\s+', text)

        batches = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > 1500 and current:
                batches.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        if current:
            batches.append(current)

        corrected = []
        for batch in batches:
            messages = [
                {"role": "system", "content": (
                    "Исправь грамматические и пунктуационные ошибки. "
                    "НЕ меняй смысл. НЕ добавляй от себя. "
                    "Выведи только исправленный текст."
                )},
                {"role": "user", "content": f"Исправь:\n\n{batch}"}
            ]
            c = self._run(messages, max_tokens=len(batch.split()) * 2)
            corrected.append(c)

        result = ' '.join(corrected)
        mx.clear_cache()
        gc.collect()
        return result

    def cleanup(self):
        """Освободить память."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        mx.clear_cache()
        gc.collect()


_pipeline = None


def get_pipeline() -> QwenPipeline:
    """Синглтон."""
    global _pipeline
    if _pipeline is None:
        _pipeline = QwenPipeline()
    return _pipeline


def check_qwen_model() -> bool:
    """Проверить, скачана ли модель."""
    try:
        import mlx.core as mx
        from mlx_lm import load
        model, tok = load(MODEL_NAME)
        del model, tok
        mx.clear_cache()
        return True
    except Exception:
        return False


def download_qwen_model():
    """Скачать модель."""
    log.info("Downloading %s", MODEL_NAME)
    print(f"Downloading Qwen 4-bit ({MODEL_SIZE_MB}MB)...")
    model, tok = load(MODEL_NAME)
    del model, tok
    mx.clear_cache()
    gc.collect()
    log.info("Qwen downloaded")
    print("Qwen model ready")
