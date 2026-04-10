"""Qwen 0.8B 4-bit via MLX — summarization and grammar correction."""

import gc
import re

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

from outloud.logger import get_logger

log = get_logger("qwen")

MODEL_NAME = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
MODEL_SIZE_MB = 500  # 4-bit quantized


class QwenPipeline:
    """Qwen 0.8B 4-bit via MLX — native for Apple Silicon."""

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy load model."""
        if self._model is not None:
            return

        log.info("Loading %s (4-bit MLX)", MODEL_NAME)
        self._model, self._tokenizer = load(MODEL_NAME)
        log.info("Qwen loaded (~%dMB 4-bit)", MODEL_SIZE_MB)

    def _run(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Run generation."""
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
        """Summarize text."""
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
                "You help summarize text. "
                "Highlight the main idea in 2-4 sentences. "
                "Write only the result, no filler words."
            )},
            {"role": "user", "content": f"Summarize briefly:\n\n{text}"}
        ]

        return self._run(messages, max_tokens=256)

    def _summarize_long(self, text: str) -> str:
        """Summarize long text in batches."""
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
                    "Briefly summarize the text. 2-3 sentences. "
                    "Only the result."
                )},
                {"role": "user", "content": f"Summarize:\n\n{batch}"}
            ]
            partials.append(self._run(messages, max_tokens=200))

        if len(partials) > 1:
            combined = "\n".join(partials)
            messages = [
                {"role": "system", "content": (
                    "Combine these partial summaries into one. 2-4 sentences. "
                    "Only the result."
                )},
                {"role": "user", "content": f"Combine:\n\n{combined}"}
            ]
            return self._run(messages, max_tokens=256)

        return partials[0] if partials else ""

    def correct_grammar(self, text: str) -> str:
        """Fix grammar and punctuation."""
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
                    "Fix grammatical and punctuation errors. "
                    "DO NOT change the meaning. DO NOT add anything. "
                    "Output only the corrected text."
                )},
                {"role": "user", "content": f"Fix errors:\n\n{batch}"}
            ]
            c = self._run(messages, max_tokens=len(batch.split()) * 2)
            corrected.append(c)

        result = ' '.join(corrected)
        mx.clear_cache()
        gc.collect()
        return result

    def cleanup(self):
        """Free memory."""
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
    """Get singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = QwenPipeline()
    return _pipeline


def check_qwen_model() -> bool:
    """Check if model is downloaded."""
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
    """Download the model."""
    log.info("Downloading %s", MODEL_NAME)
    print(f"Downloading Qwen 4-bit ({MODEL_SIZE_MB}MB)...")
    model, tok = load(MODEL_NAME)
    del model, tok
    mx.clear_cache()
    gc.collect()
    log.info("Qwen downloaded")
    print("Qwen model ready")
