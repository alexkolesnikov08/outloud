"""Text summarization — extractive or Qwen."""

import gc
import re
from collections import Counter

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from outloud.logger import get_logger

log = get_logger("summarizer")

STOP_WORDS = {
    'и', 'в', 'не', 'на', 'я', 'с', 'он', 'а', 'по', 'это', 'но',
    'как', 'то', 'что', 'из', 'у', 'мы', 'ты', 'за', 'к', 'до',
    'или', 'уже', 'может', 'быть', 'нет', 'да', 'сейчас', 'тут',
    'там', 'все', 'всё', 'его', 'её', 'их', 'мой', 'твой', 'наш',
    'для', 'от', 'о', 'об', 'со', 'же', 'ли', 'бы', 'вот',
    'так', 'более', 'менее', 'очень', 'просто', 'тоже', 'только',
    'при', 'про', 'без', 'через', 'после', 'когда', 'если', 'чем',
    'ну', 'вообще', 'потом', 'ещё', 'уже', 'даже', 'сам', 'себя',
    'меня', 'мне', 'мной', 'нас', 'нам', 'этом', 'этот',
    'тот', 'та', 'те', 'тех', 'тем', 'теми',
}

SENTENCES_PER_BATCH = 50
TARGET_SUMMARY_RATIO = 0.15


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r'([.!?]+)', text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ''
        if sent:
            sentences.append(sent + punct)

    if len(sentences) <= 1:
        words = text.split()
        chunk = 15
        sentences = [' '.join(words[i:i+chunk]) for i in range(0, len(words), chunk)]

    return [s for s in sentences if s.strip()]


def summarize_extractive(text: str) -> str:
    """Extractive summarization — fast, no ML."""
    if not text.strip():
        return ""

    sentences = _split_sentences(text)
    if len(sentences) <= 5:
        return text

    target_count = max(3, int(len(sentences) * TARGET_SUMMARY_RATIO))

    words = ' '.join(sentences).lower().split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    word_freq = Counter(words)
    if not word_freq:
        return '. '.join(sentences[:target_count]) + '.'

    max_freq = max(word_freq.values())
    for w in word_freq:
        word_freq[w] /= max_freq

    all_scored = []
    for i, sent in enumerate(sentences):
        sent_words = [w.lower() for w in sent.split()
                      if w.lower() not in STOP_WORDS and len(w) > 2]
        if not sent_words:
            all_scored.append((i, 0.0))
            continue

        score = sum(word_freq.get(w, 0) for w in sent_words) / len(sent_words)
        if i < 3:
            score += 0.3 / (i + 1)
        if i >= len(sentences) - 3:
            score += 0.2

        all_scored.append((i, score))

    all_scored.sort(key=lambda x: x[1], reverse=True)
    top = []
    seen = set()
    for idx, score in all_scored:
        sent_text = sentences[idx].strip().lower()
        if sent_text not in seen and len(top) < target_count:
            top.append(idx)
            seen.add(sent_text)

    top_indices = sorted(top)
    summary = ' '.join(sentences[i] for i in top_indices)

    gc.collect()
    log.info("Extractive summary: %d -> %d sentences", len(sentences), len(top))
    return summary


def summarize_qwen(text: str) -> str:
    """Summarization via Qwen 0.8B 4-bit."""
    from outloud.qwen_llm import get_pipeline
    pipeline = get_pipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[white]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[white]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task("summarizing (qwen)", total=None)
        result = pipeline.summarize(text)

    gc.collect()
    return result


def summarize_text(text: str, engine: str = "extractive") -> str:
    """
    Summarize text.

    Args:
        text: Input text
        engine: "extractive" (fast) or "qwen" (quality)
    """
    if engine == "qwen":
        return summarize_qwen(text)
    return summarize_extractive(text)
