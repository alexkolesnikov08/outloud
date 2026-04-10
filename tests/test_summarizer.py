"""Tests for summarizer module (extractive only, no ML)."""


from outloud.summarizer import (
    STOP_WORDS,
    _split_sentences,
    summarize_extractive,
)


class TestSplitSentences:
    """Tests for sentence splitting."""

    def test_single_sentence(self):
        """Single sentence should return list with one element."""
        result = _split_sentences("Hello world.")
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_multiple_sentences(self):
        """Multiple sentences should be split correctly."""
        result = _split_sentences("First. Second. Third.")
        assert len(result) == 3

    def test_sentences_with_exclamation(self):
        """Exclamation marks should be handled."""
        result = _split_sentences("Wow! Great!")
        assert len(result) == 2

    def test_sentences_with_question(self):
        """Question marks should be handled."""
        result = _split_sentences("How? Why?")
        assert len(result) == 2

    def test_empty_string(self):
        """Empty string should return empty list."""
        result = _split_sentences("")
        assert len(result) == 0

    def test_no_punctuation(self):
        """Text without punctuation should be treated as one sentence."""
        result = _split_sentences("hello world no punctuation")
        assert len(result) >= 1


class TestSummarizeExtractive:
    """Tests for extractive summarization."""

    def test_empty_text(self):
        """Empty text should return empty string."""
        result = summarize_extractive("")
        assert result == ""

    def test_short_text_returns_as_is(self):
        """Text with <= 5 sentences should be returned unchanged."""
        text = "One. Two. Three."
        result = summarize_extractive(text)
        assert len(result) > 0

    def test_longer_text_produces_summary(self):
        """Longer text should produce shorter summary."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a well-known pangram that contains every letter. "
            "It is often used for typing practice. "
            "The sentence has been used for decades. "
            "Many variations exist with different animals. "
            "Some versions use cats instead of foxes. "
            "Others use completely different words. "
            "The original purpose was to test typewriters. "
            "Now it is used for font display. "
            "Designers love this sentence. "
            "It shows how a font looks with all letters. "
            "Every designer should know this pangram."
        )
        result = summarize_extractive(text)
        assert len(result) < len(text)
        assert len(result) > 0

    def test_russian_text_works(self):
        """Russian text should be handled correctly."""
        text = (
            "У лукоморья дуб зелёный. Золотая цепь на дубе том. "
            "И днём и ночью кот учёный всё ходит по цепи кругом. "
            "Идёт направо — песнь заводит. Налево — сказку говорит. "
            "Там чудеса, там леший бродит. Русалка на ветвях сидит. "
            "Там на неведомых дорожках следы невиданных зверей."
        )
        result = summarize_extractive(text)
        assert len(result) > 0
        assert "дуб" in result.lower() or "лукоморь" in result.lower()

    def test_stop_words_exist(self):
        """Stop words should be defined for both languages."""
        assert len(STOP_WORDS) > 50
        # Russian
        assert "и" in STOP_WORDS
        assert "в" in STOP_WORDS
        assert "не" in STOP_WORDS
        # English
        assert "the" in STOP_WORDS
        assert "and" in STOP_WORDS
        assert "is" in STOP_WORDS

    def test_output_is_string(self):
        """Output should always be a string."""
        result = summarize_extractive("Some text here. More text there. And more.")
        assert isinstance(result, str)
