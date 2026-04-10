"""Tests for config module."""


from outloud.config import (
    API_TIMEOUT,
    CLOUD_GRAMMAR_MODELS,
    CLOUD_SUMMARY_MODELS,
    LANG_TO_LLM,
    LANG_TO_VOSK,
    LOCAL_LLM_MODELS,
    VOSK_MODELS,
    detect_hardware,
    get_models_dir,
    get_output_dir,
    model_exists,
)


class TestModelRegistry:
    """Tests for model registry configuration."""

    def test_vosk_models_have_required_fields(self):
        """All Vosk models must have name, lang, size, url."""
        for key, info in VOSK_MODELS.items():
            assert "name" in info, f"{key} missing 'name'"
            assert "lang" in info, f"{key} missing 'lang'"
            assert "size" in info, f"{key} missing 'size'"
            assert "url" in info, f"{key} missing 'url'"
            assert info["lang"] in ("ru", "en"), f"{key} unknown lang: {info['lang']}"

    def test_local_llm_models_have_required_fields(self):
        """All LLM models must have mlx_name, size, langs, task."""
        for key, info in LOCAL_LLM_MODELS.items():
            assert "mlx_name" in info, f"{key} missing 'mlx_name'"
            assert "size" in info, f"{key} missing 'size'"
            assert "langs" in info, f"{key} missing 'langs'"
            assert "task" in info, f"{key} missing 'task'"
            assert isinstance(info["langs"], list), f"{key} langs must be a list"

    def test_cloud_summary_models_not_empty(self):
        """Cloud summary models must be defined."""
        assert len(CLOUD_SUMMARY_MODELS) >= 3

    def test_cloud_grammar_models_not_empty(self):
        """Cloud grammar models must be defined."""
        assert len(CLOUD_GRAMMAR_MODELS) >= 1

    def test_model_count(self):
        """Should have expected number of models."""
        assert len(VOSK_MODELS) == 4  # 2 ru + 2 en
        assert len(LOCAL_LLM_MODELS) == 4  # qwen3-0.6b, gemma3-1b, qwen3-1.8b-reasoning, lmf2.5-350m


class TestLanguageMapping:
    """Tests for language-to-model mappings."""

    def test_lang_to_vosk_has_ru_and_en(self):
        """Language mapping should include ru and en."""
        assert "ru" in LANG_TO_VOSK
        assert "en" in LANG_TO_VOSK

    def test_lang_to_vosk_refs_valid_models(self):
        """Vosk language references must exist in VOSK_MODELS."""
        for lang, model_key in LANG_TO_VOSK.items():
            assert model_key in VOSK_MODELS, f"LANG_TO_VOSK[{lang}] -> {model_key} not found"

    def test_lang_to_llm_has_ru_and_en(self):
        """LLM language mapping should include ru and en."""
        assert "ru" in LANG_TO_LLM
        assert "en" in LANG_TO_LLM

    def test_lang_to_llm_refs_valid_models(self):
        """LLM language references must exist in LOCAL_LLM_MODELS."""
        for lang, model_keys in LANG_TO_LLM.items():
            for key in model_keys:
                assert key in LOCAL_LLM_MODELS, f"LANG_TO_LLM[{lang}] -> {key} not found"

    def test_llm_model_supports_its_language(self):
        """Model must support the language it's mapped to."""
        for lang, model_keys in LANG_TO_LLM.items():
            for key in model_keys:
                info = LOCAL_LLM_MODELS[key]
                assert lang in info["langs"], f"{key} doesn't support {lang}"


class TestTimeouts:
    """Tests for timeout configuration."""

    def test_api_timeout_positive(self):
        """API timeout must be a positive number."""
        assert API_TIMEOUT > 0

    def test_download_timeout_positive(self):
        """Download timeout must be a positive number."""
        from outloud.config import DOWNLOAD_TIMEOUT
        assert DOWNLOAD_TIMEOUT > 0

    def test_ytdlp_timeout_positive(self):
        """YTDLP timeout must be a positive number."""
        from outloud.config import YTDLP_TIMEOUT
        assert YTDLP_TIMEOUT > 0

    def test_rate_limit_retry_delay_positive(self):
        """Rate limit retry delay must be positive."""
        from outloud.config import RATE_LIMIT_RETRY_DELAY
        assert RATE_LIMIT_RETRY_DELAY > 0


class TestHelpers:
    """Tests for helper functions."""

    def test_get_models_dir_returns_path(self):
        """get_models_dir must return a Path."""
        from pathlib import Path
        result = get_models_dir()
        assert isinstance(result, Path)
        assert "models" in str(result)

    def test_get_output_dir_returns_path(self):
        """get_output_dir must return a Path."""
        from pathlib import Path
        result = get_output_dir()
        assert isinstance(result, Path)

    def test_detect_hardware_returns_dict(self):
        """detect_hardware must return dict with expected keys."""
        result = detect_hardware()
        assert isinstance(result, dict)
        assert "chip" in result
        assert "ram_gb" in result
        assert "ai_model" in result

    def test_detect_hardware_ai_model_valid(self):
        """Detected AI model must be in LOCAL_LLM_MODELS."""
        result = detect_hardware()
        assert result["ai_model"] in LOCAL_LLM_MODELS

    def test_model_exists_unknown_vosk(self):
        """model_exists should return False for unknown Vosk model."""
        assert model_exists("nonexistent-model", "vosk") is False

    def test_model_exists_unknown_llm(self):
        """model_exists should return False for unknown LLM model."""
        assert model_exists("nonexistent-model", "llm") is False
