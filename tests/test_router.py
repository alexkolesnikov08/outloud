"""Tests for ProviderRouter."""


from outloud.config import LANG_TO_LLM, LANG_TO_VOSK
from outloud.router import ProviderRouter


class TestProviderRouterInit:
    """Tests for router initialization."""

    def test_default_language_is_russian(self):
        """Default language should be Russian."""
        router = ProviderRouter()
        assert router.language == "ru"

    def test_custom_language(self):
        """Custom language should be set."""
        router = ProviderRouter(language="en")
        assert router.language == "en"

    def test_cloud_mode_flag(self):
        """Cloud mode should be configurable."""
        router_local = ProviderRouter(cloud=False)
        router_cloud = ProviderRouter(cloud=True)
        assert router_local.cloud is False
        assert router_cloud.cloud is True


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_russian_text(self):
        """Russian text should be detected."""
        text = "Привет мир как дела сегодня"
        assert ProviderRouter.detect_language(text) == "ru"

    def test_detect_english_text(self):
        """English text should be detected."""
        text = "Hello world how are you today"
        assert ProviderRouter.detect_language(text) == "en"

    def test_detect_empty_text(self):
        """Empty text should default to Russian."""
        assert ProviderRouter.detect_language("") == "ru"

    def test_detect_mixed_cyrillic(self):
        """Text with mostly Cyrillic should be Russian."""
        text = "Hello Привет мир world"
        assert ProviderRouter.detect_language(text) == "ru"

    def test_detect_mixed_latin(self):
        """Text with mostly Latin should be English."""
        text = "Привет Hello world today"
        assert ProviderRouter.detect_language(text) == "en"


class TestASRModelSelection:
    """Tests for ASR model selection."""

    def test_russian_gets_russian_model(self):
        """Russian language should get Russian Vosk model."""
        router = ProviderRouter(language="ru")
        model = router.get_asr_model()
        assert model in LANG_TO_VOSK.values()
        assert "ru" in model

    def test_english_gets_english_model(self):
        """English language should get English Vosk model."""
        router = ProviderRouter(language="en")
        model = router.get_asr_model()
        assert model in LANG_TO_VOSK.values()
        assert "en" in model


class TestLLMModelSelection:
    """Tests for LLM model selection."""

    def test_russian_gets_russian_models(self):
        """Russian language should get models that support Russian."""
        ProviderRouter(language="ru")
        models = LANG_TO_LLM.get("ru", [])
        assert len(models) > 0

    def test_english_gets_english_models(self):
        """English language should get models that support English."""
        ProviderRouter(language="en")
        models = LANG_TO_LLM.get("en", [])
        assert len(models) > 0


class TestRouterCleanup:
    """Tests for router cleanup."""

    def test_cleanup_no_models(self):
        """Cleanup should work even without loaded models."""
        router = ProviderRouter()
        router.cleanup()  # Should not raise
