"""Tests for exception classes."""


from outloud.exceptions import (
    APIKeyError,
    AudioError,
    ConfigError,
    ConversionError,
    DownloadError,
    InvalidURLError,
    ModelNotFoundError,
    NetworkError,
    OutLoudError,
    QuotaExceededError,
    RateLimitError,
    RecordingError,
)


class TestModelNotFoundError:
    def test_message_contains_model_key(self):
        exc = ModelNotFoundError("vosk-small-ru", "vosk")
        assert "vosk-small-ru" in str(exc)
        assert "vosk" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(ModelNotFoundError("test"), OutLoudError)


class TestDownloadError:
    def test_message_contains_url(self):
        exc = DownloadError("http://example.com", "timeout")
        assert "http://example.com" in str(exc)
        assert "timeout" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(DownloadError("http://x.com"), OutLoudError)


class TestNetworkError:
    def test_message_contains_service(self):
        exc = NetworkError("Groq", "timed out")
        assert "Groq" in str(exc)
        assert "timed out" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(NetworkError("Test"), OutLoudError)


class TestRateLimitError:
    def test_message_without_retry(self):
        exc = RateLimitError("Groq")
        assert "Groq" in str(exc)
        assert "Rate limit" in str(exc)

    def test_message_with_retry(self):
        exc = RateLimitError("Groq", 10)
        assert "10s" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(RateLimitError("Test"), OutLoudError)


class TestQuotaExceededError:
    def test_message_contains_service(self):
        exc = QuotaExceededError("Groq")
        assert "Groq" in str(exc)
        assert "quota" in str(exc).lower()

    def test_inherits_from_outloud_error(self):
        assert isinstance(QuotaExceededError("Test"), OutLoudError)


class TestAudioError:
    def test_message_contains_reason(self):
        exc = AudioError("no audio data")
        assert "no audio data" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(AudioError("test"), OutLoudError)


class TestConversionError:
    def test_message_contains_details(self):
        exc = ConversionError("audio.m4a", "wav", "ffmpeg not found")
        assert "audio.m4a" in str(exc)
        assert "wav" in str(exc)
        assert "ffmpeg not found" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(ConversionError("a", "b", "c"), OutLoudError)


class TestRecordingError:
    def test_message_contains_reason(self):
        exc = RecordingError("no input device")
        assert "no input device" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(RecordingError("test"), OutLoudError)


class TestInvalidURLError:
    def test_message_contains_url(self):
        exc = InvalidURLError("not-a-url")
        assert "not-a-url" in str(exc)
        assert "yt-dlp" in str(exc).lower()

    def test_inherits_from_outloud_error(self):
        assert isinstance(InvalidURLError("x"), OutLoudError)


class TestConfigError:
    def test_message_contains_key(self):
        exc = ConfigError("api_key")
        assert "api_key" in str(exc)

    def test_message_with_reason(self):
        exc = ConfigError("api_key", "missing")
        assert "missing" in str(exc)

    def test_inherits_from_outloud_error(self):
        assert isinstance(ConfigError("test"), OutLoudError)


class TestAPIKeyError:
    def test_message_mentions_cloud_setup(self):
        exc = APIKeyError("Groq")
        assert "cloud-setup" in str(exc)

    def test_inherits_from_config_error(self):
        assert isinstance(APIKeyError("Test"), ConfigError)
        assert isinstance(APIKeyError("Test"), OutLoudError)
