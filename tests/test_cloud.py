"""Tests for cloud module (non-network parts)."""

from unittest.mock import MagicMock, patch

from outloud.cloud import (
    _classify_error,
    check_keys,
)
from outloud.exceptions import (
    APIKeyError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
)


class TestErrorClassification:
    """Tests for API error classification."""

    def test_rate_limit_429_status(self):
        """429 status should be classified as rate limit."""
        from openai import APIStatusError
        mock_err = MagicMock(spec=APIStatusError)
        mock_err.status_code = 429

        result = _classify_error(mock_err)
        assert isinstance(result, RateLimitError)

    def test_unauthorized_401_status(self):
        """401 status should be classified as API key error."""
        from openai import APIStatusError
        mock_err = MagicMock(spec=APIStatusError)
        mock_err.status_code = 401

        result = _classify_error(mock_err)
        assert isinstance(result, APIKeyError)

    def test_server_error_500_status(self):
        """500 status should be classified as network error."""
        from openai import APIStatusError
        mock_err = MagicMock(spec=APIStatusError)
        mock_err.status_code = 500

        result = _classify_error(mock_err)
        assert isinstance(result, NetworkError)

    def test_timeout_error(self):
        """Timeout error should be classified as network error."""
        from openai import APITimeoutError
        mock_err = MagicMock(spec=APITimeoutError)

        result = _classify_error(mock_err)
        assert isinstance(result, NetworkError)

    def test_quota_exceeded_message(self):
        """Message with 'quota' should be classified as quota exceeded."""
        err = Exception("insufficient_quota: you have exceeded your quota")
        result = _classify_error(err)
        assert isinstance(result, QuotaExceededError)

    def test_rate_limit_message(self):
        """Message with 'rate_limit' should be classified as rate limit."""
        err = Exception("rate_limit_exceeded: too many requests")
        result = _classify_error(err)
        assert isinstance(result, RateLimitError)

    def test_timeout_message(self):
        """Message with 'timed out' should be classified as network error."""
        err = Exception("Request timed out after 120s")
        result = _classify_error(err)
        assert isinstance(result, NetworkError)

    def test_unknown_error_pass_through(self):
        """Unknown errors should be returned as-is."""
        err = ValueError("some random error")
        result = _classify_error(err)
        assert result is err


class TestKeyManagement:
    """Tests for API key management."""

    def test_check_keys_without_keys(self):
        """check_keys should return False when no keys configured."""
        with patch("outloud.cloud.KEYS_FILE") as mock_file:
            mock_file.exists.return_value = False
            assert check_keys() is False
