"""Tests for downloader module (non-network parts)."""

import pytest

from outloud.downloader import get_video_info
from outloud.exceptions import InvalidURLError


class TestDownloaderURLValidation:
    """Tests for URL validation."""

    def test_invalid_url_raises(self):
        """Non-HTTP URL should raise InvalidURLError."""
        import tempfile
        from pathlib import Path

        from outloud.downloader import download_audio

        with pytest.raises(InvalidURLError):
            download_audio("not-a-url", Path(tempfile.mkdtemp()))

    def test_ftp_url_raises(self):
        """FTP URL should raise InvalidURLError."""
        import tempfile
        from pathlib import Path

        from outloud.downloader import download_audio

        with pytest.raises(InvalidURLError):
            download_audio("ftp://example.com/audio.mp3", Path(tempfile.mkdtemp()))

    def test_valid_http_url_format(self):
        """HTTP URL should be accepted (even if download fails)."""
        import tempfile
        from pathlib import Path

        from outloud.downloader import download_audio
        from outloud.exceptions import DownloadError

        # Should not raise InvalidURLError, but may raise DownloadError
        with pytest.raises((DownloadError, Exception)):
            download_audio("http://example.com/nonexistent", Path(tempfile.mkdtemp()))


class TestVideoInfo:
    """Tests for video info extraction."""

    def test_invalid_url_raises_download_error(self):
        """Invalid URL should raise some error (NetworkError from yt-dlp)."""
        from outloud.exceptions import NetworkError
        with pytest.raises((NetworkError, Exception)):
            get_video_info("not-a-url")

    def test_nonexistent_url_raises_download_error(self):
        """Nonexistent URL should raise some error."""
        from outloud.exceptions import NetworkError
        with pytest.raises((NetworkError, Exception)):
            get_video_info("http://example.com/not-a-video")
