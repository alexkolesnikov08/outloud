from setuptools import setup, find_packages

setup(
    name="outloud",
    version="0.3.0",
    description="Record voice -> get text + summary",
    author="OutLoud Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "vosk>=0.3.44",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "scipy>=1.11.0",
        "pydub>=0.25.0",
        "mlx-lm>=0.20.0",
        "openai>=1.0.0",
        "yt-dlp>=2024.1.0",
    ],
    entry_points={
        "console_scripts": [
            "outloud=outloud.cli:main",
        ],
    },
)
