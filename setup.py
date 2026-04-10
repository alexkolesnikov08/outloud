from setuptools import setup, find_packages

setup(
    name="outloud",
    version="0.2.0",
    description="Audio transcription and summarization",
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
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "sentencepiece>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "outloud=outloud.cli:main",
        ],
    },
)
