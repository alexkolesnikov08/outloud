# OutLoud

Record your voice, get text and a summary. Works locally or in the cloud. No GUI, no fluff — just a terminal and a few commands.

I'm 17, this is my first real project, and it was built basically on my knee. The code isn't perfect, but it works. I'm learning as I go and improving it with every commit.

---

## What it does

You speak → OutLoud gives you back:
1. **Full transcription** — every word you said
2. **Study notes** — a structured markdown summary (main idea, key points, takeaway)
3. **Grammar correction** — cleans up transcription errors (cloud mode)

It works in **two modes**:

| Mode | Speech | Summary | Speed | Cost |
|------|--------|---------|-------|------|
| **Local** | Vosk (70MB model) | Extractive summary + LLM formatting | ~10s for 2min audio | Free |
| **Cloud** | Whisper Large v3 Turbo | GPT-OSS 20B (fallback: Qwen 32B → Llama 70B → Llama 8B) | ~4s for 2min audio | Free (Groq API) |

Local mode runs entirely on your machine — no data leaves your computer. Cloud mode uses Groq's free API (H100 GPUs) and gives much better quality.

---

## Installation

### Requirements
- Python 3.11+
- macOS (Apple Silicon) or Linux
- `ffmpeg` (for audio conversion)

### Install

```bash
# Clone the repo
git clone https://github.com/alexkolesnikov08/Outloud.git
cd Outloud

# Install
pip install -e .

# Install ffmpeg
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
```

### Setup

```bash
# Download default models (runs once)
outloud setup

# Check what's available
outloud models
```

This downloads:
- **Vosk small-ru** (70MB) — Russian speech recognition
- **Qwen 3 0.6B 4-bit** (400MB) — local summarization

---

## Quick Start

```bash
# Record from microphone
outloud record

# Record with cloud models (better quality)
outloud record --cloud

# Add grammar correction
outloud record --cloud --grammar

# Process an existing file
outloud file lecture.m4a

# Process any URL (YouTube, Vimeo, etc.)
outloud url "https://youtube.com/watch?v=..."

# Specify language
outloud record --lang en
```

Output goes to `~/Desktop/outloud_TIMESTAMP/` as `.md` files.

---

## Commands

| Command | Description |
|---------|-------------|
| `outloud setup` | Download default models |
| `outloud setup --model <key>` | Download a specific model |
| `outloud setup --all` | Download everything |
| `outloud models` | List all available models and status |
| `outloud record` | Record from mic → text + summary |
| `outloud file <path>` | Process an audio file |
| `outloud url <url>` | Process audio from any URL |
| `outloud cloud-setup` | Configure Groq API key |
| `outloud cloud-status` | Check API key status |

### Options

| Flag | What it does |
|------|-------------|
| `--cloud` | Use cloud models (Whisper + GPT-OSS) |
| `--grammar` | Fix grammar in transcription |
| `--lang ru\|en` | Set language (auto-detect if omitted) |
| `--model <path>` | Use a custom GGUF/MLX model |

---

## Models

### Speech Recognition (Vosk)

| Key | Language | Size | Quality |
|-----|----------|------|---------|
| `vosk-small-ru` | Russian | 70MB | Good for clear speech |
| `vosk-medium-ru` | Russian | 800MB | Better accuracy |
| `vosk-small-en` | English | 50MB | Good for clear speech |
| `vosk-medium-en` | English | 1.6GB | Better accuracy |

### Local LLMs (MLX 4-bit, Apple Silicon)

| Key | Size | Languages | Notes |
|-----|------|-----------|-------|
| `qwen3-0.6b` | 400MB | RU, EN | Default — fast, decent |
| `gemma3-1b` | 800MB | RU, EN | Better quality, slower |
| `qwen3-1.8b-reasoning` | 1.2GB | RU, EN | Best local quality |
| `lmf2.5-350m` | 250MB | EN only | English only, very small |

### Custom Models

You can use your own GGUF/MLX model by passing the path:

```bash
outloud record --model /path/to/my/model.gguf
```

### Cloud Models (Groq)

| Task | Model | Notes |
|------|-------|-------|
| Speech | Whisper Large v3 Turbo | ~98% accuracy |
| Summary | GPT-OSS 20B | Falls back to Qwen 32B → Llama 70B → Llama 8B |
| Grammar | Llama 3.1 8B | Falls back to Llama 4 Scout |

All cloud models are free via Groq's API ( generous limits).

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Microphone  │────▶│  Transcribe   │────▶│   Summarize    │
│  / File / URL│     │  (Vosk /      │     │  (LLM /        │
│              │     │   Whisper)    │     │   Extractive)  │
└─────────────┘     └──────────────┘     └───────────────┘
                                                  │
                                           ┌──────▼───────┐
                                           │  Output (.md) │
                                           │  - transcription│
                                           │  - summary    │
                                           │  - corrected  │
                                           └──────────────┘
```

### The Pipeline

1. **Record / Load** — captures audio from mic, file, or downloads from URL (yt-dlp supports hundreds of sites)
2. **Convert** — ffmpeg converts to mono WAV 16kHz (Vosk requirement)
3. **Transcribe** — local Vosk or cloud Whisper turns audio into text
4. **Summarize** — ProviderRouter picks the best model:
   - Cloud: GPT-OSS 20B with fallback chain
   - Local: extractive summary (no ML) or LLM formatting
5. **Grammar** (optional) — rule-based for small models, LLM for cloud
6. **Save** — everything goes to `~/Desktop/outloud_TIMESTAMP/` as `.md` files

### Language Detection

If you don't specify `--lang`, OutLoud auto-detects the language from the first few words of transcription and picks the right models automatically.

### Large Files

Files over 20MB are automatically split into 5-minute chunks (exported as 64kbps MP3 to stay under Groq's 25MB limit). Each chunk is transcribed separately and merged back together.

### Fallback

When cloud models hit rate limits, OutLoud silently falls back to local models with a warning message. You never lose your work.

---

## Configuration

### Cloud API Key

```bash
outloud cloud-setup
```

You'll need a free Groq key from [console.groq.com/keys](https://console.groq.com/keys). The key is saved to `~/.outloud/api_keys.json` with `chmod 600` — it never leaves your machine.

### Output Directory

By default, results go to `~/Desktop/outloud_TIMESTAMP/`. Change it in `outloud/config.py`:

```python
OUTPUT_DIR = Path.home() / "Documents" / "outloud"
```

### Adding Models

```bash
# Download a specific model
outloud setup --model vosk-medium-ru
outloud setup --model gemma3-1b

# Download everything
outloud setup --all

# Check status
outloud models
```

---

## Output Format

Every session creates a folder on your Desktop:

```
outloud_20250410_142510/
├── audio.wav          # Original recording
├── transcription.md   # Full text
├── summary.md         # Study notes
└── corrected.md       # Grammar-fixed version (--grammar flag)
```

### Summary Format

```markdown
## 📝 Конспект

**Текст:** 11 предложений, 164 слов

> **Главная мысль:** У лукоморья дуб зелёный...

**Ключевые моменты:**
- Кот учёный ходит по цепи кругом
- Там чудеса: леший, русалка, избушка
- Тридцать витязей и колдун несут богатыря

> **Итог:** И я там был, и мёд я пил...
```

---

## Troubleshooting

### `ffmpeg not found`
```bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg
```

### `No audio input device`
Make sure your microphone is connected and not being used by another app (Zoom, Discord, etc.)

### `Model not found`
```bash
outloud setup --model vosk-small-ru
outloud models  # verify it's there
```

### Cloud mode gives "quota exceeded"
Groq has rate limits. Wait a few minutes or use local mode (no `--cloud` flag). You can also check limits at [console.groq.com/settings/project/limits](https://console.groq.com/settings/project/limits).

### Transcription quality is bad
Local Vosk is ~60-80% accurate. Switch to cloud mode (`--cloud`) for ~95%+ accuracy. For better local quality, download the medium model:
```bash
outloud setup --model vosk-medium-ru
```

### "Module not found" after cloning
```bash
pip install -e .
```

### Memory issues on 4GB RAM
Stick to the smallest models. OutLoud auto-detects your RAM and recommends `lmf2.5-350m` (250MB) for 4GB machines. Avoid `--all` — it downloads everything.

---

## Project Structure

```
Outloud/
├── outloud/
│   ├── __init__.py        # Version
│   ├── __main__.py        # Entry point
│   ├── cli.py             # CLI commands (click)
│   ├── config.py          # Model registry, settings
│   ├── router.py          # ProviderRouter — model selection + fallback
│   ├── llm_pipeline.py    # MLX inference for any model
│   ├── transcriber.py     # Vosk speech-to-text
│   ├── cloud.py           # Groq API (Whisper, GPT-OSS, Llama)
│   ├── summarizer.py      # Extractive summarization
│   ├── downloader.py      # Audio from any URL (yt-dlp)
│   ├── recorder.py        # Microphone recording + VU meter
│   ├── utils.py           # Audio conversion
│   ├── exceptions.py      # Custom error classes
│   └── logger.py          # Logging with rotation
├── tests/                 # 93 unit tests
├── .github/workflows/     # CI/CD
├── pyproject.toml         # Build + lint + test config
└── README.md
```

---

## Contributing

This is my first open-source project, so I'm still figuring things out. But if you want to help:

1. Fork the repo
2. Create a branch (`git checkout -b feature/something-cool`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Run linter (`ruff check outloud/`)
6. Push and open a PR

No contribution is too small — typo fixes, better prompts, new model support, anything. I'll review it.

---

## License

MIT. Do whatever you want with it.

---

> Built by a 17-year-old who just wanted to turn voice recordings into notes. No funding, no team, no experience — just curiosity and a Mac with 4GB of RAM.
