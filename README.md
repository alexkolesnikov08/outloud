# OutLoud

Запишите голос → получите текст и краткое содержание.

## Установка

```bash
pip install -e .
brew install ffmpeg
outloud setup          # один раз — скачает модели
```

## Команды

| Команда | Что делает |
|---------|-----------|
| `outloud record` | Записать голос → текст + краткое |
| `outloud file audio.m4a` | То же, из файла |
| `outloud yt "ссылка"` | То же, с YouTube |
| `outloud cloud-setup` | Подключить облачные модели |

## Облако (Whisper + GPT-OSS + Llama)

Один раз:
```bash
outloud cloud-setup
```

Потом:
```bash
outloud record --cloud
outloud record --cloud --grammar    # с исправлением ошибок
```

## Что получится

```
outloud_20260410_150000/
├── audio.wav          # запись
├── transcription.txt  # текст
├── summary.txt        # краткое
└── corrected.txt      # с исправлениями (--grammar)
```

## Модели

| | Локально | Облако |
|--|----------|--------|
| Расшифровка | Vosk small | Whisper Large v3 (H100) |
| Суммаризация | Qwen 0.8B 4-bit | GPT-OSS 20B |
| Грамматика | Qwen 0.8B 4-bit | Llama 3.1 8B |

Ключ Groq — https://console.groq.com/keys (бесплатно, 10K запросов/день)
