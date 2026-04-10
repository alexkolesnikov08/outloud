"""Логирование в файл outloud-logs/."""

import logging
from pathlib import Path
from datetime import datetime

from outloud.config import LOGS_DIR


def get_logger(name: str = "outloud") -> logging.Logger:
    """Получить логгер с записью в файл."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOGS_DIR / f"outloud_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Только file handler, в консоль не пишем
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
