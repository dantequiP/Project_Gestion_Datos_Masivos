"""
logger.py — Módulo de logging centralizado para el pipeline RAW.

Configura un logger con salida simultánea a consola y archivo rotativo.
Todos los módulos del pipeline importan desde aquí para mantener consistencia.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB por archivo
    backup_count: int = 5,
) -> logging.Logger:
    """
    Crea y retorna un logger configurado con handlers de consola y archivo.

    Parameters
    ----------
    name : str
        Nombre del logger (generalmente __name__ del módulo invocador).
    log_dir : Optional[Path]
        Directorio donde se escriben los archivos de log.
        Si es None, solo se usa el handler de consola.
    level : str
        Nivel de logging: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    max_bytes : int
        Tamaño máximo de cada archivo de log antes de rotar.
    backup_count : int
        Número de archivos de respaldo a conservar.

    Returns
    -------
    logging.Logger
        Logger configurado y listo para usar.
    """
    logger = logging.getLogger(name)

    # Evita duplicar handlers si el logger ya fue configurado
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Handler de consola ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # --- Handler de archivo (rotativo) ---
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
