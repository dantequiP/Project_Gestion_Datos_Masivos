"""
bronze_pipeline.py — Pipeline Bronze (RAW) de la Arquitectura Medallion.

Este módulo es el punto de entrada de la capa Bronze. Carga la configuración,
inicializa el IngestionService y delega la ejecución.

Puede ejecutarse directamente o ser importado desde main.py.

Uso directo:
    python -m app.pipelines.bronze.bronze_pipeline --source all
    python -m app.pipelines.bronze.bronze_pipeline --source siaf
    python -m app.pipelines.bronze.bronze_pipeline --source sismepre
    python -m app.pipelines.bronze.bronze_pipeline --source renamu
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Asegura que el root del proyecto esté en el path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ingestion_service import IngestionService
from app.utils.logger import get_logger


def load_config(config_path: Path) -> dict:
    """
    Carga el archivo config.yaml y sobreescribe valores desde variables de entorno.

    Parameters
    ----------
    config_path : Path
        Ruta al archivo config.yaml.

    Returns
    -------
    dict
        Configuración lista para usar.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Sobreescribir con variables de entorno si están definidas
    env_overrides = {
        "http.timeout": ("HTTP_TIMEOUT", int),
        "http.max_retries": ("HTTP_MAX_RETRIES", int),
        "http.backoff_factor": ("HTTP_BACKOFF_FACTOR", int),
        "ckan.page_limit": ("CKAN_PAGE_LIMIT", int),
        "ckan.base_url": ("MEF_API_BASE", str),
    }

    for config_key, (env_var, cast_fn) in env_overrides.items():
        env_val = os.getenv(env_var)
        if env_val is not None:
            keys = config_key.split(".")
            node = config
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = cast_fn(env_val)

    return config


def run_bronze_pipeline(source: str = "all") -> None:
    """
    Ejecuta el pipeline Bronze para la fuente especificada.

    Parameters
    ----------
    source : str
        'all' | 'siaf' | 'sismepre' | 'renamu'
    """
    # ---- Carga de entorno y configuración ----
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    config_path = PROJECT_ROOT / "app" / "config" / "config.yaml"
    config = load_config(config_path)

    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_dir = Path(os.getenv("LOG_DIR", config["paths"]["logs_root"]))

    logger = get_logger("bronze_pipeline", log_dir=log_dir, level=log_level)
    logger.info("Configuración cargada desde: %s", config_path)
    logger.info("Fuente a procesar: %s", source)

    audit_dir = Path(config["paths"]["audit_root"])
    service = IngestionService(config=config, audit_dir=audit_dir)

    # ---- Despacho por fuente ----
    source = source.lower().strip()

    if source == "all":
        service.run_all()
    elif source == "siaf":
        service.ingest_siaf_csv()
        service.ingest_siaf_api()
    elif source == "siaf_csv":
        service.ingest_siaf_csv()
    elif source == "siaf_api":
        service.ingest_siaf_api()
    elif source == "sismepre":
        service.ingest_sismepre()
    elif source == "renamu":
        service.ingest_renamu()
    else:
        logger.error(
            "Fuente desconocida: '%s'. Opciones válidas: all, siaf, sismepre, renamu",
            source,
        )
        sys.exit(1)

    logger.info("Pipeline Bronze finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline RAW — Capa Bronze de la Arquitectura Medallion"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "siaf", "siaf_csv", "siaf_api", "sismepre", "renamu"],
        help="Fuente a procesar (default: all)",
    )
    args = parser.parse_args()
    run_bronze_pipeline(source=args.source)