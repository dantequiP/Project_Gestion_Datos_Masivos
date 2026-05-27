"""
main.py — Punto de entrada principal del proyecto de Ingeniería de Datos.

Arquitectura: Medallion (Bronze → Silver → Gold)
Capa actual: Bronze / RAW Ingestion

Uso:
    python main.py                         # Ingesta completa (todas las fuentes)
    python main.py --source siaf           # Solo SIAF
    python main.py --source sismepre       # Solo SISMEPRE
    python main.py --source renamu         # Solo RENAMU
    python main.py --verify                # Verifica los Parquet generados
    python main.py --source siaf --verify  # Ingesta SIAF + verificación
"""

import argparse
import sys
from pathlib import Path

# ---- Punto de entrada ----
from app.pipelines.bronze.bronze_pipeline import run_bronze_pipeline
from app.utils.logger import get_logger

logger = get_logger("main", log_dir=Path("logs"))


def verify_parquets(bronze_root: Path = Path("data/bronze")) -> None:
    """
    Verifica los archivos Parquet generados imprimiendo un resumen
    de registros, columnas y tamaño en disco.

    Parameters
    ----------
    bronze_root : Path
        Raíz de la capa bronze.
    """
    import pyarrow.parquet as pq

    parquet_files = sorted(bronze_root.rglob("*.parquet"))

    if not parquet_files:
        logger.warning("No se encontraron archivos Parquet en: %s", bronze_root)
        return

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  VERIFICACIÓN DE PARQUETS RAW")
    logger.info("  Directorio: %s", bronze_root)
    logger.info("  Total archivos: %d", len(parquet_files))
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    total_rows = 0
    total_size_bytes = 0

    for pq_file in parquet_files:
        try:
            table = pq.read_table(pq_file)
            size_kb = pq_file.stat().st_size / 1024
            total_rows += table.num_rows
            total_size_bytes += pq_file.stat().st_size

            logger.info(
                "  ✓ %s | rows=%d | cols=%d | size=%.1f KB",
                pq_file.relative_to(bronze_root),
                table.num_rows,
                table.num_columns,
                size_kb,
            )
        except Exception as exc:
            logger.error("  ✗ %s — Error: %s", pq_file, exc)

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(
        "  TOTAL | archivos=%d | rows=%d | size=%.2f MB",
        len(parquet_files),
        total_rows,
        total_size_bytes / (1024 * 1024),
    )
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline RAW — Arquitectura Medallion | Capa Bronze\n"
            "Descarga y almacena datos exactamente como vienen de las fuentes oficiales."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "siaf", "siaf_csv", "siaf_api", "sismepre", "renamu"],
        help=(
            "Fuente a ingestar. Default: all\n"
            "  siaf_csv  → solo CSV históricos 2012-2021\n"
            "  siaf_api  → solo API CKAN 2022-2026\n"
            "  sismepre  → solo SISMEPRE predial"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verifica los Parquet RAW generados después de la ingestión.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo verifica los Parquet existentes, sin ejecutar ingestión.",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_parquets()
        return

    # Ejecutar pipeline
    run_bronze_pipeline(source=args.source)

    # Verificar si se solicitó
    if args.verify:
        verify_parquets()


if __name__ == "__main__":
    main()