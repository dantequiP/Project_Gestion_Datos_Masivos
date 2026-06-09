"""
siaf_schema_validation_pipeline.py

Pipeline de validación de schema para SIAF.
"""

from datetime import datetime
from pathlib import Path

from app.schema_validation.siaf_schema_validator import run_siaf_schema_validation
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


def run_siaf_schema_validation_pipeline() -> dict[str, str]:
    """
    Pipeline de validación estructural para SIAF Ingresos.

    Lee schema_rules_siaf.yaml y compara contra los Parquet Bronze disponibles.
    Genera reporte HTML, CSVs y auditoría JSON.
    No modifica datos Bronze.
    """
    logger.info("=== Inicio pipeline schema validation SIAF ===")
    started_at = datetime.now()

    outputs = run_siaf_schema_validation(
        bronze_root=Path("data/bronze"),
        reports_root=Path("reports/schema_validation/siaf"),
        audit_root=Path("data/audit/schema_validation"),
        config_path=Path("app/config/schema_rules_siaf.yaml"),
    )

    duration = (datetime.now() - started_at).total_seconds()
    logger.info("=== Pipeline schema validation SIAF completado en %.1fs ===", duration)
    return outputs


if __name__ == "__main__":
    run_siaf_schema_validation_pipeline()