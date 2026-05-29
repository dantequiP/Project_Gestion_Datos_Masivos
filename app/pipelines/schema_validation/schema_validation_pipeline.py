"""
schema_validation_pipeline.py

Pipeline para validación estructural de SISMEPRE.
Esta etapa va después de Bronze y antes de Profiling.
"""

from pathlib import Path

from app.schema_validation.sismepre_schema_validator import run_sismepre_schema_validation
from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))


def run_schema_validation_pipeline(source: str = "sismepre") -> dict[str, str]:
    if source != "sismepre":
        raise ValueError("Por ahora solo está implementada la validación de esquema para source='sismepre'.")

    logger.info("Ejecutando pipeline de validación de esquema | source=%s", source)

    result = run_sismepre_schema_validation(
        bronze_root=Path("data/bronze"),
        reports_root=Path("reports/schema_validation/sismepre"),
        audit_root=Path("data/audit/schema_validation"),
    )

    logger.info("Pipeline de validación de esquema finalizado | outputs=%s", result)
    return result
