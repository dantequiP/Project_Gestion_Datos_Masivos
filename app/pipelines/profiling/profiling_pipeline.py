
from pathlib import Path

from app.profiling.dataset_profiler import run_sismepre_profiling
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


def run_profiling_pipeline(source: str = "sismepre") -> dict[str, str]:
    if source != "sismepre":
        raise ValueError("Por ahora solo está implementado profiling para source='sismepre'.")

    logger.info("Ejecutando pipeline de profiling | source=%s", source)

    result = run_sismepre_profiling(
        bronze_root=Path("data/bronze"),
        reports_root=Path("reports/profiling/sismepre"),
        audit_root=Path("data/audit/profiling"),
    )

    logger.info("Pipeline de profiling finalizado | outputs=%s", result)
    return result
