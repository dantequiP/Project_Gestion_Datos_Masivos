"""
siaf_quality_pipeline.py

Orquestador del pipeline de calidad de datos para SIAF Ingresos.

Flujo completo:
    1. Carga quality_rules_siaf.yaml.
    2. Descubre Parquets Bronze (sin _diario — regla crítica SIAF).
    3. Carga DataFrame PySpark con mergeSchema=True.
    4. Ejecuta SiafQualityChecker → 60+ reglas en 8 dimensiones.
    5. Genera CSV, Parquet, HTML (reporte + dashboard) y auditoría JSON.
    6. NO modifica datos Bronze.

Archivos generados:
    reports/quality/siaf/siaf_quality_detail.csv
    reports/quality/siaf/siaf_quality_summary.csv
    reports/quality/siaf/siaf_quality_failed_samples.csv
    reports/quality/siaf/siaf_quality_dashboard.html
    reports/quality/siaf/siaf_ingresos_quality.html
    data/quality/siaf/siaf_quality_detail.parquet
    data/quality/siaf/siaf_quality_summary.parquet
    data/audit/quality/year=YYYY/month=MM/day=DD/siaf_quality_*.json
"""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pyspark.sql import SparkSession

from app.quality.siaf_quality_checker import SiafQualityChecker
from app.quality.siaf_quality_report import (
    build_summary,
    write_siaf_html,
    write_siaf_dashboard_html,
)
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))

# ── Rutas por defecto ─────────────────────────────────────────────────────────
BRONZE_ROOT = Path("data/bronze/siaf")
REPORTS_ROOT = Path("reports/quality/siaf")
DATA_QUALITY_ROOT = Path("data/quality/siaf")
AUDIT_ROOT = Path("data/audit/quality")
CONFIG_PATH = Path("app/config/quality_rules_siaf.yaml")


# ── Carga de configuración ────────────────────────────────────────────────────

def load_quality_rules(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Carga y valida el YAML de reglas de calidad SIAF."""
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de reglas: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("El YAML de reglas SIAF no tiene una estructura válida.")
    for section in ["settings", "completitud", "validez", "exactitud"]:
        if section not in config:
            raise ValueError(f"Falta sección obligatoria en quality_rules_siaf.yaml: '{section}'")
    logger.info("Reglas de calidad SIAF cargadas desde: %s", config_path)
    return config


# ── Descubrimiento de Parquets ────────────────────────────────────────────────

def discover_parquet_paths(bronze_root: Path) -> list[str]:
    """
    Descubre Parquets Bronze de SIAF excluyendo _diario.

    REGLA CRÍTICA heredada del schema validation:
    Los archivos _diario son acumulativos y duplicarían montos.
    Solo se procesan archivos mensual/anuales.
    """
    patron = str(bronze_root / "**" / "*.parquet")
    todas = glob.glob(patron, recursive=True)
    filtradas = sorted([r for r in todas if "_diario" not in r.lower()])
    excluidos = len(todas) - len(filtradas)

    logger.info(
        "Parquets Bronze SIAF | total=%d | excluidos(_diario)=%d | a_procesar=%d",
        len(todas), excluidos, len(filtradas),
    )
    for p in filtradas:
        logger.info("  → %s", p)

    if not filtradas:
        raise FileNotFoundError(
            f"No se encontraron Parquets Bronze en {bronze_root} (sin _diario). "
            "Verifica que el pipeline Bronze haya corrido primero."
        )
    return filtradas


# ── Auditoría JSON ────────────────────────────────────────────────────────────

def write_quality_audit(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    failed_samples_df: pd.DataFrame,
    audit_root: Path,
    reports_root: Path,
    data_quality_root: Path,
    total_rows: int,
    n_parquets: int,
    started_at: datetime,
) -> Path:
    """Auditoría JSON compatible con el formato del SISMEPRE."""
    finished_at = datetime.now()
    n_rules = int(len(detail_df))
    n_obs_rules = int((detail_df["failed_rows"] > 0).sum())
    n_observations = int(detail_df["failed_rows"].sum())
    n_bloqueantes = int(
        detail_df[
            (detail_df["failed_rows"] > 0) &
            (detail_df["severity"].isin(["Bloqueante", "Alta"]))
        ].shape[0]
    )
    status = "success" if n_bloqueantes == 0 else "warning"

    record = {
        "pipeline_name": "quality_siaf_ingresos",
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "total_rows_bronze": total_rows,
        "parquets_procesados": n_parquets,
        "rules_evaluated": n_rules,
        "rules_with_observations": n_obs_rules,
        "observations_total": n_observations,
        "rules_bloqueantes_o_altas_con_observacion": n_bloqueantes,
        "failed_samples_total": int(len(failed_samples_df)),
        "outputs": {
            "summary_csv": str(reports_root / "siaf_quality_summary.csv"),
            "detail_csv": str(reports_root / "siaf_quality_detail.csv"),
            "failed_samples_csv": str(reports_root / "siaf_quality_failed_samples.csv"),
            "dashboard_html": str(reports_root / "siaf_quality_dashboard.html"),
            "report_html": str(reports_root / "siaf_ingresos_quality.html"),
            "detail_parquet": str(data_quality_root / "siaf_quality_detail.parquet"),
            "summary_parquet": str(data_quality_root / "siaf_quality_summary.parquet"),
        },
    }

    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = (
        audit_dir / f"siaf_quality_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    )
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Auditoría JSON guardada: %s", audit_path)
    return audit_path


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_siaf_quality_pipeline(
    spark: SparkSession,
    bronze_root: Path = BRONZE_ROOT,
    reports_root: Path = REPORTS_ROOT,
    data_quality_root: Path = DATA_QUALITY_ROOT,
    audit_root: Path = AUDIT_ROOT,
    config_path: Path = CONFIG_PATH,
) -> dict[str, str]:
    """
    Orquesta la evaluación de calidad completa de SIAF Ingresos.

    Parameters
    ----------
    spark : SparkSession
        Sesión Spark activa (creada en run_siaf_quality.py).
    bronze_root, reports_root, data_quality_root, audit_root, config_path
        Rutas del proyecto. Los valores por defecto siguen la estructura estándar.

    Returns
    -------
    dict[str, str]
        Rutas absolutas de todos los archivos generados.
    """
    started_at = datetime.now()
    logger.info("=== Inicio pipeline Quality SIAF Ingresos ===")

    reports_root.mkdir(parents=True, exist_ok=True)
    data_quality_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    # 1. Reglas
    config = load_quality_rules(config_path)
    settings = config["settings"]

    # 2. Parquets
    parquet_paths = discover_parquet_paths(bronze_root)

    # 3. Carga Spark
    logger.info("Cargando %d Parquets en Spark...", len(parquet_paths))
    df = spark.read.option("mergeSchema", "true").parquet(*parquet_paths)
    total_rows = df.count()
    logger.info("DataFrame cargado: %d filas × %d columnas", total_rows, len(df.columns))

    # 4. Evaluación de calidad
    checker = SiafQualityChecker(spark=spark, df=df, config=config)
    detail_df, failed_samples_df = checker.run()
    summary_df = build_summary(detail_df)

    # ── Salidas ───────────────────────────────────────────────────────────────
    detail_csv = reports_root / "siaf_quality_detail.csv"
    summary_csv = reports_root / "siaf_quality_summary.csv"
    failed_samples_csv = reports_root / "siaf_quality_failed_samples.csv"
    dashboard_html = reports_root / "siaf_quality_dashboard.html"
    report_html = reports_root / "siaf_ingresos_quality.html"
    detail_parquet = data_quality_root / "siaf_quality_detail.parquet"
    summary_parquet = data_quality_root / "siaf_quality_summary.parquet"

    # CSV (utf-8-sig para apertura directa en Excel)
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    failed_samples_df.to_csv(failed_samples_csv, index=False, encoding="utf-8-sig")

    # Parquet (para consumo por Silver/Gold o analítica posterior)
    detail_df.to_parquet(detail_parquet, index=False)
    summary_df.to_parquet(summary_parquet, index=False)

    # HTML — reporte detallado por source
    summary_row = summary_df.iloc[0] if not summary_df.empty else pd.Series()
    write_siaf_html(
        detail_df=detail_df,
        summary_row=summary_row,
        config=config,
        output_path=report_html,
        total_rows=total_rows,
        n_parquets=len(parquet_paths),
    )

    # HTML — dashboard ejecutivo
    write_siaf_dashboard_html(
        summary_df=summary_df,
        detail_df=detail_df,
        config=config,
        output_path=dashboard_html,
        total_rows=total_rows,
        n_parquets=len(parquet_paths),
    )

    # Auditoría JSON
    audit_path = write_quality_audit(
        summary_df=summary_df,
        detail_df=detail_df,
        failed_samples_df=failed_samples_df,
        audit_root=audit_root,
        reports_root=reports_root,
        data_quality_root=data_quality_root,
        total_rows=total_rows,
        n_parquets=len(parquet_paths),
        started_at=started_at,
    )

    duration = (datetime.now() - started_at).total_seconds()
    logger.info("=== Pipeline Quality SIAF completado en %.1fs ===", duration)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "failed_samples_csv": str(failed_samples_csv),
        "dashboard_html": str(dashboard_html),
        "report_html": str(report_html),
        "detail_parquet": str(detail_parquet),
        "summary_parquet": str(summary_parquet),
        "audit_path": str(audit_path),
        "reports_dir": str(reports_root),
    }