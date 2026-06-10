"""
renamu_profiler.py

Profiling de la capa Bronze de RENAMU 2022.
Reutiliza las funciones genéricas de dataset_profiler.py.
Genera reportes CSV y HTML en reports/profiling/renamu/.
Registra auditoría JSON en data/audit/profiling/.

Diferencias con SISMEPRE:
  1. Un único dataset: municipalidades_2022.
  2. El Parquet Bronze se localiza con rglob (path con timestamp).
  3. Los vacíos son strings '' o ' ', no NaN reales — ya manejado
     por las funciones genéricas via is_missing_like_series.
"""

from __future__ import annotations

import json
import html as html_lib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from app.profiling.dataset_profiler import (
    profile_column,
    get_parquet_types,
    percent_bar,
    base_css,
    write_dashboard_html,
)
from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))

RENAMU_DATASET = "municipalidades_2022"
RENAMU_BRONZE_SUBFOLDER = "2022_municipalidades__Base_RENAMU_2022_f"


@dataclass
class RenamuProfilingPaths:
    bronze_root: Path = Path("data/bronze")
    reports_root: Path = Path("reports/profiling/renamu")
    audit_root: Path = Path("data/audit/profiling")


def find_renamu_parquet(bronze_root: Path) -> Path:
    search_root = bronze_root / "renamu" / RENAMU_BRONZE_SUBFOLDER
    candidates = sorted(
        search_root.rglob("*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ningún Parquet Bronze en: {search_root}\n"
            "Verifica que la ingesta haya generado el archivo correctamente."
        )
    if len(candidates) > 1:
        logger.warning(
            "Se encontraron %d Parquet en %s — se usa el más reciente: %s",
            len(candidates), search_root, candidates[0],
        )
    return candidates[0]


def profile_renamu_dataset(paths: RenamuProfilingPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet_path = find_renamu_parquet(paths.bronze_root)

    logger.info("Perfilando dataset RENAMU: %s", RENAMU_DATASET)
    df = pd.read_parquet(parquet_path)

    row_count = int(len(df))
    column_count = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = round((duplicate_rows / row_count) * 100, 4) if row_count else 0.0
    memory_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 4)
    parquet_types = get_parquet_types(parquet_path)

    profiles = [
        profile_column(RENAMU_DATASET, col, df[col], parquet_types.get(col, "unknown"), row_count)
        for col in df.columns
    ]
    profile_df = pd.DataFrame(profiles)

    summary = {
        "dataset": RENAMU_DATASET,
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": duplicate_rows_pct,
        "memory_mb": memory_mb,
        "columns_with_missing": int((profile_df["missing_like_count"] > 0).sum()),
        "columns_all_missing": int((profile_df["missing_like_pct"] == 100).sum()),
        "columns_high_missing_50pct": int((profile_df["missing_like_pct"] >= 50).sum()),
        "avg_missing_pct": round(float(profile_df["missing_like_pct"].mean()), 4) if len(profile_df) else 0.0,
        "avg_unique_pct": round(float(profile_df["unique_pct"].mean()), 4) if len(profile_df) else 0.0,
        "numeric_like_columns": int((profile_df["numeric_valid_pct"] >= 98).sum()),
        "datetime_like_columns": int((profile_df["datetime_valid_pct"] >= 90).sum()),
        "empty_or_null_columns": int((profile_df["inferred_type"] == "empty_or_null").sum()),
        "parquet_path": str(parquet_path),
        "profile_status": "OK",
    }

    logger.info(
        "Profiling %s terminado | filas=%s | columnas=%s | duplicados=%s | columnas_con_missing=%s",
        RENAMU_DATASET, row_count, column_count, duplicate_rows, summary["columns_with_missing"],
    )
    return profile_df, summary


def write_renamu_html_report(
    profile_df: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    doc = f"""
    <!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
    <title>Profiling RENAMU — {html_lib.escape(RENAMU_DATASET)}</title>
    <style>{base_css()}</style></head>
    <body><header><h1>Profiling RENAMU — {html_lib.escape(RENAMU_DATASET)}</h1>
    <p>Generado: {generated_at} | Capa: Bronze | No transforma datos</p></header>
    <main>
    <div class="cards">
      <div class="card"><div class="label">Filas</div><div class="value">{summary['row_count']:,}</div></div>
      <div class="card"><div class="label">Columnas</div><div class="value">{summary['column_count']:,}</div></div>
      <div class="card"><div class="label">Duplicados</div><div class="value">{summary['duplicate_rows']:,}</div></div>
      <div class="card"><div class="label">Cols con faltantes</div><div class="value">{summary['columns_with_missing']:,}</div></div>
    </div>
    <h2 class="section-title">Resumen del dataset</h2>{pd.DataFrame([summary]).to_html(index=False, escape=False)}
    <h2 class="section-title">Perfil de columnas</h2>{profile_df.to_html(index=False, escape=False)}
    </main></body></html>
    """
    output_path.write_text(doc, encoding="utf-8")


def write_renamu_audit(
    summary: dict[str, Any],
    paths: RenamuProfilingPaths,
    started_at: datetime,
) -> Path:
    finished_at = datetime.now()
    record = {
        "pipeline_name": "profiling_renamu",
        "status": "success",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "datasets_profiled": 1,
        "total_rows_profiled": summary["row_count"],
        "total_columns_profiled": summary["column_count"],
        "total_duplicate_rows": summary["duplicate_rows"],
        "outputs": {
            "summary_csv": str(paths.reports_root / "renamu_profile_summary.csv"),
            "detail_csv": str(paths.reports_root / "renamu_profile_detail.csv"),
            "dataset_html": str(paths.reports_root / "renamu_profile.html"),
            "dashboard_html": str(paths.reports_root / "renamu_profile_dashboard.html"),
        },
    }
    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = paths.audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"renamu_profiling_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


def run_renamu_profiling(
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/profiling/renamu"),
    audit_root: Path = Path("data/audit/profiling"),
) -> dict[str, str]:
    started_at = datetime.now()
    paths = RenamuProfilingPaths(Path(bronze_root), Path(reports_root), Path(audit_root))
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.audit_root.mkdir(parents=True, exist_ok=True)

    logger.info("Inicio de profiling RENAMU.")

    profile_df, summary = profile_renamu_dataset(paths)

    detail_csv = paths.reports_root / "renamu_profile_detail.csv"
    summary_csv = paths.reports_root / "renamu_profile_summary.csv"
    dataset_html = paths.reports_root / "renamu_profile.html"
    dashboard_html = paths.reports_root / "renamu_profile_dashboard.html"

    profile_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_renamu_html_report(profile_df, summary, dataset_html)
    write_dashboard_html(pd.DataFrame([summary]), dashboard_html)

    audit_path = write_renamu_audit(summary, paths, started_at)

    logger.info("Profiling RENAMU completado.")
    logger.info("Reporte HTML : %s", dataset_html)
    logger.info("Auditoría   : %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "dataset_html": str(dataset_html),
        "dashboard_html": str(dashboard_html),
        "audit_path": str(audit_path),
    }