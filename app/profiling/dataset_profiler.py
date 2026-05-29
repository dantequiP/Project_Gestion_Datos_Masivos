
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))

SISMEPRE_DATASETS = [
    "rentas_preguntas",
    "rentas_estadistica",
    "rentas_formulario",
    "rentas_esat_estadistica_atm",
    "rentas_respuestas",
    "rentas_ano_aplicacion",
    "rentas_entidad_estado",
]

MISSING_TOKENS = {"", "nan", "NaN", "None", "none", "NULL", "null", "<NA>", "NaT"}


@dataclass
class ProfilingPaths:
    bronze_root: Path = Path("data/bronze")
    reports_root: Path = Path("reports/profiling/sismepre")
    audit_root: Path = Path("data/audit/profiling")


def safe_strip(value: Any) -> str:
    """Convierte cualquier valor a texto seguro. Evita error con float/NA en columnas Arrow."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_clean_text_series(series: pd.Series) -> pd.Series:
    return series.map(safe_strip)


def is_missing_like_series(series: pd.Series) -> pd.Series:
    text = to_clean_text_series(series)
    return series.isna() | text.isin(MISSING_TOKENS)


def normalize_sample_values(values: pd.Series, limit: int = 5) -> str:
    out = []
    for value in values.map(safe_strip):
        if value and value not in MISSING_TOKENS and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return json.dumps(out, ensure_ascii=False)


def top_values_as_json(series: pd.Series, limit: int = 10) -> str:
    mask = is_missing_like_series(series)
    s = to_clean_text_series(series[~mask])
    return json.dumps(s.value_counts(dropna=True).head(limit).to_dict(), ensure_ascii=False)


def infer_logical_type(series: pd.Series, sample_size: int = 10000) -> str:
    mask = is_missing_like_series(series)
    s = to_clean_text_series(series[~mask])
    if s.empty:
        return "empty_or_null"
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)

    upper_values = set(s.str.upper().unique())
    if upper_values and upper_values.issubset({"0", "1", "S", "N", "SI", "NO", "TRUE", "FALSE", "A", "I"}):
        return "categorical_indicator"

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() >= 0.98:
        non_null_numeric = numeric.dropna()
        if not non_null_numeric.empty and (non_null_numeric % 1 == 0).all():
            return "integer"
        return "decimal"

    best_date_ratio = 0.0
    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        best_date_ratio = max(best_date_ratio, parsed.notna().mean())
    if best_date_ratio >= 0.90:
        return "datetime_or_date"

    email_ratio = s.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False).mean()
    if email_ratio >= 0.80:
        return "email"

    unique_ratio = s.nunique(dropna=True) / len(s)
    if unique_ratio <= 0.10:
        return "categorical_text"
    return "text"


def detect_best_datetime_ratio(series: pd.Series) -> tuple[float, str, str, str]:
    mask = is_missing_like_series(series)
    s = to_clean_text_series(series[~mask])
    if s.empty:
        return 0.0, "", "", ""

    best_ratio = 0.0
    best_format = ""
    best_parsed = None
    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best_format = fmt
            best_parsed = parsed

    if best_parsed is not None and best_parsed.notna().any():
        return float(best_ratio), best_format, str(best_parsed.min()), str(best_parsed.max())
    return float(best_ratio), best_format, "", ""


def classify_cardinality(unique_pct: float) -> str:
    if unique_pct > 95:
        return "alta_posible_identificador"
    if unique_pct > 50:
        return "media_alta"
    if unique_pct > 5:
        return "media"
    return "baja_categorica"


def profile_column(dataset: str, column_name: str, series: pd.Series, parquet_type: str, row_count: int) -> dict[str, Any]:
    text_series = to_clean_text_series(series)
    missing_mask = is_missing_like_series(series)
    as_text = text_series[~missing_mask]

    null_count = int(series.isna().sum())
    empty_string_count = int(text_series.eq("").sum())
    missing_like_count = int(missing_mask.sum())
    unique_count = int(as_text.nunique(dropna=True))
    unique_pct = round((unique_count / row_count) * 100, 4) if row_count else 0.0

    lengths = as_text.map(len) if not as_text.empty else pd.Series(dtype=int)
    min_length = int(lengths.min()) if not lengths.empty else 0
    max_length = int(lengths.max()) if not lengths.empty else 0
    avg_length = round(float(lengths.mean()), 4) if not lengths.empty else 0.0

    numeric = pd.to_numeric(as_text, errors="coerce") if not as_text.empty else pd.Series(dtype=float)
    numeric_valid_count = int(numeric.notna().sum())
    numeric_valid_pct = round((numeric_valid_count / len(as_text)) * 100, 4) if len(as_text) else 0.0

    numeric_min = numeric_max = numeric_mean = q1 = q3 = iqr = outlier_count = outlier_pct = ""
    if numeric_valid_count > 0:
        numeric_min = float(numeric.min())
        numeric_max = float(numeric.max())
        numeric_mean = round(float(numeric.mean()), 4)
        q1v = numeric.quantile(0.25)
        q3v = numeric.quantile(0.75)
        iqrv = q3v - q1v
        q1 = round(float(q1v), 4)
        q3 = round(float(q3v), 4)
        iqr = round(float(iqrv), 4)
        if iqrv > 0:
            lower = q1v - 1.5 * iqrv
            upper = q3v + 1.5 * iqrv
            outlier_count = int(((numeric < lower) | (numeric > upper)).sum())
            outlier_pct = round((outlier_count / numeric_valid_count) * 100, 4)

    date_ratio, date_format, date_min, date_max = detect_best_datetime_ratio(series)

    return {
        "dataset": dataset,
        "column_name": column_name,
        "parquet_type": parquet_type,
        "inferred_type": infer_logical_type(series),
        "row_count": row_count,
        "null_count": null_count,
        "null_pct": round((null_count / row_count) * 100, 4) if row_count else 0.0,
        "empty_string_count": empty_string_count,
        "empty_string_pct": round((empty_string_count / row_count) * 100, 4) if row_count else 0.0,
        "missing_like_count": missing_like_count,
        "missing_like_pct": round((missing_like_count / row_count) * 100, 4) if row_count else 0.0,
        "non_missing_count": int(len(as_text)),
        "unique_count": unique_count,
        "unique_pct": unique_pct,
        "cardinality_class": classify_cardinality(unique_pct),
        "min_length": min_length,
        "max_length": max_length,
        "avg_length": avg_length,
        "numeric_valid_count": numeric_valid_count,
        "numeric_valid_pct": numeric_valid_pct,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "numeric_mean": numeric_mean,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "outlier_count_iqr": outlier_count,
        "outlier_pct_iqr": outlier_pct,
        "datetime_valid_pct": round(date_ratio * 100, 4),
        "datetime_detected_format": date_format,
        "datetime_min": date_min,
        "datetime_max": date_max,
        "sample_values": normalize_sample_values(as_text),
        "top_values": top_values_as_json(series),
    }


def get_parquet_path(dataset: str, bronze_root: Path) -> Path:
    return bronze_root / "sismepre" / dataset / f"{dataset}_raw.parquet"


def get_parquet_types(parquet_path: Path) -> dict[str, str]:
    schema = pq.read_schema(parquet_path)
    return {field.name: str(field.type) for field in schema}


def profile_dataset(dataset: str, paths: ProfilingPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet_path = get_parquet_path(dataset, paths.bronze_root)
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe el Parquet Bronze esperado: {parquet_path}")

    logger.info("Perfilando dataset SISMEPRE: %s", dataset)
    df = pd.read_parquet(parquet_path)

    row_count = int(len(df))
    column_count = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = round((duplicate_rows / row_count) * 100, 4) if row_count else 0.0
    memory_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 4)
    parquet_types = get_parquet_types(parquet_path)

    profiles = [
        profile_column(dataset, col, df[col], parquet_types.get(col, "unknown"), row_count)
        for col in df.columns
    ]
    profile_df = pd.DataFrame(profiles)

    summary = {
        "dataset": dataset,
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
        dataset, row_count, column_count, duplicate_rows, summary["columns_with_missing"]
    )
    return profile_df, summary


def percent_bar(value: float) -> str:
    value = max(0.0, min(100.0, float(value)))
    return (
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{value:.2f}%"></div></div>'
        f'<span class="bar-label">{value:.1f}%</span>'
    )


def base_css() -> str:
    return """
    body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
    header{background:#1f2430;color:white;padding:24px 32px}
    header h1{margin:0;font-size:24px}
    header p{margin:8px 0 0;color:#cbd5e1}
    main{padding:24px 32px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #3b82f6}
    .label{color:#6b7280;text-transform:uppercase;font-size:12px;letter-spacing:.05em}
    .value{font-size:28px;font-weight:bold;margin-top:8px;color:#2f7d46}
    .section-title{color:#4b5563;text-transform:uppercase;letter-spacing:.08em;font-size:13px;margin:24px 0 12px}
    table{border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:middle}
    th{background:#111827;color:white}
    tr:nth-child(even){background:#fafafa}
    .bar-wrap{display:inline-block;width:120px;height:7px;background:#e5e7eb;border-radius:999px;vertical-align:middle;margin-right:8px}
    .bar-fill{height:7px;background:#2f7d46;border-radius:999px}
    .bar-label{color:#2f7d46;font-weight:bold}
    .badge{padding:5px 10px;border-radius:999px;font-weight:bold;font-size:12px}
    .ok{background:#d9eadf;color:#245c36}
    .warn{background:#fff3cd;color:#7a5b00}
    """


def write_dataset_html_report(dataset: str, profile_df: pd.DataFrame, summary: dict[str, Any], output_path: Path) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    doc = f"""
    <!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
    <title>Profiling SISMEPRE - {html.escape(dataset)}</title><style>{base_css()}</style></head>
    <body><header><h1>Profiling SISMEPRE — {html.escape(dataset)}</h1>
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


def write_dashboard_html(summary_df: pd.DataFrame, output_path: Path) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    total_datasets = len(summary_df)
    total_rows = int(summary_df["row_count"].sum())
    total_columns = int(summary_df["column_count"].sum())
    total_duplicates = int(summary_df["duplicate_rows"].sum())

    rows_html = []
    for _, row in summary_df.iterrows():
        completeness = max(0, 100 - float(row["avg_missing_pct"]))
        status = "REVISAR" if row["columns_all_missing"] > 0 or row["columns_high_missing_50pct"] > 0 else "OK"
        badge_class = "warn" if status == "REVISAR" else "ok"
        rows_html.append(f"""
        <tr>
          <td><b>{html.escape(str(row['dataset']))}</b></td>
          <td>{int(row['row_count']):,}</td>
          <td>{int(row['column_count']):,}</td>
          <td>{int(row['duplicate_rows']):,}</td>
          <td>{percent_bar(completeness)}</td>
          <td>{int(row['columns_with_missing'])}</td>
          <td>{int(row['columns_all_missing'])}</td>
          <td>{int(row['numeric_like_columns'])}</td>
          <td>{int(row['datetime_like_columns'])}</td>
          <td><span class="badge {badge_class}">{status}</span></td>
        </tr>
        """)

    doc = f"""
    <!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
    <title>Dashboard Profiling SISMEPRE — Bronze</title><style>{base_css()}</style></head>
    <body><header><h1>Dashboard de Profiling — SISMEPRE Bronze</h1>
    <p>Generado: {generated_at} | Pipeline: profiling_sismepre</p></header>
    <main>
    <h2 class="section-title">Resumen general</h2>
    <div class="cards">
      <div class="card"><div class="label">Datasets analizados</div><div class="value">{total_datasets}</div></div>
      <div class="card"><div class="label">Filas perfiladas</div><div class="value">{total_rows:,}</div></div>
      <div class="card"><div class="label">Columnas perfiladas</div><div class="value">{total_columns:,}</div></div>
      <div class="card"><div class="label">Duplicados totales</div><div class="value">{total_duplicates:,}</div></div>
    </div>
    <h2 class="section-title">Tabla resumen</h2>
    <table><thead><tr>
      <th>Dataset</th><th>Filas</th><th>Columnas</th><th>Duplicados</th><th>Completitud aprox.</th>
      <th>Cols con faltantes</th><th>Cols 100% vacías</th><th>Cols numéricas</th><th>Cols fecha</th><th>Estado</th>
    </tr></thead><tbody>{''.join(rows_html)}</tbody></table>
    </main></body></html>
    """
    output_path.write_text(doc, encoding="utf-8")


def write_audit(summary_df: pd.DataFrame, paths: ProfilingPaths, started_at: datetime) -> Path:
    finished_at = datetime.now()
    record = {
        "pipeline_name": "profiling_sismepre",
        "status": "success",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "datasets_profiled": int(len(summary_df)),
        "total_rows_profiled": int(summary_df["row_count"].sum()),
        "total_columns_profiled": int(summary_df["column_count"].sum()),
        "total_duplicate_rows": int(summary_df["duplicate_rows"].sum()),
        "outputs": {
            "summary_csv": str(paths.reports_root / "sismepre_profile_summary.csv"),
            "detail_csv": str(paths.reports_root / "sismepre_profile_detail.csv"),
            "dashboard_html": str(paths.reports_root / "sismepre_profile_dashboard.html"),
        },
    }
    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = paths.audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"sismepre_profiling_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


def run_sismepre_profiling(
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/profiling/sismepre"),
    audit_root: Path = Path("data/audit/profiling"),
) -> dict[str, str]:
    started_at = datetime.now()
    paths = ProfilingPaths(Path(bronze_root), Path(reports_root), Path(audit_root))
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.audit_root.mkdir(parents=True, exist_ok=True)

    logger.info("Inicio de profiling SISMEPRE.")
    all_profiles = []
    summaries = []

    for dataset in SISMEPRE_DATASETS:
        profile_df, summary = profile_dataset(dataset, paths)
        profile_df.to_csv(paths.reports_root / f"{dataset}_profile.csv", index=False, encoding="utf-8-sig")
        write_dataset_html_report(dataset, profile_df, summary, paths.reports_root / f"{dataset}_profile.html")
        all_profiles.append(profile_df)
        summaries.append(summary)

    detail_df = pd.concat(all_profiles, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_csv = paths.reports_root / "sismepre_profile_detail.csv"
    summary_csv = paths.reports_root / "sismepre_profile_summary.csv"
    dashboard_html = paths.reports_root / "sismepre_profile_dashboard.html"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_dashboard_html(summary_df, dashboard_html)
    audit_path = write_audit(summary_df, paths, started_at)

    logger.info("Profiling SISMEPRE completado.")
    logger.info("Dashboard HTML: %s", dashboard_html)
    logger.info("Auditoría: %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "dashboard_html": str(dashboard_html),
        "reports_dir": str(paths.reports_root),
        "audit_path": str(audit_path),
    }
