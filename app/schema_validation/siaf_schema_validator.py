"""
siaf_schema_validator.py

Verificación estructural de SIAF Ingresos:
- Compara columnas descargadas en Bronze (Parquet) contra el contrato
  declarado en schema_rules_siaf.yaml.
- Revisa tipo lógico oficial, tipo técnico Parquet y tipo inferido
  desde valores reales.
- Evalúa columnas críticas, extra y política de columnas adicionales.
- Genera reportes CSV/HTML.
- Registra auditoría JSON y logs.

Esta etapa NO transforma datos y NO genera Silver.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de rutas
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationPaths:
    bronze_root: Path = Path("data/bronze")
    reports_root: Path = Path("reports/schema_validation/siaf")
    audit_root: Path = Path("data/audit/schema_validation")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def normalize_column_name(value: Any) -> str:
    """Normaliza nombres de columnas para comparar sin errores por espacios o mayúsculas."""
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    text = re.sub(r"\s+", "_", text)
    return text.upper()


def is_empty_value(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "nan", "NaN", "None", "null", "NULL"}


def infer_logical_type(series: pd.Series, sample_size: int = 10000) -> str:
    """
    Infere un tipo lógico de apoyo desde los valores reales.
    No reemplaza al diccionario oficial. Sirve como soporte para auditoría.
    """
    s = series.dropna().astype(str).map(str.strip)
    s = s[~s.map(is_empty_value)]

    if s.empty:
        return "empty_or_null"

    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)

    upper_values = set(s.str.upper().unique())
    if upper_values and upper_values.issubset({"0", "1", "S", "N", "SI", "NO", "TRUE", "FALSE", "A", "I", "E", "R", "M"}):
        return "categorical_indicator"

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio >= 0.98:
        non_null_numeric = numeric.dropna()
        if (non_null_numeric % 1 == 0).all():
            return "integer"
        return "decimal"

    unique_ratio = s.nunique(dropna=True) / len(s)
    if unique_ratio <= 0.10:
        return "categorical_text"

    return "text"


def load_schema_rules(config_path: Path) -> dict[str, Any]:
    """Carga el YAML de reglas de schema."""
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de schema: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("El archivo YAML de schema no tiene estructura válida.")
    for section in ["settings", "expected_columns", "column_types", "critical_columns"]:
        if section not in config:
            raise ValueError(f"Falta la sección obligatoria en schema_rules_siaf.yaml: {section}")
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Lectura de Bronze
# ──────────────────────────────────────────────────────────────────────────────

def get_bronze_parquet_paths(bronze_root: Path) -> dict[str, Path]:
    """
    Descubre todos los Parquet Bronze disponibles en data/bronze/siaf.

    SIAF puede tener múltiples archivos por año (2012_ingreso, 2024_ingreso, etc.).
    Cada subcarpeta directamente bajo siaf/ se trata como un dataset independiente.
    """
    siaf_root = bronze_root / "siaf"
    if not siaf_root.exists():
        return {}

    paths: dict[str, Path] = {}
    for subdir in sorted(siaf_root.iterdir()):
        if not subdir.is_dir():
            continue
        # Buscar primero parquet con nombre estándar, luego cualquier parquet en la raíz.
        candidates = [
            subdir / f"{subdir.name}.parquet",
            subdir / f"{subdir.name}_raw.parquet",
        ] + sorted(subdir.glob("*.parquet"))

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                paths[subdir.name] = candidate
                break

    return paths


def read_parquet_schema_and_inference(dataset: str, parquet_path: Path) -> tuple[pd.DataFrame, int]:
    """
    Lee schema técnico Parquet e infiere tipo lógico de apoyo desde valores reales.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe Parquet Bronze: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    row_count = parquet_file.metadata.num_rows

    df = pd.read_parquet(parquet_path)

    records = []
    for field in schema:
        col_name = field.name
        norm_col = normalize_column_name(col_name)
        inferred_type = infer_logical_type(df[col_name]) if col_name in df.columns else "unknown"

        records.append({
            "dataset": dataset,
            "downloaded_column": norm_col,
            "original_downloaded_column": col_name,
            "parquet_type": str(field.type),
            "nullable": field.nullable,
            "inferred_type": inferred_type,
            "local_rows_parquet": row_count,
            "parquet_path": str(parquet_path),
        })

    return pd.DataFrame(records), row_count


# ──────────────────────────────────────────────────────────────────────────────
# Comparación contra YAML
# ──────────────────────────────────────────────────────────────────────────────

def compare_dataset(
    dataset: str,
    parquet_path: Path,
    config: dict[str, Any],
    paths: ValidationPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compara el schema del Parquet Bronze contra el contrato declarado en el YAML.

    Evalúa:
    - Columnas esperadas vs presentes.
    - Columnas críticas faltantes.
    - Columnas extra según política.
    - Tipo lógico esperado vs inferido.
    - Nullable esperado vs real.
    """
    logger.info("Validando schema de dataset SIAF: %s", dataset)

    expected_columns: list[str] = [normalize_column_name(c) for c in config["expected_columns"]]
    column_types: dict[str, dict[str, Any]] = {
        normalize_column_name(k): v for k, v in config.get("column_types", {}).items()
    }
    critical_columns: list[str] = [normalize_column_name(c) for c in config.get("critical_columns", [])]
    extra_policy: str = config.get("extra_columns_policy", {}).get("action", "warn")
    extra_severity: str = config.get("extra_columns_policy", {}).get("severity", "Baja")

    downloaded_df, row_count = read_parquet_schema_and_inference(dataset, parquet_path)
    downloaded_set = set(downloaded_df["downloaded_column"].tolist())

    rows: list[dict[str, Any]] = []
    missing_columns: list[str] = []
    extra_columns: list[str] = []
    missing_critical: list[str] = []

    # ── Columnas esperadas ────────────────────────────────────────────────────
    for col in expected_columns:
        present = col in downloaded_set
        parquet_row = downloaded_df[downloaded_df["downloaded_column"] == col]
        parquet_type = parquet_row["parquet_type"].iloc[0] if not parquet_row.empty else "AUSENTE"
        inferred_type = parquet_row["inferred_type"].iloc[0] if not parquet_row.empty else "AUSENTE"
        nullable_parquet = parquet_row["nullable"].iloc[0] if not parquet_row.empty else None

        expected_meta = column_types.get(col, {})
        expected_logical = expected_meta.get("logical_type", "")
        expected_nullable = expected_meta.get("nullable", True)

        if not present:
            status = "FALTA_EN_PARQUET"
            missing_columns.append(col)
            if col in critical_columns:
                missing_critical.append(col)
        else:
            nullable_mismatch = (expected_nullable is False) and (nullable_parquet is True)
            status = "OK" if not nullable_mismatch else "NULLABLE_MISMATCH"

        rows.append({
            "dataset": dataset,
            "official_column": col,
            "expected_logical_type": expected_logical,
            "expected_nullable": expected_nullable,
            "is_critical": col in critical_columns,
            "parquet_type": parquet_type,
            "inferred_type": inferred_type,
            "nullable_parquet": nullable_parquet,
            "comparison_status": status,
            "dictionary_source": "schema_rules_siaf.yaml",
        })

    # ── Columnas extra (en Parquet pero no en YAML) ───────────────────────────
    for col in sorted(downloaded_set - set(expected_columns)):
        extra_columns.append(col)
        parquet_row = downloaded_df[downloaded_df["downloaded_column"] == col]
        parquet_type = parquet_row["parquet_type"].iloc[0] if not parquet_row.empty else ""
        inferred_type = parquet_row["inferred_type"].iloc[0] if not parquet_row.empty else ""

        rows.append({
            "dataset": dataset,
            "official_column": col,
            "expected_logical_type": "",
            "expected_nullable": None,
            "is_critical": False,
            "parquet_type": parquet_type,
            "inferred_type": inferred_type,
            "nullable_parquet": parquet_row["nullable"].iloc[0] if not parquet_row.empty else None,
            "comparison_status": f"EXTRA_EN_PARQUET_{extra_policy.upper()}",
            "dictionary_source": f"extra_column_policy={extra_policy}",
        })

    has_missing_critical = len(missing_critical) > 0
    has_missing = len(missing_columns) > 0
    has_extra = len(extra_columns) > 0

    if has_missing_critical:
        validation_status = "FALLA_CRITICA"
    elif has_missing or (has_extra and extra_policy == "fail"):
        validation_status = "REVISAR"
    else:
        validation_status = "OK"

    summary = {
        "dataset": dataset,
        "dictionary_source": "schema_rules_siaf.yaml",
        "expected_columns": len(expected_columns),
        "downloaded_columns": len(downloaded_set),
        "missing_in_parquet": len(missing_columns),
        "extra_in_parquet": len(extra_columns),
        "missing_critical": len(missing_critical),
        "missing_critical_names": ", ".join(missing_critical) if missing_critical else "",
        "local_rows_parquet": row_count,
        "parquet_path": str(parquet_path),
        "validation_status": validation_status,
    }

    logger.info(
        "Resultado %s | status=%s | esperadas=%s | parquet=%s | faltantes=%s | extras=%s | criticas_faltantes=%s | filas=%s",
        dataset,
        validation_status,
        len(expected_columns),
        len(downloaded_set),
        len(missing_columns),
        len(extra_columns),
        len(missing_critical),
        row_count,
    )

    return pd.DataFrame(rows), summary


# ──────────────────────────────────────────────────────────────────────────────
# Reportes
# ──────────────────────────────────────────────────────────────────────────────

def _css() -> str:
    return """
    body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
    header{background:#1f2430;color:white;padding:24px 32px}
    header h1{margin:0;font-size:25px} header p{margin:8px 0 0;color:#cbd5e1}
    main{padding:24px 32px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #3b82f6}
    .label{color:#6b7280;text-transform:uppercase;font-size:12px;letter-spacing:.05em}
    .value{font-size:28px;font-weight:bold;margin-top:8px;color:#2f7d46}
    .section-title{color:#4b5563;text-transform:uppercase;letter-spacing:.08em;font-size:13px;margin:24px 0 12px}
    table{border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:middle}
    th{background:#111827;color:white}
    tr:nth-child(even){background:#fafafa}
    .badge{padding:5px 10px;border-radius:999px;font-weight:bold;font-size:12px;white-space:nowrap}
    .ok{background:#d9eadf;color:#245c36}
    .falla{background:#fee2e2;color:#991b1b}
    .revisar{background:#fff3cd;color:#7a5b00}
    .note{background:#eef2ff;border-left:4px solid #3b82f6;border-radius:10px;padding:14px 16px;margin-bottom:20px}
    code{background:#eef2ff;padding:2px 4px;border-radius:4px}
    .alert-card{background:white;border-left:4px solid #c24141;border-radius:10px;padding:12px 16px;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    """


def _status_badge(status: str) -> str:
    import html as _html
    cls = "ok" if status == "OK" else ("falla" if "CRITICA" in status else "revisar")
    return f'<span class="badge {cls}">{_html.escape(status)}</span>'


def write_html_report(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    import html as _html
    generated_at = datetime.now().isoformat(timespec="seconds")

    total_ok = int((summary_df["validation_status"] == "OK").sum())
    total_critical = int((summary_df["validation_status"] == "FALLA_CRITICA").sum())
    total_review = int((summary_df["validation_status"] == "REVISAR").sum())
    total_rows = int(summary_df["local_rows_parquet"].sum()) if not summary_df.empty else 0
    total_datasets = len(summary_df)

    # Sección de observaciones
    issues = detail_df[detail_df["comparison_status"] != "OK"].copy()
    if issues.empty:
        alerts_html = '<div class="note">No se detectaron columnas faltantes, críticas ni extras.</div>'
    else:
        alerts_html = "".join([
            f"<div class='alert-card'>"
            f"<b>{_html.escape(str(r['dataset']))}</b> — "
            f"<code>{_html.escape(str(r['official_column']))}</code> | "
            f"Estado: {_html.escape(str(r['comparison_status']))} | "
            f"Crítica: {_html.escape(str(r['is_critical']))} | "
            f"Tipo esperado: {_html.escape(str(r['expected_logical_type']))} | "
            f"Tipo Parquet: {_html.escape(str(r['parquet_type']))}"
            f"</div>"
            for _, r in issues.iterrows()
        ])

    summary_rows_html = "".join([
        f"<tr>"
        f"<td><b>{_html.escape(str(r['dataset']))}</b></td>"
        f"<td>{int(r['expected_columns'])}</td>"
        f"<td>{int(r['downloaded_columns'])}</td>"
        f"<td>{int(r['missing_in_parquet'])}</td>"
        f"<td>{int(r['extra_in_parquet'])}</td>"
        f"<td><b style='color:#c24141'>{int(r['missing_critical'])}</b></td>"
        f"<td>{int(r['local_rows_parquet']):,}</td>"
        f"<td>{_status_badge(str(r['validation_status']))}</td>"
        f"</tr>"
        for _, r in summary_df.iterrows()
    ])

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Validación de Esquema SIAF</title>
  <style>{_css()}</style>
</head>
<body>
  <header>
    <h1>Validación estructural SIAF — Presupuesto y Ejecución de Ingresos</h1>
    <p>Generado: {generated_at} | Capa evaluada: Bronze | Este reporte no modifica datos</p>
  </header>
  <main>
    <div class="note">
      Este reporte compara las columnas declaradas en <code>schema_rules_siaf.yaml</code>
      contra las columnas descargadas en Bronze (Parquet). Columnas <b>críticas faltantes</b>
      impiden que el pipeline continúe. Las columnas extra se registran según la política
      configurada (<code>extra_columns_policy</code>).
    </div>

    <div class="cards">
      <div class="card"><div class="label">Datasets evaluados</div><div class="value">{total_datasets}</div></div>
      <div class="card"><div class="label">Total filas Bronze</div><div class="value">{total_rows:,}</div></div>
      <div class="card"><div class="label">Datasets OK</div><div class="value">{total_ok}</div></div>
      <div class="card"><div class="label">Fallas críticas</div><div class="value" style="color:#c24141">{total_critical}</div></div>
    </div>

    <h2 class="section-title">Resumen por dataset</h2>
    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Col. esperadas</th>
          <th>Col. en Parquet</th>
          <th>Faltantes</th>
          <th>Extras</th>
          <th>Críticas faltantes</th>
          <th>Filas Bronze</th>
          <th>Estado</th>
        </tr>
      </thead>
      <tbody>{summary_rows_html}</tbody>
    </table>

    <h2 class="section-title">Principales observaciones</h2>
    {alerts_html}

    <h2 class="section-title">Detalle columna por columna</h2>
    {detail_df.to_html(index=False, escape=True, classes="detail")}
  </main>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def write_audit(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    paths: ValidationPaths,
    started_at: datetime,
) -> Path:
    finished_at = datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()

    has_critical = (summary_df["validation_status"] == "FALLA_CRITICA").any()
    has_review = (summary_df["validation_status"] == "REVISAR").any()
    status = "critical" if has_critical else ("warning" if has_review else "success")

    audit_record = {
        "pipeline_name": "schema_validation_siaf",
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "datasets_evaluated": int(len(summary_df)),
        "datasets_ok": int((summary_df["validation_status"] == "OK").sum()),
        "datasets_review": int((summary_df["validation_status"] == "REVISAR").sum()),
        "datasets_critical": int((summary_df["validation_status"] == "FALLA_CRITICA").sum()),
        "columns_evaluated": int(len(detail_df)),
        "missing_columns_total": int((detail_df["comparison_status"] == "FALTA_EN_PARQUET").sum()),
        "extra_columns_total": int(detail_df["comparison_status"].str.startswith("EXTRA_EN_PARQUET").sum()),
        "missing_critical_total": int(summary_df["missing_critical"].sum()),
        "total_rows_bronze": int(summary_df["local_rows_parquet"].sum()),
        "outputs": {
            "summary_csv": str(paths.reports_root / "siaf_schema_validation_summary.csv"),
            "detail_csv": str(paths.reports_root / "siaf_schema_validation_detail.csv"),
            "html": str(paths.reports_root / "siaf_schema_validation.html"),
        },
    }

    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = paths.audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"siaf_schema_validation_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(audit_record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


# ──────────────────────────────────────────────────────────────────────────────
# Función principal exportable
# ──────────────────────────────────────────────────────────────────────────────

def run_siaf_schema_validation(
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/schema_validation/siaf"),
    audit_root: Path = Path("data/audit/schema_validation"),
    config_path: Path = Path("app/config/schema_rules_siaf.yaml"),
) -> dict[str, str]:
    """
    Pipeline de validación estructural para SIAF Ingresos.

    Flujo:
    1. Carga el contrato de schema desde schema_rules_siaf.yaml.
    2. Descubre todos los Parquet Bronze bajo data/bronze/siaf/.
    3. Compara cada archivo contra el contrato.
    4. Genera CSV, HTML y auditoría JSON.
    5. No modifica datos Bronze.
    """
    started_at = datetime.now()
    paths = ValidationPaths(
        bronze_root=Path(bronze_root),
        reports_root=Path(reports_root),
        audit_root=Path(audit_root),
    )

    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.audit_root.mkdir(parents=True, exist_ok=True)

    logger.info("Inicio de validación estructural SIAF.")

    config = load_schema_rules(config_path)
    parquet_paths = get_bronze_parquet_paths(paths.bronze_root)

    if not parquet_paths:
        logger.warning("No se encontraron Parquet Bronze bajo %s/siaf/", paths.bronze_root)

    all_details: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for dataset, parquet_path in parquet_paths.items():
        try:
            detail, summary = compare_dataset(dataset, parquet_path, config, paths)
            all_details.append(detail)
            summaries.append(summary)
        except Exception as exc:
            logger.error("Error procesando dataset %s: %s", dataset, exc)
            summaries.append({
                "dataset": dataset,
                "dictionary_source": "schema_rules_siaf.yaml",
                "expected_columns": 0,
                "downloaded_columns": 0,
                "missing_in_parquet": 0,
                "extra_in_parquet": 0,
                "missing_critical": 0,
                "missing_critical_names": "",
                "local_rows_parquet": 0,
                "parquet_path": "",
                "validation_status": f"ERROR: {exc}",
            })

    detail_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    detail_csv = paths.reports_root / "siaf_schema_validation_detail.csv"
    summary_csv = paths.reports_root / "siaf_schema_validation_summary.csv"
    html_report = paths.reports_root / "siaf_schema_validation.html"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_html_report(detail_df, summary_df, html_report)

    audit_path = write_audit(summary_df, detail_df, paths, started_at)

    logger.info("Validación estructural SIAF completada.")
    logger.info("Reporte HTML: %s", html_report)
    logger.info("Auditoría: %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "html_report": str(html_report),
        "audit_path": str(audit_path),
    }