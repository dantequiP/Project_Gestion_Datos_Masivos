"""
sismepre_schema_validator.py

Verificación estructural de SISMEPRE:
- Compara columnas descargadas en Bronze (Parquet) contra diccionarios oficiales.
- Revisa tipo lógico oficial, tipo técnico Parquet y tipo inferido desde valores reales.
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
import requests

from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))


SISMEPRE_DATASETS: dict[str, str] = {
    "rentas_preguntas": "96b53d87-dc98-41ee-8fea-3b45e6201942",
    "rentas_estadistica": "69abaf52-1d36-4efb-86d0-38af739896c7",
    "rentas_formulario": "5473b6da-2453-48d0-bc50-e38afc8e732c",
    "rentas_esat_estadistica_atm": "52f97ee4-6a52-465a-98e8-7e5ba8328b30",
    "rentas_respuestas": "d03c11b0-3c33-4c61-85fc-507bbcaf9cae",
    "rentas_ano_aplicacion": "d107f21c-686a-4217-ac0c-9f96b4f00df0",
    "rentas_entidad_estado": "5989e884-f198-4a92-817d-890d71e8984a",
}

CKAN_BASE_URL = "https://api.datosabiertos.mef.gob.pe/DatosAbiertos/v1/datastore_search"
DICTIONARY_URL_TEMPLATE = "https://fs.datosabiertos.mef.gob.pe/datastorefiles/{dataset}_diccionario.csv"


@dataclass
class ValidationPaths:
    bronze_root: Path = Path("data/bronze")
    reports_root: Path = Path("reports/schema_validation/sismepre")
    audit_root: Path = Path("data/audit/schema_validation")


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
    No reemplaza al diccionario oficial. Sirve como soporte para profiling/calidad/Silver.
    """
    s = series.dropna().astype(str).map(str.strip)
    s = s[~s.map(is_empty_value)]

    if s.empty:
        return "empty_or_null"

    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)

    # Boolean / indicador frecuente
    upper_values = set(s.str.upper().unique())
    if upper_values and upper_values.issubset({"0", "1", "S", "N", "SI", "NO", "TRUE", "FALSE", "A", "I"}):
        return "categorical_indicator"

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio >= 0.98:
        non_null_numeric = numeric.dropna()
        if (non_null_numeric % 1 == 0).all():
            return "integer"
        return "decimal"

    # Fechas frecuentes en portal MEF: dd/mm/yyyy hh:mm:ss, dd/mm/yyyy, yyyy-mm-dd
    date_formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    best_ratio = 0.0
    for fmt in date_formats:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        best_ratio = max(best_ratio, parsed.notna().mean())
    if best_ratio >= 0.90:
        return "datetime_or_date"

    # Email aproximado
    email_ratio = s.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False).mean()
    if email_ratio >= 0.80:
        return "email"

    unique_ratio = s.nunique(dropna=True) / len(s)
    if unique_ratio <= 0.10:
        return "categorical_text"

    return "text"


def read_official_dictionary(dataset: str) -> tuple[pd.DataFrame, str, str | None]:
    """
    Lee el archivo separado *_diccionario.csv.
    Retorna:
      - dataframe normalizado con official_column, official_type, official_description
      - fuente usada
      - error si hubo
    """
    url = DICTIONARY_URL_TEMPLATE.format(dataset=dataset)

    try:
        df = pd.read_csv(url, dtype=str, encoding="utf-8-sig")
    except Exception as exc:
        return pd.DataFrame(), "dictionary_csv_error", str(exc)

    if df.empty:
        return pd.DataFrame(), "dictionary_csv_empty", "El CSV de diccionario no devolvió filas."

    raw_cols = {normalize_column_name(c): c for c in df.columns}

    # Formato común de archivos separados: VARIABLE, TIPO_DATO, DESCRIPCION
    variable_col = raw_cols.get("VARIABLE") or raw_cols.get("COLUMNA") or raw_cols.get("CAMPO")
    type_col = raw_cols.get("TIPO_DATO") or raw_cols.get("TIPO") or raw_cols.get("TYPE")
    description_col = raw_cols.get("DESCRIPCION") or raw_cols.get("DESCRIPCIÓN") or raw_cols.get("DESCRIPTION")

    if not variable_col:
        return pd.DataFrame(), "dictionary_csv_unrecognized", f"Columnas no reconocidas: {list(df.columns)}"

    out = pd.DataFrame({
        "dataset": dataset,
        "official_column": df[variable_col].map(normalize_column_name),
        "official_type": df[type_col].astype(str).str.strip() if type_col else "",
        "official_description": df[description_col].astype(str).str.strip() if description_col else "",
        "dictionary_source": "dictionary_csv",
        "dictionary_url": url,
    })

    out = out[out["official_column"] != ""].drop_duplicates(subset=["dataset", "official_column"])
    return out, "dictionary_csv", None


def read_api_fields_as_fallback(dataset: str, resource_id: str, timeout: int = 60) -> tuple[pd.DataFrame, str, str | None]:
    """
    Respaldo cuando el archivo *_diccionario.csv falla.
    Usa datastore_search limit=0 para obtener fields del recurso de datos.
    Normalmente trae nombres de columnas y tipos técnicos del portal.
    """
    try:
        response = requests.get(
            CKAN_BASE_URL,
            params={"resource_id": resource_id, "limit": 0},
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as exc:
        return pd.DataFrame(), "api_fields_error", str(exc)

    result = raw.get("result", {}) or {}
    fields = result.get("fields", []) or []

    records = []
    for field in fields:
        name = field.get("id") or field.get("name")
        if not name or str(name).startswith("_"):
            continue
        records.append({
            "dataset": dataset,
            "official_column": normalize_column_name(name),
            "official_type": str(field.get("type", "")),
            "official_description": "",
            "dictionary_source": "api_fields_fallback",
            "dictionary_url": CKAN_BASE_URL,
        })

    if not records:
        return pd.DataFrame(), "api_fields_empty", "La API no devolvió fields útiles."

    return pd.DataFrame(records).drop_duplicates(subset=["dataset", "official_column"]), "api_fields_fallback", None


def get_bronze_parquet_path(dataset: str, bronze_root: Path) -> Path:
    return bronze_root / "sismepre" / dataset / f"{dataset}_raw.parquet"


def read_parquet_schema_and_inference(dataset: str, parquet_path: Path) -> tuple[pd.DataFrame, int]:
    """
    Lee schema técnico Parquet e infiere tipo lógico de apoyo desde valores reales.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe Parquet Bronze: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    row_count = parquet_file.metadata.num_rows

    # Para inferencia necesitamos valores reales.
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


def read_latest_metadata_json(dataset: str, bronze_root: Path) -> dict[str, Any]:
    dataset_dir = bronze_root / "sismepre" / dataset
    json_files = sorted(dataset_dir.rglob("raw_json/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return {}
    try:
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


def compare_dataset(dataset: str, resource_id: str, paths: ValidationPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger.info("Validando dataset SISMEPRE: %s", dataset)

    official_df, dict_source, dict_error = read_official_dictionary(dataset)

    if official_df.empty:
        logger.warning("Diccionario separado no disponible para %s. Intentando fallback API fields.", dataset)
        official_df, dict_source, fallback_error = read_api_fields_as_fallback(dataset, resource_id)
        dict_error = dict_error or fallback_error

    parquet_path = get_bronze_parquet_path(dataset, paths.bronze_root)
    downloaded_df, row_count = read_parquet_schema_and_inference(dataset, parquet_path)
    metadata = read_latest_metadata_json(dataset, paths.bronze_root)

    if official_df.empty:
        # Último respaldo: usar columnas descargadas para no romper el reporte,
        # pero queda marcado claramente como parquet_only.
        official_df = downloaded_df[["dataset", "downloaded_column"]].rename(
            columns={"downloaded_column": "official_column"}
        )
        official_df["official_type"] = ""
        official_df["official_description"] = ""
        official_df["dictionary_source"] = "parquet_only_no_official_dictionary"
        official_df["dictionary_url"] = ""
        dict_source = "parquet_only_no_official_dictionary"

    comparison = official_df.merge(
        downloaded_df,
        left_on=["dataset", "official_column"],
        right_on=["dataset", "downloaded_column"],
        how="outer",
        indicator=True,
    )

    def status_from_merge(value: str) -> str:
        if value == "both":
            return "OK"
        if value == "left_only":
            return "FALTA_EN_PARQUET"
        return "EXTRA_EN_PARQUET"

    comparison["comparison_status"] = comparison["_merge"].map(status_from_merge)
    comparison = comparison.drop(columns=["_merge"])

    missing_count = int((comparison["comparison_status"] == "FALTA_EN_PARQUET").sum())
    extra_count = int((comparison["comparison_status"] == "EXTRA_EN_PARQUET").sum())

    summary = {
        "dataset": dataset,
        "resource_id": resource_id,
        "dictionary_source": dict_source,
        "dictionary_error": dict_error or "",
        "official_columns": int(official_df["official_column"].nunique()),
        "downloaded_columns": int(downloaded_df["downloaded_column"].nunique()),
        "missing_in_parquet": missing_count,
        "extra_in_parquet": extra_count,
        "local_rows_parquet": int(row_count),
        "local_rows_metadata_json": int(metadata.get("rows", -1)) if str(metadata.get("rows", "")).isdigit() else metadata.get("rows", ""),
        "pages_fetched_metadata_json": metadata.get("pages_fetched", ""),
        "parquet_path": str(parquet_path),
        "validation_status": "OK" if missing_count == 0 and extra_count == 0 else "REVISAR",
    }

    logger.info(
        "Resultado %s | status=%s | columnas oficiales=%s | parquet=%s | faltantes=%s | extras=%s | filas=%s",
        dataset,
        summary["validation_status"],
        summary["official_columns"],
        summary["downloaded_columns"],
        missing_count,
        extra_count,
        row_count,
    )

    return comparison, summary


def write_html_report(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    status_counts = detail_df["comparison_status"].value_counts(dropna=False).to_dict()

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <title>Validación de Esquema SISMEPRE</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        h1, h2 {{ color: #1f4e79; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
        th {{ background: #f2f6fa; }}
        .OK {{ background: #e8f5e9; }}
        .REVISAR, .FALTA_EN_PARQUET, .EXTRA_EN_PARQUET {{ background: #fff3e0; }}
        .small {{ color: #666; font-size: 12px; }}
        code {{ background: #f5f5f5; padding: 2px 4px; }}
      </style>
    </head>
    <body>
      <h1>Validación estructural SISMEPRE</h1>
      <p class="small">Generado: {generated_at}</p>
      <p>
        Este reporte compara las columnas documentadas en los diccionarios oficiales o fuentes de respaldo
        contra las columnas descargadas en la capa Bronze Parquet. No modifica datos y no genera Silver.
      </p>

      <h2>Resumen</h2>
      <p><b>Conteo de estados:</b> {status_counts}</p>
      {summary_df.to_html(index=False, escape=False, classes="summary")}

      <h2>Detalle columna por columna</h2>
      {detail_df.to_html(index=False, escape=False, classes="detail")}
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def write_audit(summary_df: pd.DataFrame, detail_df: pd.DataFrame, paths: ValidationPaths, started_at: datetime) -> Path:
    finished_at = datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()

    status = "success" if (summary_df["validation_status"] == "OK").all() else "warning"

    audit_record = {
        "pipeline_name": "schema_validation_sismepre",
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "datasets_evaluated": int(len(summary_df)),
        "datasets_ok": int((summary_df["validation_status"] == "OK").sum()),
        "datasets_review": int((summary_df["validation_status"] != "OK").sum()),
        "columns_evaluated": int(len(detail_df)),
        "missing_columns_total": int((detail_df["comparison_status"] == "FALTA_EN_PARQUET").sum()),
        "extra_columns_total": int((detail_df["comparison_status"] == "EXTRA_EN_PARQUET").sum()),
        "outputs": {
            "summary_csv": str(paths.reports_root / "sismepre_schema_validation_summary.csv"),
            "detail_csv": str(paths.reports_root / "sismepre_schema_validation_detail.csv"),
            "html": str(paths.reports_root / "sismepre_schema_validation.html"),
        },
    }

    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = paths.audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"sismepre_schema_validation_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(audit_record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


def run_sismepre_schema_validation(
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/schema_validation/sismepre"),
    audit_root: Path = Path("data/audit/schema_validation"),
) -> dict[str, str]:
    started_at = datetime.now()
    paths = ValidationPaths(
        bronze_root=Path(bronze_root),
        reports_root=Path(reports_root),
        audit_root=Path(audit_root),
    )

    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.audit_root.mkdir(parents=True, exist_ok=True)

    all_details: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    logger.info("Inicio de validación estructural SISMEPRE.")

    for dataset, resource_id in SISMEPRE_DATASETS.items():
        detail, summary = compare_dataset(dataset, resource_id, paths)
        all_details.append(detail)
        summaries.append(summary)

    detail_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_csv = paths.reports_root / "sismepre_schema_validation_detail.csv"
    summary_csv = paths.reports_root / "sismepre_schema_validation_summary.csv"
    html_report = paths.reports_root / "sismepre_schema_validation.html"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_html_report(detail_df, summary_df, html_report)

    audit_path = write_audit(summary_df, detail_df, paths, started_at)

    logger.info("Validación estructural SISMEPRE completada.")
    logger.info("Reporte HTML: %s", html_report)
    logger.info("Auditoría: %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "html_report": str(html_report),
        "audit_path": str(audit_path),
    }
