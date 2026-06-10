"""
renamu_schema_validator.py

Verificación estructural de RENAMU 2022:
- Compara columnas descargadas en Bronze (Parquet) contra el diccionario local CSV.
- Revisa tipo técnico Parquet y tipo lógico inferido desde valores reales.
- Genera reportes CSV y HTML en reports/schema_validation/renamu/.
- Registra auditoría JSON en data/audit/schema_validation/.

RENAMU difiere de SISMEPRE en tres puntos clave:
  1. Un único dataset: municipalidades_2022.
  2. Diccionario local CSV (no URL ni API), columna 'nombre_campo'.
  3. El Parquet Bronze está dentro de una partición de fecha y con timestamp
     en el nombre — se localiza con rglob.

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

from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))

# Nombre canónico del dataset RENAMU dentro de la arquitectura.
RENAMU_DATASET = "municipalidades_2022"

# Carpeta Bronze dentro de data/bronze/renamu/ que contiene el Parquet.
# Coincide con el nombre generado por la ingesta.
RENAMU_BRONZE_SUBFOLDER = "2022_municipalidades__Base_RENAMU_2022_f"

# Ruta del diccionario CSV relativa a la raíz del proyecto.
RENAMU_DICTIONARY_PATH = Path("app/config/renamu_2022_diccionario.csv")


# ---------------------------------------------------------------------------
# Helpers compartidos con SISMEPRE (duplicados aquí para independencia)
# ---------------------------------------------------------------------------

def normalize_column_name(value: Any) -> str:
    """Normaliza nombres de columnas: elimina BOM, strip, espacios → '_', uppercase."""
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    text = re.sub(r"\s+", "_", text)
    return text.upper()


def is_empty_value(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "nan", "NaN", "None", "null", "NULL"}


def infer_logical_type(series: pd.Series, sample_size: int = 10_000) -> str:
    """
    Infiere tipo lógico de apoyo desde valores reales.
    No reemplaza al diccionario oficial; sirve como contexto para profiling/quality/Silver.

    Nota RENAMU: los vacíos son strings '' o ' ', no NaN reales.
    Se filtran antes de clasificar.
    """
    s = series.dropna().astype(str).map(str.strip)
    s = s[s != ""]  # vacíos RENAMU son string vacío

    if s.empty:
        return "empty_or_null"

    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)

    upper_values = set(s.str.upper().unique())
    if upper_values and upper_values.issubset({"0", "1", "S", "N", "SI", "NO", "TRUE", "FALSE"}):
        return "categorical_indicator"

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio >= 0.98:
        non_null_numeric = numeric.dropna()
        if (non_null_numeric % 1 == 0).all():
            return "integer"
        return "decimal"

    date_formats = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    best_ratio = 0.0
    for fmt in date_formats:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        best_ratio = max(best_ratio, parsed.notna().mean())
    if best_ratio >= 0.90:
        return "datetime_or_date"

    email_ratio = s.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False).mean()
    if email_ratio >= 0.80:
        return "email"

    unique_ratio = s.nunique(dropna=True) / len(s)
    if unique_ratio <= 0.10:
        return "categorical_text"

    return "text"


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@dataclass
class ValidationPaths:
    bronze_root: Path = Path("data/bronze")
    reports_root: Path = Path("reports/schema_validation/renamu")
    audit_root: Path = Path("data/audit/schema_validation")


# ---------------------------------------------------------------------------
# Lectura del diccionario local RENAMU
# ---------------------------------------------------------------------------

def read_renamu_dictionary(
    dictionary_path: Path = RENAMU_DICTIONARY_PATH,
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Lee el diccionario local CSV del RENAMU 2022.

    Estructura esperada del CSV (separador ';'):
      - nombre_campo       : nombre de la columna tal como aparece en el CSV original
      - descripcion        : descripción de la variable
      - modulo             : módulo del cuestionario
      - valores_permitidos : valores codificados (vacío cuando no aplica)
      - numero_pregunta    : número de pregunta en el cuestionario
      - tema_pregunta      : tema de la pregunta
      - pagina_pdf         : página en el diccionario PDF
      - estado_extraccion  : estado de la extracción del diccionario (siempre 'ok')
      - asignacion_valores : 'sin_valores' | 'asignado_por_posicion'

    Retorna:
      - DataFrame normalizado con official_column, official_description, official_module,
        official_allowed_values, dictionary_source
      - fuente usada
      - error si hubo
    """
    if not dictionary_path.exists():
        return pd.DataFrame(), "local_csv_not_found", f"No existe: {dictionary_path}"

    try:
        df = pd.read_csv(dictionary_path, sep=";", dtype=str, encoding="utf-8")
    except Exception as exc:
        return pd.DataFrame(), "local_csv_error", str(exc)

    if df.empty or "nombre_campo" not in df.columns:
        return pd.DataFrame(), "local_csv_bad_format", (
            f"El CSV no tiene columna 'nombre_campo'. Columnas encontradas: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "dataset": RENAMU_DATASET,
        "official_column": df["nombre_campo"].map(normalize_column_name),
        "official_description": df["descripcion"].fillna("").astype(str).str.strip(),
        "official_module": df["modulo"].fillna("").astype(str).str.strip(),
        "official_allowed_values": df["valores_permitidos"].fillna("").astype(str).str.strip(),
        "dictionary_source": "local_csv",
        "dictionary_path": str(dictionary_path),
    })

    # Eliminar filas sin nombre de columna válido
    out = out[out["official_column"] != ""].drop_duplicates(subset=["dataset", "official_column"])

    return out, "local_csv", None


# ---------------------------------------------------------------------------
# Localización del Parquet Bronze
# ---------------------------------------------------------------------------

def find_renamu_parquet(bronze_root: Path) -> Path:
    """
    Localiza el Parquet Bronze más reciente del dataset municipalidades_2022.

    El path exacto generado por la ingesta tiene esta estructura:
      data/bronze/renamu/<RENAMU_BRONZE_SUBFOLDER>/year=YYYY/month=MM/day=DD/<nombre_timestamp>.parquet

    Se usa rglob para no depender del timestamp ni de la partición de fecha.
    """
    search_root = bronze_root / "renamu" / RENAMU_BRONZE_SUBFOLDER
    candidates = sorted(
        search_root.rglob("*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ningún Parquet Bronze en: {search_root}\n"
            f"Verifica que la ingesta haya generado el archivo correctamente."
        )
    if len(candidates) > 1:
        logger.warning(
            "Se encontraron %d Parquet en %s — se usa el más reciente: %s",
            len(candidates), search_root, candidates[0],
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Lectura de schema e inferencia de tipo lógico
# ---------------------------------------------------------------------------

def read_parquet_schema_and_inference(parquet_path: Path) -> tuple[pd.DataFrame, int]:
    """
    Lee el schema técnico del Parquet e infiere tipo lógico desde valores reales.

    Nota: el Parquet RENAMU almacena todo como string (dtype=str en la ingesta),
    por eso parquet_type será mayoritariamente 'string[pyarrow]' o 'large_string'.
    El tipo lógico inferido es el valor añadido real de esta función.
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
            "dataset": RENAMU_DATASET,
            "downloaded_column": norm_col,
            "original_downloaded_column": col_name,
            "parquet_type": str(field.type),
            "nullable": field.nullable,
            "inferred_type": inferred_type,
            "local_rows_parquet": row_count,
            "parquet_path": str(parquet_path),
        })

    return pd.DataFrame(records), row_count


# ---------------------------------------------------------------------------
# Comparación diccionario vs Parquet
# ---------------------------------------------------------------------------

def compare_dataset(paths: ValidationPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compara las columnas del diccionario oficial RENAMU contra las del Parquet Bronze.

    Notas sobre el resultado esperado:
    - El diccionario tiene 1,369 entradas; el Parquet tiene ~1,368 columnas.
      Una diferencia de 1 puede deberse a una entrada sin columna directa en el dataset
      (p.ej. la entrada base 'VFI' cuando el dataset tiene solo variantes 'VFI_P*').
    - Columnas extra en Parquet que no están en el diccionario quedan como EXTRA_EN_PARQUET.
    - Columnas del diccionario ausentes en Parquet quedan como FALTA_EN_PARQUET.
    """
    logger.info("Validando dataset RENAMU: %s", RENAMU_DATASET)

    official_df, dict_source, dict_error = read_renamu_dictionary()

    if official_df.empty:
        logger.error("No se pudo leer el diccionario RENAMU: %s", dict_error)
        # Continúa sin diccionario para no romper el reporte completo.
        official_df = pd.DataFrame(columns=[
            "dataset", "official_column", "official_description",
            "official_module", "official_allowed_values",
            "dictionary_source", "dictionary_path",
        ])

    parquet_path = find_renamu_parquet(paths.bronze_root)
    downloaded_df, row_count = read_parquet_schema_and_inference(parquet_path)

    if official_df.empty:
        # Sin diccionario: marca todo como parquet_only y lo registra.
        official_df = downloaded_df[["dataset", "downloaded_column"]].rename(
            columns={"downloaded_column": "official_column"}
        )
        official_df["official_description"] = ""
        official_df["official_module"] = ""
        official_df["official_allowed_values"] = ""
        official_df["dictionary_source"] = "parquet_only_no_official_dictionary"
        official_df["dictionary_path"] = ""
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
    ok_count = int((comparison["comparison_status"] == "OK").sum())

    summary: dict[str, Any] = {
        "dataset": RENAMU_DATASET,
        "dictionary_source": dict_source,
        "dictionary_error": dict_error or "",
        "official_columns": int(official_df["official_column"].nunique()),
        "downloaded_columns": int(downloaded_df["downloaded_column"].nunique()),
        "matched_columns": ok_count,
        "missing_in_parquet": missing_count,
        "extra_in_parquet": extra_count,
        "local_rows_parquet": int(row_count),
        "parquet_path": str(parquet_path),
        "validation_status": "OK" if missing_count == 0 and extra_count == 0 else "REVISAR",
    }

    logger.info(
        "Resultado %s | status=%s | columnas_dict=%s | columnas_parquet=%s | "
        "coinciden=%s | faltantes=%s | extras=%s | filas=%s",
        RENAMU_DATASET,
        summary["validation_status"],
        summary["official_columns"],
        summary["downloaded_columns"],
        ok_count,
        missing_count,
        extra_count,
        row_count,
    )

    return comparison, summary


# ---------------------------------------------------------------------------
# Reporte HTML
# ---------------------------------------------------------------------------

def write_html_report(
    detail_df: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    status_counts = detail_df["comparison_status"].value_counts(dropna=False).to_dict()
    status_color = "#e8f5e9" if summary["validation_status"] == "OK" else "#fff3e0"

    summary_df = pd.DataFrame([summary])

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <title>Validación de Esquema RENAMU 2022</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        h1, h2 {{ color: #1f4e79; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; vertical-align: top; }}
        th {{ background: #f2f6fa; }}
        tr.OK td {{ background: #e8f5e9; }}
        tr.FALTA_EN_PARQUET td, tr.EXTRA_EN_PARQUET td {{ background: #fff3e0; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                  font-size: 12px; font-weight: bold; }}
        .badge-ok {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-warn {{ background: #fff3e0; color: #e65100; }}
        .small {{ color: #666; font-size: 12px; }}
        code {{ background: #f5f5f5; padding: 2px 4px; font-size: 11px; }}
      </style>
    </head>
    <body>
      <h1>Validación estructural — RENAMU 2022</h1>
      <p class="small">Generado: {generated_at}</p>
      <p>
        Compara las columnas del diccionario oficial RENAMU 2022 (CSV local)
        contra las columnas descargadas en la capa Bronze Parquet.
        No modifica datos y no genera Silver.
      </p>

      <h2>Resumen del dataset</h2>
      <p>
        Estado general:
        <span class="badge {'badge-ok' if summary['validation_status'] == 'OK' else 'badge-warn'}">
          {summary['validation_status']}
        </span>
      </p>
      <p><b>Conteo de estados por columna:</b> {status_counts}</p>
      {summary_df.to_html(index=False, escape=False)}

      <h2>Detalle columna por columna</h2>
      <p class="small">
        FALTA_EN_PARQUET: columna documentada en el diccionario pero ausente en el Parquet.<br>
        EXTRA_EN_PARQUET: columna presente en el Parquet pero no documentada en el diccionario.<br>
        OK: columna coincide entre diccionario y Parquet.
      </p>
    """

    # Tabla con clases CSS por estado para colorear filas
    rows_html = ""
    for _, row in detail_df.iterrows():
        status = row.get("comparison_status", "")
        css_class = status if status in {"OK", "FALTA_EN_PARQUET", "EXTRA_EN_PARQUET"} else ""
        cells = "".join(
            f"<td>{'' if pd.isna(v) else v}</td>"
            for v in row
        )
        rows_html += f'<tr class="{css_class}">{cells}</tr>\n'

    header_html = "".join(f"<th>{c}</th>" for c in detail_df.columns)
    html += f"""
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Auditoría JSON
# ---------------------------------------------------------------------------

def write_audit(
    summary: dict[str, Any],
    detail_df: pd.DataFrame,
    paths: ValidationPaths,
    started_at: datetime,
) -> Path:
    finished_at = datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()

    audit_record = {
        "pipeline_name": "schema_validation_renamu",
        "status": "success" if summary["validation_status"] == "OK" else "warning",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "datasets_evaluated": 1,
        "datasets_ok": 1 if summary["validation_status"] == "OK" else 0,
        "datasets_review": 0 if summary["validation_status"] == "OK" else 1,
        "columns_evaluated": int(len(detail_df)),
        "missing_columns_total": int((detail_df["comparison_status"] == "FALTA_EN_PARQUET").sum()),
        "extra_columns_total": int((detail_df["comparison_status"] == "EXTRA_EN_PARQUET").sum()),
        "matched_columns_total": int((detail_df["comparison_status"] == "OK").sum()),
        "local_rows_parquet": summary.get("local_rows_parquet", -1),
        "outputs": {
            "summary_csv": str(paths.reports_root / "renamu_schema_validation_summary.csv"),
            "detail_csv": str(paths.reports_root / "renamu_schema_validation_detail.csv"),
            "html": str(paths.reports_root / "renamu_schema_validation.html"),
        },
    }

    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = paths.audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"renamu_schema_validation_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(audit_record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


# ---------------------------------------------------------------------------
# Función de entrada pública
# ---------------------------------------------------------------------------

def run_renamu_schema_validation(
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/schema_validation/renamu"),
    audit_root: Path = Path("data/audit/schema_validation"),
) -> dict[str, str]:
    """
    Ejecuta la validación estructural del RENAMU 2022.

    Flujo:
    1. Lee el diccionario local CSV.
    2. Localiza el Parquet Bronze más reciente.
    3. Compara columnas diccionario vs Parquet.
    4. Genera CSV (summary + detail) y HTML.
    5. Registra auditoría JSON.
    6. No modifica ningún archivo Bronze.
    """
    started_at = datetime.now()
    paths = ValidationPaths(
        bronze_root=Path(bronze_root),
        reports_root=Path(reports_root),
        audit_root=Path(audit_root),
    )

    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.audit_root.mkdir(parents=True, exist_ok=True)

    logger.info("Inicio de validación estructural RENAMU.")

    detail_df, summary = compare_dataset(paths)

    detail_csv = paths.reports_root / "renamu_schema_validation_detail.csv"
    summary_csv = paths.reports_root / "renamu_schema_validation_summary.csv"
    html_report = paths.reports_root / "renamu_schema_validation.html"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_html_report(detail_df, summary, html_report)

    audit_path = write_audit(summary, detail_df, paths, started_at)

    logger.info("Validación estructural RENAMU completada.")
    logger.info("Reporte HTML : %s", html_report)
    logger.info("Auditoría   : %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "html_report": str(html_report),
        "audit_path": str(audit_path),
    }