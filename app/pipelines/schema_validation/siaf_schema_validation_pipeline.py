"""
siaf_schema_validation_pipeline.py

Orquestador del pipeline de validación estructural para SIAF Ingresos.

Responsabilidades:
    - Leer el contrato de schema desde app/config/schema_rules_siaf.yaml.
    - Descubrir los Parquet Bronze con glob, excluyendo _diario (CRÍTICO).
    - Cargar los Parquets en un único DataFrame PySpark (mergeSchema=true).
    - Delegar la validación a SiafSchemaValidator.
    - Orquestar la generación de reportes CSV, HTML y auditoría JSON.

Regla de ingesta CRÍTICA:
    Se excluyen rutas que contienen "_diario" para evitar duplicar millones
    de registros. Los archivos _diario son acumulativos dentro del año;
    solo los _mensual representan el cierre mensual correcto.
"""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pyspark.sql import DataFrame, SparkSession

from app.schema_validation.siaf_schema_validator import SiafSchemaValidator
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))

# ── Rutas por defecto ─────────────────────────────────────────────────────────
BRONZE_ROOT = Path("data/bronze/siaf")
REPORTS_ROOT = Path("reports/schema_validation/siaf")
AUDIT_ROOT = Path("data/audit/schema_validation")
CONFIG_PATH = Path("app/config/schema_rules_siaf.yaml")


# ── Descubrimiento de Parquets ────────────────────────────────────────────────

def discover_parquet_paths(bronze_root: Path) -> list[str]:
    """
    Descubre todos los Parquet Bronze de SIAF excluyendo _diario.

    REGLA CRÍTICA: los archivos *_diario* son granularidad diaria acumulativa.
    Incluirlos duplicaría los montos presupuestales. Solo se procesan los
    archivos mensual/anuales (sin _diario en el nombre).

    Retorna lista ordenada de rutas absolutas como strings.
    """
    patron = str(bronze_root / "**" / "*.parquet")
    todas = glob.glob(patron, recursive=True)

    # Exclusión explícita: _diario en cualquier parte de la ruta
    filtradas = [r for r in todas if "_diario" not in r.lower()]
    filtradas.sort()

    print(f"[INFO] Parquets encontrados en Bronze SIAF: {len(todas)}")
    print(f"[INFO] Parquets después de excluir _diario: {len(filtradas)}")
    for ruta in filtradas:
        print(f"[INFO]   → {ruta}")

    if not filtradas:
        raise FileNotFoundError(
            f"No se encontraron Parquets Bronze en {bronze_root} "
            f"(excluyendo _diario). Verifica que el Bronze esté descargado."
        )

    return filtradas


# ── Carga en PySpark ──────────────────────────────────────────────────────────

def load_bronze_spark(spark: SparkSession, parquet_paths: list[str]) -> DataFrame:
    """
    Carga todos los Parquets Bronze en un único DataFrame PySpark.

    mergeSchema=True permite combinar años con esquemas ligeramente distintos
    sin errores. Spark hace la lectura distribuida de forma eficiente.
    """
    print(f"[INFO] Cargando {len(parquet_paths)} archivos Parquet en Spark...")
    df = spark.read.option("mergeSchema", "true").parquet(*parquet_paths)
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"[INFO] DataFrame Spark cargado: {total_rows:,} filas × {total_cols} columnas.")
    return df


# ── Carga del YAML ────────────────────────────────────────────────────────────

def load_schema_config(config_path: Path) -> dict[str, Any]:
    """Carga el YAML de reglas de schema. Es la única fuente de verdad."""
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de schema: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("El YAML de schema no tiene una estructura válida.")
    for seccion in ["settings", "expected_columns", "column_types", "critical_columns"]:
        if seccion not in config:
            raise ValueError(f"Falta sección obligatoria en schema_rules_siaf.yaml: '{seccion}'")
    print(f"[INFO] Schema YAML cargado desde: {config_path}")
    return config


# ── Generación de reportes ────────────────────────────────────────────────────

def _css() -> str:
    """CSS base compartido con SISMEPRE — misma paleta y estructura."""
    return """
    body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
    header{background:#1f2430;color:white;padding:24px 32px}
    header h1{margin:0;font-size:25px}
    header p{margin:8px 0 0;color:#cbd5e1}
    main{padding:24px 32px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .card{background:white;border-radius:10px;padding:16px;
          box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #3b82f6}
    .label{color:#6b7280;text-transform:uppercase;font-size:12px;letter-spacing:.05em}
    .value{font-size:28px;font-weight:bold;margin-top:8px;color:#2f7d46}
    .section-title{color:#4b5563;text-transform:uppercase;letter-spacing:.08em;
                   font-size:13px;margin:24px 0 12px}
    table{border-collapse:collapse;width:100%;background:white;border-radius:10px;
          overflow:hidden;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);
          margin-bottom:20px}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;
          text-align:left;vertical-align:middle}
    th{background:#111827;color:white}
    tr:nth-child(even){background:#fafafa}
    .badge{padding:5px 10px;border-radius:999px;font-weight:bold;font-size:12px}
    .ok{background:#d9eadf;color:#245c36}
    .warn{background:#fff3cd;color:#7a5b00}
    .falla{background:#fee2e2;color:#991b1b}
    .note{background:#eef2ff;border-left:4px solid #3b82f6;border-radius:8px;
          padding:14px 16px;margin-bottom:20px;font-size:13px}
    code{background:#eef2ff;padding:2px 5px;border-radius:4px;font-size:12px}
    .alert-card{background:white;border-left:4px solid #c24141;border-radius:8px;
                padding:12px 16px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,.07)}
    """


def _status_badge(status: str) -> str:
    import html as _h
    cls = "ok" if status == "OK" else ("falla" if "CRITICA" in status else "warn")
    return f'<span class="badge {cls}">{_h.escape(status)}</span>'


def write_html_report(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    parquet_paths: list[str],
    total_rows: int,
    output_path: Path,
) -> None:
    """
    Genera el reporte HTML de schema validation para SIAF.
    Mismo estilo visual que SISMEPRE, adaptado al contexto SIAF.
    """
    import html as _h
    generated_at = datetime.now().isoformat(timespec="seconds")

    # ── Métricas resumen ──────────────────────────────────────
    n_ok = int((detail_df["estado"] == "OK").sum())
    n_falta = int((detail_df["estado"] == "FALTA_EN_PARQUET").sum())
    n_extra = int((detail_df["estado"] == "EXTRA_EN_PARQUET").sum())
    n_criticas = int((detail_df["es_critica"] & (detail_df["estado"] == "FALTA_EN_PARQUET")).sum())
    n_esperadas = int(len(detail_df[detail_df["estado"] != "EXTRA_EN_PARQUET"]))
    estado_global = "FALLA_CRITICA" if n_criticas > 0 else ("REVISAR" if n_falta > 0 else "OK")

    # ── Tarjetas ──────────────────────────────────────────────
    cards_html = f"""
    <div class="cards">
      <div class="card">
        <div class="label">Total filas Bronze</div>
        <div class="value">{total_rows:,}</div>
      </div>
      <div class="card">
        <div class="label">Archivos procesados</div>
        <div class="value">{len(parquet_paths)}</div>
      </div>
      <div class="card">
        <div class="label">Columnas OK</div>
        <div class="value">{n_ok}</div>
      </div>
      <div class="card">
        <div class="label">Críticas faltantes</div>
        <div class="value" style="color:#c24141">{n_criticas}</div>
      </div>
    </div>
    """

    # ── Resumen tabular ───────────────────────────────────────
    resumen_html = f"""
    <table>
      <thead><tr>
        <th>Métrica</th><th>Valor</th>
      </tr></thead>
      <tbody>
        <tr><td>Estado global</td><td>{_status_badge(estado_global)}</td></tr>
        <tr><td>Columnas esperadas (YAML)</td><td>{n_esperadas}</td></tr>
        <tr><td>Columnas en Parquet</td><td>{int(summary_df["columnas_en_parquet"].iloc[0]) if not summary_df.empty else "-"}</td></tr>
        <tr><td>Columnas OK</td><td>{n_ok}</td></tr>
        <tr><td>Faltantes en Parquet</td><td>{n_falta}</td></tr>
        <tr><td>Extras en Parquet</td><td>{n_extra}</td></tr>
        <tr><td>Críticas faltantes</td><td><b style="color:#c24141">{n_criticas}</b></td></tr>
        <tr><td>Total filas Bronze (sin _diario)</td><td>{total_rows:,}</td></tr>
      </tbody>
    </table>
    """

    # ── Observaciones ─────────────────────────────────────────
    issues = detail_df[detail_df["estado"] != "OK"]
    if issues.empty:
        obs_html = '<div class="note">✅ No se detectaron columnas faltantes ni extras. El schema es conforme al contrato YAML.</div>'
    else:
        obs_items = "".join([
            f"<div class='alert-card'>"
            f"<b>{_h.escape(str(r['columna']))}</b> — "
            f"Estado: <code>{_h.escape(str(r['estado']))}</code> | "
            f"Crítica: <b>{'SÍ' if r['es_critica'] else 'No'}</b> | "
            f"Tipo esperado: {_h.escape(str(r.get('tipo_logico_esperado', '')))} | "
            f"Tipo Parquet: {_h.escape(str(r.get('tipo_parquet', '-')))}"
            f"</div>"
            for _, r in issues.iterrows()
        ])
        obs_html = obs_items

    # ── Lista de archivos procesados ──────────────────────────
    archivos_html = "".join([
        f"<tr><td><code>{_h.escape(p)}</code></td></tr>"
        for p in parquet_paths
    ])

    doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Schema Validation SIAF</title>
  <style>{_css()}</style>
</head>
<body>
  <header>
    <h1>Validación Estructural — SIAF Ingresos (Bronze)</h1>
    <p>Generado: {generated_at} &nbsp;|&nbsp; Capa evaluada: Bronze
       &nbsp;|&nbsp; Esta etapa NO modifica datos</p>
  </header>
  <main>
    <div class="note">
      Este reporte compara las columnas declaradas en
      <code>app/config/schema_rules_siaf.yaml</code> contra las columnas
      disponibles en los Parquet Bronze de SIAF. Se excluyen archivos
      <code>_diario</code> para evitar duplicar montos presupuestales.
      Las columnas <b>críticas faltantes</b> detienen el pipeline.
    </div>

    {cards_html}

    <h2 class="section-title">Resumen de validación</h2>
    {resumen_html}

    <h2 class="section-title">Observaciones</h2>
    {obs_html}

    <h2 class="section-title">Detalle columna por columna</h2>
    {detail_df.to_html(index=False, escape=True, classes="detail")}

    <h2 class="section-title">Archivos Bronze procesados (sin _diario)</h2>
    <table>
      <thead><tr><th>Ruta Parquet</th></tr></thead>
      <tbody>{archivos_html}</tbody>
    </table>

    <h2 class="section-title">Resumen ejecutivo (CSV)</h2>
    {summary_df.to_html(index=False, escape=True)}
  </main>
</body>
</html>"""

    output_path.write_text(doc, encoding="utf-8")
    print(f"[INFO] Reporte HTML generado: {output_path}")


def write_audit(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    parquet_paths: list[str],
    total_rows: int,
    audit_root: Path,
    reports_root: Path,
    started_at: datetime,
) -> Path:
    """Escribe el JSON de auditoría con la misma estructura que SISMEPRE."""
    finished_at = datetime.now()
    n_criticas = int(
        (detail_df["es_critica"] & (detail_df["estado"] == "FALTA_EN_PARQUET")).sum()
    )
    n_falta = int((detail_df["estado"] == "FALTA_EN_PARQUET").sum())
    status = "critical" if n_criticas > 0 else ("warning" if n_falta > 0 else "success")

    record = {
        "pipeline_name": "schema_validation_siaf",
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "parquets_procesados": len(parquet_paths),
        "parquets_excluidos_diario": "ver log — todos los _diario fueron omitidos",
        "total_rows_bronze": total_rows,
        "columnas_evaluadas": int(len(detail_df)),
        "columnas_ok": int((detail_df["estado"] == "OK").sum()),
        "columnas_faltantes": n_falta,
        "columnas_extras": int((detail_df["estado"] == "EXTRA_EN_PARQUET").sum()),
        "columnas_criticas_faltantes": n_criticas,
        "outputs": {
            "summary_csv": str(reports_root / "siaf_schema_validation_summary.csv"),
            "detail_csv": str(reports_root / "siaf_schema_validation_detail.csv"),
            "html": str(reports_root / "siaf_schema_validation.html"),
        },
    }

    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"siaf_schema_validation_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Auditoría JSON guardada: {audit_path}")
    return audit_path


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_siaf_schema_validation_pipeline(
    spark: SparkSession,
    bronze_root: Path = BRONZE_ROOT,
    reports_root: Path = REPORTS_ROOT,
    audit_root: Path = AUDIT_ROOT,
    config_path: Path = CONFIG_PATH,
) -> dict[str, str]:
    """
    Orquesta la validación estructural completa de SIAF Ingresos.

    Flujo:
        1. Carga schema_rules_siaf.yaml (contrato).
        2. Descubre Parquets Bronze (sin _diario).
        3. Carga en Spark con mergeSchema.
        4. Valida con SiafSchemaValidator (operaciones de conjuntos).
        5. Genera CSV, HTML y auditoría JSON.
        6. Retorna dict con rutas de salida.
    """
    started_at = datetime.now()
    print("[INFO] Inicio pipeline Schema Validation SIAF.")

    reports_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    # 1. YAML
    config = load_schema_config(config_path)

    # 2. Descubrimiento de Parquets (sin _diario)
    parquet_paths = discover_parquet_paths(bronze_root)

    # 3. Carga en Spark
    df_spark = load_bronze_spark(spark, parquet_paths)
    total_rows = df_spark.count()

    # 4. Validación con la clase especializada
    validator = SiafSchemaValidator(config)
    detail_df = validator.validate(df_spark)          # Pandas DataFrame
    summary_df = validator.build_summary(detail_df, total_rows, parquet_paths)

    # 5. Outputs
    detail_csv = reports_root / "siaf_schema_validation_detail.csv"
    summary_csv = reports_root / "siaf_schema_validation_summary.csv"
    html_report = reports_root / "siaf_schema_validation.html"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    write_html_report(detail_df, summary_df, parquet_paths, total_rows, html_report)
    audit_path = write_audit(
        summary_df, detail_df, parquet_paths, total_rows,
        audit_root, reports_root, started_at,
    )

    duration = (datetime.now() - started_at).total_seconds()
    print(f"[INFO] Pipeline Schema Validation SIAF completado en {duration:.1f}s.")

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "html_report": str(html_report),
        "audit_path": str(audit_path),
    }