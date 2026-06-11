"""
siaf_silver_report.py

Genera los reportes HTML del pipeline Silver SIAF.
Mismo patrón que silver_report.py de SISMEPRE, adaptado a SIAF.

Funciones exportadas:
    write_dashboard_html  — dashboard general SIAF Silver
    write_dataset_html    — reporte individual por dataset (ingresos)
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


# ── CSS compartido (mismo estilo que SISMEPRE) ────────────────────────────────

CSS = """
<style>
  * { box-sizing: border-box; }
  body {
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f3f4f6;
    color: #1f2937;
  }
  header {
    background: #1f2430;
    color: white;
    padding: 24px 32px;
  }
  header h1 { margin: 0; font-size: 24px; }
  header p  { margin: 8px 0 0; color: #cbd5e1; font-size: 13px; }
  main { padding: 24px 32px; }

  /* Tarjetas métricas */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }
  .metric {
    background: white;
    border-left: 5px solid #3b82f6;
    border-radius: 8px;
    padding: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,.07);
  }
  .metric.green  { border-left-color: #16a34a; }
  .metric.orange { border-left-color: #d97706; }
  .metric.red    { border-left-color: #dc2626; }
  .metric.purple { border-left-color: #7c3aed; }
  .metric .label {
    color: #6b7280;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .05em;
  }
  .metric .value {
    color: #111827;
    font-size: 26px;
    font-weight: bold;
    margin-top: 6px;
  }

  /* Secciones */
  .card {
    background: white;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }
  h2 {
    color: #374151;
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin: 0 0 14px;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 8px;
  }
  h3 { color: #111827; font-size: 14px; margin: 0 0 10px; }

  /* Tablas */
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    background: white;
  }
  th, td {
    border: 1px solid #e5e7eb;
    padding: 7px 10px;
    vertical-align: top;
    text-align: left;
  }
  th { background: #111827; color: white; }
  tr:nth-child(even) { background: #f9fafb; }

  /* Estados */
  .ok   { color: #166534; font-weight: bold; }
  .warn { color: #92400e; font-weight: bold; }
  .err  { color: #991b1b; font-weight: bold; }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: bold;
  }
  .badge-ok     { background: #d1fae5; color: #065f46; }
  .badge-warn   { background: #fef3c7; color: #78350f; }
  .badge-err    { background: #fee2e2; color: #7f1d1d; }

  /* Notas y glosario */
  .note {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    margin-bottom: 20px;
  }
  code {
    background: #f3f4f6;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 12px;
  }
  .small { font-size: 11px; color: #6b7280; }

  /* Columnas derivadas */
  .derived-tag {
    background: #ede9fe;
    color: #4c1d95;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
  }
  .audit-tag {
    background: #fce7f3;
    color: #831843;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
  }
</style>
"""


# ── Utilidades internas ───────────────────────────────────────────────────────

def _status_class(status: Any) -> str:
    text = str(status)
    if text == "OK":
        return "ok"
    if "ERROR" in text or "FAIL" in text:
        return "err"
    return "warn"


def _status_badge(status: Any) -> str:
    text = str(status)
    if text == "OK":
        return f'<span class="badge badge-ok">{escape(text)}</span>'
    if "ERROR" in text or "FAIL" in text:
        return f'<span class="badge badge-err">{escape(text)}</span>'
    return f'<span class="badge badge-warn">{escape(text)}</span>'


def _origin_tag(origin: Any) -> str:
    text = str(origin)
    if text == "audit_column":
        return f'<span class="audit-tag">técnica</span>'
    if text == "derived_column":
        return f'<span class="derived-tag">derivada</span>'
    return escape(text)


def _df_to_html_table(df: pd.DataFrame, max_rows: int | None = None, tag_cols: list[str] | None = None) -> str:
    """Convierte DataFrame a tabla HTML. tag_cols recibe tratamiento especial."""
    if df is None or df.empty:
        return "<p class='small'>Sin registros.</p>"

    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    tag_cols_set = set(tag_cols or [])
    html = ["<table>", "<thead><tr>"]
    for col in view.columns:
        html.append(f"<th>{escape(str(col))}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in view.iterrows():
        html.append("<tr>")
        for col in view.columns:
            raw = row[col]
            value = "" if pd.isna(raw) else str(raw)
            if col == "status":
                html.append(f"<td class='{_status_class(value)}'>{escape(value)}</td>")
            elif col == "origin":
                html.append(f"<td>{_origin_tag(value)}</td>")
            elif col in tag_cols_set:
                html.append(f"<td>{escape(value)}</td>")
            else:
                html.append(f"<td>{escape(value)}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    if max_rows is not None and len(df) > max_rows:
        html.append(f"<p class='small'>Mostrando {max_rows} de {len(df)} filas.</p>")
    return "\n".join(html)


def _pipeline_steps_html(summary_row: pd.Series) -> str:
    """Genera tabla resumen de los pasos del pipeline con sus métricas."""
    bronze = int(summary_row.get("bronze_rows", 0))
    filtered = int(summary_row.get("filtered_rows", 0))
    silver = int(summary_row.get("silver_rows", 0))
    dupes = int(summary_row.get("duplicates_removed", 0))
    filtered_out = bronze - filtered

    steps = [
        ("1", "Lectura Bronze", f"{bronze:,} filas", "Todos los Parquet SIAF (sin _diario) combinados en Spark.", "ok"),
        ("2", "Filtro NIVEL_GOBIERNO=M", f"{filtered:,} filas ({filtered_out:,} excluidas)", "Solo municipalidades. Niveles E y R se conservan en Bronze.", "ok" if filtered_out >= 0 else "warn"),
        ("3", "Cleaning", "null_tokens → null, trim, lpad en códigos", "Normalización de tokens vacíos y espacios laterales.", "ok"),
        ("4", "Imputaciones", "SECTOR, PLIEGO → '00' / 'SIN DATO'", "Rellena vacíos estructurales post-filtro.", "ok"),
        ("5", "Conversión de tipos", f"{int(summary_row.get('columns_converted', 0))} columnas", "Enteros, decimales y strings según output_schema.", "ok" if int(summary_row.get("conversion_errors", 0)) == 0 else "warn"),
        ("6", "Columnas derivadas", "7 columnas nuevas", "ubigeo, nombre_norm, sk_tiempo, anio_mes, ano_particion, flags.", "ok"),
        ("7", "Columnas técnicas", "5 columnas", "source_system, source_dataset, silver_processed_at, _audit_loaded_at, record_hash.", "ok"),
        ("8", "Deduplicación", f"{dupes:,} filas eliminadas", "row_number() por clave de negocio, keep_first.", "ok" if dupes == 0 else "warn"),
        ("9", "Escritura Silver", f"{silver:,} filas finales", "Parquet particionado por ano_particion.", "ok"),
    ]

    rows_html = "".join([
        f"<tr>"
        f"<td><b>{escape(s[0])}</b></td>"
        f"<td>{escape(s[1])}</td>"
        f"<td>{escape(s[2])}</td>"
        f"<td class='{s[4]}'>{escape(s[3])}</td>"
        f"</tr>"
        for s in steps
    ])

    return f"""
    <table>
      <thead><tr>
        <th>#</th><th>Paso</th><th>Resultado</th><th>Descripción</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


def _glossary_html() -> str:
    """Glosario de columnas técnicas y derivadas SIAF."""
    items = [
        ("source_system", "técnica", "Sistema de origen. Valor fijo: 'siaf'."),
        ("source_dataset", "técnica", "Dataset Bronze de origen (ej: ingresos)."),
        ("silver_processed_at", "técnica", "Timestamp ISO de procesamiento hacia Silver."),
        ("_audit_loaded_at", "técnica", "Timestamp de registro en auditoría. Permite rastrear re-ejecuciones."),
        ("record_hash", "técnica", "SHA-256 de las 36 columnas originales Bronze. Detecta cambios."),
        ("ubigeo_ejecutora", "derivada", "lpad(concat(DPTO, PROV, DIST), 6, '0'). Llave para join con RENAMU y SISMEPRE en Gold."),
        ("nombre_municipalidad_normalizado", "derivada", "upper(trim(regexp_replace(EJECUTORA_NOMBRE, '\\s+', ' '))). Para fuzzy match con CategoriasMunicipalidades.csv en Gold."),
        ("sk_tiempo_mensual", "derivada", "ANO_DOC * 100 + MES_DOC. Clave surrogate temporal para dimensión tiempo en Gold."),
        ("anio_mes_key", "derivada", "Formato YYYY-MM (ej: 2026-01). Filtros y joins en Gold y Power BI."),
        ("ano_particion", "derivada", "ANO_DOC como entero. Columna de particionado del Parquet Silver."),
        ("flag_tipo_transaccion", "derivada", "NORMAL si MONTO_RECAUDADO >= 0, REVERSION si < 0."),
        ("flag_monto_pia_activo", "derivada", "Boolean: True si MONTO_PIA > 0."),
    ]

    rows_html = "".join([
        f"<tr>"
        f"<td><code>{escape(it[0])}</code></td>"
        f"<td>{_origin_tag(it[1] + '_column')}</td>"
        f"<td>{escape(it[2])}</td>"
        f"</tr>"
        for it in items
    ])

    return f"""
    <table>
      <thead><tr>
        <th>Columna</th><th>Tipo</th><th>Descripción</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


# ── Dashboard principal ───────────────────────────────────────────────────────

def write_dashboard_html(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera el dashboard HTML principal del pipeline Silver SIAF.
    Mismo estilo visual que SISMEPRE, contenido adaptado a SIAF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    # ── Métricas principales ──────────────────────────────────
    bronze_rows  = int(summary_df["bronze_rows"].sum())         if not summary_df.empty else 0
    filtered_rows= int(summary_df["filtered_rows"].sum())       if not summary_df.empty else 0
    silver_rows  = int(summary_df["silver_rows"].sum())         if not summary_df.empty else 0
    dupes        = int(summary_df["duplicates_removed"].sum())  if not summary_df.empty else 0
    conv_errors  = int(summary_df["conversion_errors"].sum())   if not summary_df.empty else 0
    derived      = int(summary_df["derived_columns_added"].sum()) if "derived_columns_added" in summary_df.columns else 7
    silver_cols  = int(summary_df["silver_columns"].sum())      if not summary_df.empty else 0

    status_global = "OK" if conv_errors == 0 else "WITH_CONVERSION_ERRORS"
    pct_municipal = round(filtered_rows / bronze_rows * 100, 1) if bronze_rows > 0 else 0

    # ── HTML ──────────────────────────────────────────────────
    summary_row_for_steps = summary_df.iloc[0] if not summary_df.empty else pd.Series({})

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dashboard Silver SIAF</title>
  {CSS}
</head>
<body>
  <header>
    <h1>Dashboard Silver — SIAF Ingresos</h1>
    <p>
      Generado: {generated_at} &nbsp;|&nbsp;
      Capa: Bronze → Silver &nbsp;|&nbsp;
      Motor: PySpark &nbsp;|&nbsp;
      Estado: {_status_badge(status_global)}
    </p>
  </header>
  <main>

    <div class="note">
      La capa Silver aplica el filtro <code>NIVEL_GOBIERNO = 'M'</code> (municipalidades),
      estandariza tipos, normaliza vacíos, genera columnas derivadas para Gold
      (<code>ubigeo_ejecutora</code>, <code>nombre_municipalidad_normalizado</code>)
      y agrega trazabilidad técnica por registro.
      Bronze permanece intacto.
    </div>

    <!-- Tarjetas de métricas -->
    <div class="grid">
      <div class="metric">
        <div class="label">Filas Bronze (entrada)</div>
        <div class="value">{bronze_rows:,}</div>
      </div>
      <div class="metric green">
        <div class="label">Filas Silver (salida)</div>
        <div class="value">{silver_rows:,}</div>
      </div>
      <div class="metric purple">
        <div class="label">% Municipalidades</div>
        <div class="value">{pct_municipal}%</div>
      </div>
      <div class="metric orange">
        <div class="label">Columnas Silver</div>
        <div class="value">{silver_cols}</div>
      </div>
      <div class="metric purple">
        <div class="label">Cols. derivadas</div>
        <div class="value">{derived}</div>
      </div>
      <div class="metric {'red' if conv_errors > 0 else 'green'}">
        <div class="label">Errores de conv.</div>
        <div class="value">{conv_errors:,}</div>
      </div>
      <div class="metric {'orange' if dupes > 0 else 'green'}">
        <div class="label">Duplicados elim.</div>
        <div class="value">{dupes:,}</div>
      </div>
    </div>

    <!-- Resumen ejecutivo -->
    <div class="card">
      <h2>Resumen ejecutivo</h2>
      <p>
        El pipeline Silver SIAF procesó <b>{bronze_rows:,}</b> registros históricos
        (años 2012–2026, excluyendo archivos <code>_diario</code>).
        Tras el filtro <code>NIVEL_GOBIERNO = 'M'</code> quedaron <b>{filtered_rows:,}</b>
        registros municipales (<b>{pct_municipal}%</b> del total).
        El Parquet Silver final contiene <b>{silver_rows:,}</b> filas particionadas por
        <code>ano_particion</code>, con <b>{silver_cols}</b> columnas
        (incluyendo 7 derivadas y 5 técnicas de auditoría).
      </p>
    </div>

    <!-- Pasos del pipeline -->
    <div class="card">
      <h2>Pasos del pipeline</h2>
      {_pipeline_steps_html(summary_row_for_steps)}
    </div>

    <!-- Resumen por dataset -->
    <div class="card">
      <h2>Resumen por dataset</h2>
      {_df_to_html_table(summary_df)}
    </div>

    <!-- Glosario de columnas nuevas -->
    <div class="card">
      <h2>Columnas nuevas en Silver</h2>
      {_glossary_html()}
    </div>

    <!-- Detalle por columna -->
    <div class="card">
      <h2>Detalle de transformaciones por columna</h2>
      {_df_to_html_table(detail_df, max_rows=300)}
    </div>

    <!-- Diccionario Silver -->
    <div class="card">
      <h2>Diccionario Silver (todas las columnas)</h2>
      {_df_to_html_table(dictionary_df, max_rows=300)}
    </div>

  </main>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ── Reporte por dataset ───────────────────────────────────────────────────────

def write_dataset_html(
    dataset: str,
    summary_row: pd.Series,
    detail_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera el reporte HTML individual por dataset SIAF.
    Para SIAF solo hay un dataset: 'ingresos'.
    Mismo patrón que write_dataset_html de SISMEPRE.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")

    summary_table = pd.DataFrame([summary_row.to_dict()])
    dataset_dictionary = dictionary_df[dictionary_df["dataset"] == dataset].copy()

    bronze_rows = int(summary_row.get("bronze_rows", 0))
    silver_rows = int(summary_row.get("silver_rows", 0))
    status = str(summary_row.get("status", "OK"))

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Silver SIAF — {escape(dataset)}</title>
  {CSS}
</head>
<body>
  <header>
    <h1>Reporte Silver — SIAF / {escape(dataset)}</h1>
    <p>
      Generado: {generated_at} &nbsp;|&nbsp;
      Bronze: {bronze_rows:,} filas &nbsp;→&nbsp;
      Silver: {silver_rows:,} filas &nbsp;|&nbsp;
      Estado: {_status_badge(status)}
    </p>
  </header>
  <main>

    <div class="note">
      Este reporte muestra las transformaciones aplicadas al dataset
      <code>SIAF/{escape(dataset)}</code> en la capa Silver.
      Los datos Bronze permanecen intactos. El Parquet Silver está
      particionado por <code>ano_particion</code>.
    </div>

    <!-- Métricas del dataset -->
    <div class="grid">
      <div class="metric">
        <div class="label">Filas Bronze</div>
        <div class="value">{bronze_rows:,}</div>
      </div>
      <div class="metric green">
        <div class="label">Filas Silver</div>
        <div class="value">{silver_rows:,}</div>
      </div>
      <div class="metric orange">
        <div class="label">Duplicados eliminados</div>
        <div class="value">{int(summary_row.get('duplicates_removed', 0)):,}</div>
      </div>
      <div class="metric {'red' if int(summary_row.get('conversion_errors', 0)) > 0 else 'green'}">
        <div class="label">Errores de conversión</div>
        <div class="value">{int(summary_row.get('conversion_errors', 0)):,}</div>
      </div>
    </div>

    <!-- Resumen -->
    <div class="card">
      <h2>Resumen del dataset</h2>
      {_df_to_html_table(summary_table)}
    </div>

    <!-- Pasos -->
    <div class="card">
      <h2>Pasos del pipeline</h2>
      {_pipeline_steps_html(summary_row)}
    </div>

    <!-- Columnas nuevas -->
    <div class="card">
      <h2>Columnas nuevas en Silver</h2>
      {_glossary_html()}
    </div>

    <!-- Detalle por columna -->
    <div class="card">
      <h2>Detalle de transformaciones</h2>
      {_df_to_html_table(detail_df)}
    </div>

    <!-- Diccionario -->
    <div class="card">
      <h2>Diccionario Silver</h2>
      {_df_to_html_table(dataset_dictionary)}
    </div>

  </main>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")