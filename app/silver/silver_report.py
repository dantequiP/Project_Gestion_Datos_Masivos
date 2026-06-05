from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


CSS = """
<style>
  body { font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f9fafb; }
  h1, h2, h3 { color: #111827; }
  .card { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin: 14px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
  .metric { background: #ffffff; border-left: 5px solid #64748b; border-radius: 8px; padding: 12px; }
  .metric .label { color: #64748b; font-size: 12px; text-transform: uppercase; }
  .metric .value { color: #111827; font-size: 24px; font-weight: bold; margin-top: 4px; }
  table { border-collapse: collapse; width: 100%; background: white; font-size: 13px; }
  th, td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }
  th { background: #f3f4f6; text-align: left; }
  tr:nth-child(even) { background: #f9fafb; }
  .ok { color: #166534; font-weight: bold; }
  .warn { color: #92400e; font-weight: bold; }
  .err { color: #991b1b; font-weight: bold; }
  .small { font-size: 12px; color: #6b7280; }
  code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
</style>
"""


def _status_class(status: Any) -> str:
    text = str(status)
    if text == "OK":
        return "ok"
    if "ERROR" in text:
        return "err"
    return "warn"


def _df_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convierte un DataFrame en tabla HTML sencilla y segura."""
    if df.empty:
        return "<p>No hay registros para mostrar.</p>"

    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    html = ["<table>", "<thead><tr>"]
    for column in view.columns:
        html.append(f"<th>{escape(str(column))}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in view.iterrows():
        html.append("<tr>")
        for column in view.columns:
            value = "" if pd.isna(row[column]) else str(row[column])
            if column == "status":
                css_class = _status_class(value)
                html.append(f"<td class='{css_class}'>{escape(value)}</td>")
            else:
                html.append(f"<td>{escape(value)}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    if max_rows is not None and len(df) > max_rows:
        html.append(f"<p class='small'>Mostrando {max_rows} de {len(df)} filas.</p>")
    return "\n".join(html)


def write_dashboard_html(summary_df: pd.DataFrame, detail_df: pd.DataFrame, dictionary_df: pd.DataFrame, output_path: Path) -> None:
    """Genera dashboard general de transformación Silver."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_datasets = int(len(summary_df))
    total_bronze_rows = int(summary_df["bronze_rows"].sum()) if not summary_df.empty else 0
    total_silver_rows = int(summary_df["silver_rows"].sum()) if not summary_df.empty else 0
    total_errors = int(summary_df["conversion_errors"].sum()) if not summary_df.empty else 0
    total_dropped = int(summary_df["columns_dropped"].sum()) if not summary_df.empty else 0

    glossary = pd.DataFrame(
        [
            {"concepto": "source_system", "explicacion": "Sistema de origen de los datos. Para este proceso: SISMEPRE."},
            {"concepto": "source_dataset", "explicacion": "Dataset Bronze del cual proviene cada registro."},
            {"concepto": "silver_processed_at", "explicacion": "Fecha y hora en que se procesó el registro hacia Silver."},
            {"concepto": "record_hash", "explicacion": "Huella SHA-256 para trazabilidad y detección de cambios."},
            {"concepto": "anio_periodo_key", "explicacion": "Clave derivada de ANO_APLICACION y PERIODO, útil para integración temporal."},
            {"concepto": "0 preservado", "explicacion": "El valor 0 se mantiene cuando puede representar una respuesta, monto o cantidad válida."},
            {"concepto": "null", "explicacion": "Representa ausencia real de dato después de normalizar vacíos."},
        ]
    )

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Dashboard Silver SISMEPRE</title>
{CSS}
</head>
<body>
<h1>Dashboard Silver — SISMEPRE</h1>
<p class="small">Reporte generado automáticamente por el pipeline Silver.</p>

<div class="grid">
  <div class="metric"><div class="label">Datasets procesados</div><div class="value">{total_datasets}</div></div>
  <div class="metric"><div class="label">Filas Bronze</div><div class="value">{total_bronze_rows:,}</div></div>
  <div class="metric"><div class="label">Filas Silver</div><div class="value">{total_silver_rows:,}</div></div>
  <div class="metric"><div class="label">Errores de conversión</div><div class="value">{total_errors:,}</div></div>
  <div class="metric"><div class="label">Columnas eliminadas</div><div class="value">{total_dropped:,}</div></div>
</div>

<div class="card">
<h2>Resumen ejecutivo</h2>
<p>La capa Silver estandariza tipos de datos, normaliza vacíos reales a <code>null</code>, conserva valores cero cuando representan datos válidos, agrega columnas técnicas de trazabilidad y aplica reglas específicas documentadas por dataset.</p>
</div>

<div class="card">
<h2>Resumen por dataset</h2>
{_df_to_html_table(summary_df)}
</div>

<div class="card">
<h2>Glosario técnico</h2>
{_df_to_html_table(glossary)}
</div>

<div class="card">
<h2>Detalle de transformaciones por columna</h2>
{_df_to_html_table(detail_df, max_rows=250)}
</div>

<div class="card">
<h2>Diccionario Silver</h2>
{_df_to_html_table(dictionary_df, max_rows=250)}
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_dataset_html(dataset: str, summary_row: pd.Series, detail_df: pd.DataFrame, dictionary_df: pd.DataFrame, output_path: Path) -> None:
    """Genera reporte HTML individual por dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_table = pd.DataFrame([summary_row.to_dict()])
    dataset_dictionary = dictionary_df[dictionary_df["dataset"] == dataset].copy()

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Silver — {escape(dataset)}</title>
{CSS}
</head>
<body>
<h1>Reporte Silver — {escape(dataset)}</h1>
<p class="small">Transformaciones aplicadas desde Bronze hacia Silver.</p>

<div class="card">
<h2>Resumen del dataset</h2>
{_df_to_html_table(summary_table)}
</div>

<div class="card">
<h2>Detalle por columna</h2>
{_df_to_html_table(detail_df)}
</div>

<div class="card">
<h2>Diccionario Silver del dataset</h2>
{_df_to_html_table(dataset_dictionary)}
</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
