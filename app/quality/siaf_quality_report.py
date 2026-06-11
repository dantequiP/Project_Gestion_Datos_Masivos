"""
siaf_quality_report.py

Generación de reportes HTML de calidad para SIAF Ingresos.

Reutiliza las funciones de quality_report.py de SISMEPRE para:
- build_summary()  → resumen por dimensión
- severity_summary()
- status_badge(), _bar(), _score_value()

Añade específicamente para SIAF:
- write_siaf_html()  → reporte único (SIAF es un solo dataset, no 7 como SISMEPRE)
- write_siaf_dashboard_html()  → dashboard con contexto de hallazgos del YAML
- Las notas explicativas de los hallazgos documentados (montos negativos, PIA=0, etc.)
"""

from __future__ import annotations

import html as _h
from pathlib import Path

import pandas as pd
import yaml

# Reutilizar el motor de reporte del SISMEPRE
from app.quality.quality_report import (
    DIMENSIONS,
    DIMENSION_DESCRIPTIONS,
    COLUMN_LABELS,
    _css,
    _score_value,
    _score_width,
    _bar,
    status_badge,
    build_summary,
    severity_summary,
    _glossary_html,
    _rename_for_html,
    _dimension_cards,
)
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


# ── Contexto de hallazgos documentados del YAML ───────────────────────────────

def _findings_html(config: dict) -> str:
    """
    Genera una sección HTML con los hallazgos del profiling documentados en el YAML.
    Esto es exclusivo de SIAF — el SISMEPRE no tiene esta sección.
    """
    findings = config.get("profiling_findings", {})
    if not findings:
        return ""

    items = []
    colors = {
        "DOCUMENTAR_COMO_ESPERADO": "#d9eadf",
        "EVALUAR_CON_CLAVE_COMPLETA": "#fff3cd",
        "REPORTAR_COMO_OBSERVACION_NO_ERROR": "#fff3cd",
        "MARCAR_COMO_ANO_PARCIAL_EN_METADATA": "#eef2ff",
    }

    for key, f in findings.items():
        pct = f.get("pct_afectado") or f.get("pim_pct") or f.get("pct_ceros") or ""
        filas = f.get("filas_afectadas") or f.get("pim_negativos") or f.get("filas_con_cero") or ""
        accion = f.get("accion_quality", "")
        causa = str(f.get("causa_raiz", "")).replace("\n", " ").strip()
        color = colors.get(accion, "#f8f9fa")

        items.append(f"""
        <div style="background:{color};border-radius:8px;padding:12px 16px;margin:8px 0;
                    border-left:4px solid #6b7280;">
          <b>{_h.escape(key)}</b>
          {f'<span style="float:right;color:#374151;font-size:12px">{pct}% · {filas:,} filas</span>'
           if pct and filas else ''}
          <br><span style="font-size:12px;color:#374151">{_h.escape(causa)}</span>
          <br><code style="font-size:11px;background:#fff;padding:2px 5px;border-radius:4px">
            accion_quality: {_h.escape(accion)}</code>
        </div>
        """)

    return f"""
    <h2 class="section-title">Hallazgos documentados del profiling (base empírica)</h2>
    <div class="note">
      Estos hallazgos fueron identificados durante el profiling de 10,777,068 filas (2012–2026).
      Son comportamientos esperados del sistema SIAF, <b>no errores de datos</b>.
      Las reglas de calidad los tienen en cuenta y no los penalizan salvo que excedan los umbrales documentados.
    </div>
    {''.join(items)}
    """


# ── Reporte HTML único para SIAF ─────────────────────────────────────────────

def write_siaf_html(
    detail_df: pd.DataFrame,
    summary_row: pd.Series,
    config: dict,
    output_path: Path,
    total_rows: int,
    n_parquets: int,
) -> None:
    """
    Genera el reporte HTML de calidad para SIAF Ingresos.

    SIAF es un único dataset (a diferencia de SISMEPRE con 7 tablas),
    por eso hay un solo reporte detallado más el dashboard.
    """
    alerts = (
        detail_df[detail_df["failed_rows"] > 0]
        .sort_values("failed_rows", ascending=False)
        .head(10)
    )

    if alerts.empty:
        alerts_html = (
            '<div class="alert-card">No se detectaron observaciones en las reglas evaluadas.</div>'
        )
    else:
        alerts_html = "".join([
            f"<div class='alert-card'>"
            f"<b>{_h.escape(str(r['dimension']))}</b> — {_h.escape(str(r['rule_name']))}<br>"
            f"Columna: <code>{_h.escape(str(r['column_name']))}</code> | "
            f"Filas observadas: {int(r['failed_rows']):,} | "
            f"Cumplimiento: {_score_value(r['score'])} | "
            f"Severidad: <b>{_h.escape(str(r['severity']))}</b> | "
            f"Fuente: {_h.escape(str(r['source']))}"
            f"</div>"
            for _, r in alerts.iterrows()
        ])

    sev_df = severity_summary(detail_df)
    sev_html = (
        _rename_for_html(sev_df).to_html(index=False, escape=False)
        if not sev_df.empty
        else '<div class="alert-card">No hay observaciones por severidad.</div>'
    )

    detail_display = _rename_for_html(detail_df)
    settings = config.get("settings", {})

    doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Calidad SIAF Ingresos — Bronze</title>
  <style>{_css()}</style>
</head>
<body>
  <header>
    <h1>Calidad de Datos — SIAF Ingresos (Bronze)</h1>
    <p>Fuente: {_h.escape(settings.get('source','siaf'))} ·
       Dataset: {_h.escape(settings.get('dataset_name','ingresos'))} ·
       Capa evaluada: Bronze · Este reporte NO modifica datos</p>
  </header>
  <main>
    <div class="note">
      Evaluación de calidad sobre <b>{total_rows:,} filas</b> provenientes de
      <b>{n_parquets}</b> archivo(s) Parquet Bronze (sin _diario).
      Las reglas están justificadas por diccionario MEF, profiling empírico sobre
      el histórico 2012–2026 y reglas de negocio del sistema presupuestal peruano.
    </div>

    <div class="cards">
      <div class="card">
        <div class="label">Total filas Bronze</div>
        <div class="value">{total_rows:,}</div>
      </div>
      <div class="card">
        <div class="label">Reglas evaluadas</div>
        <div class="value">{int(summary_row['rules_evaluated']):,}</div>
      </div>
      <div class="card">
        <div class="label">Reglas con observación</div>
        <div class="value">{int(summary_row['rules_with_observations']):,}</div>
      </div>
      <div class="card">
        <div class="label">Score general</div>
        <div class="value">{_score_value(summary_row['score_general'])}</div>
      </div>
    </div>

    <h2 class="section-title">Resumen por dimensión de calidad</h2>
    <div class="quality-grid">{_dimension_cards(detail_df)}</div>

    <h2 class="section-title">Observaciones por severidad</h2>
    {sev_html}

    <h2 class="section-title">Principales observaciones</h2>
    {alerts_html}

    {_findings_html(config)}

    {_glossary_html()}

    <h2 class="section-title">Detalle completo de reglas</h2>
    {detail_display.to_html(index=False, escape=False)}
  </main>
</body>
</html>"""

    output_path.write_text(doc, encoding="utf-8")
    logger.info("Reporte HTML SIAF calidad generado: %s", output_path)


# ── Dashboard ejecutivo ───────────────────────────────────────────────────────

def write_siaf_dashboard_html(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    config: dict,
    output_path: Path,
    total_rows: int,
    n_parquets: int,
) -> None:
    """
    Dashboard ejecutivo de calidad SIAF.

    Incluye:
    - KPIs globales
    - Cards por dimensión
    - Observaciones por severidad
    - Top observaciones
    - Hallazgos documentados del profiling
    - Tabla completa de detalle
    """
    total_rules = int(summary_df["rules_evaluated"].sum())
    total_obs_rules = int(summary_df["rules_with_observations"].sum())
    total_observations = int(summary_df["observations_total"].sum())
    avg_score = summary_df["score_general"].dropna().mean()

    dim_cards = _dimension_cards(detail_df)

    top_alerts = (
        detail_df[detail_df["failed_rows"] > 0]
        .sort_values("failed_rows", ascending=False)
        .head(10)
    )
    alerts_html = "".join([
        f"<div class='alert-card'>"
        f"<b>{_h.escape(str(r['dimension']))}</b> — {_h.escape(str(r['rule_name']))}<br>"
        f"Columna: <code>{_h.escape(str(r['column_name']))}</code> | "
        f"Filas observadas: {int(r['failed_rows']):,} | "
        f"Cumplimiento: {_score_value(r['score'])} | "
        f"Severidad: {_h.escape(str(r['severity']))}"
        f"</div>"
        for _, r in top_alerts.iterrows()
    ]) or '<div class="alert-card">No se detectaron observaciones.</div>'

    sev_df = severity_summary(detail_df)
    sev_html = (
        _rename_for_html(sev_df).to_html(index=False, escape=False)
        if not sev_df.empty
        else '<div class="alert-card">No hay observaciones por severidad.</div>'
    )

    settings = config.get("settings", {})

    doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dashboard Calidad SIAF Ingresos</title>
  <style>{_css()}</style>
</head>
<body>
  <header>
    <h1>Dashboard de Calidad de Datos — SIAF Ingresos (Bronze)</h1>
    <p>8 dimensiones: Completitud · Validez · Exactitud · Consistencia ·
       Unicidad · Integridad · Oportunidad · Conformidad</p>
  </header>
  <main>
    <div class="note">
      Evaluación sobre <b>{total_rows:,} filas</b> históricas (2012–2026) en
      <b>{n_parquets}</b> archivo(s) Parquet Bronze. Se excluyen archivos <code>_diario</code>
      para evitar duplicar montos. Base empírica: profiling de {settings.get('total_rows_profiled',0):,}
      filas ejecutado en {settings.get('profiling_date','2026-05')}.
    </div>

    <div class="cards">
      <div class="card">
        <div class="label">Total filas evaluadas</div>
        <div class="value">{total_rows:,}</div>
      </div>
      <div class="card">
        <div class="label">Reglas evaluadas</div>
        <div class="value">{total_rules}</div>
      </div>
      <div class="card">
        <div class="label">Reglas con observación</div>
        <div class="value">{total_obs_rules}</div>
      </div>
      <div class="card">
        <div class="label">Score general</div>
        <div class="value">{_score_value(avg_score)}</div>
      </div>
    </div>

    <h2 class="section-title">Resumen por dimensión de calidad</h2>
    <div class="quality-grid">{dim_cards}</div>

    <h2 class="section-title">Observaciones por severidad</h2>
    {sev_html}

    <h2 class="section-title">Top 10 observaciones</h2>
    {alerts_html}

    {_findings_html(config)}

    {_glossary_html()}

    <h2 class="section-title">Detalle completo de reglas</h2>
    {_rename_for_html(detail_df).to_html(index=False, escape=False)}
  </main>
</body>
</html>"""

    output_path.write_text(doc, encoding="utf-8")
    logger.info("Dashboard HTML SIAF calidad generado: %s", output_path)