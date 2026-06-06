from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

DIMENSIONS = ["Completitud", "Exactitud", "Consistencia", "Integridad", "Razonabilidad", "Oportunidad", "Unicidad", "Validez"]

DIMENSION_DESCRIPTIONS = {
    "Completitud": "Campos obligatorios con datos disponibles.",
    "Exactitud": "Valores con formato correcto según el tipo esperado.",
    "Consistencia": "Coherencia entre columnas relacionadas dentro del dataset.",
    "Integridad": "Relaciones válidas entre datasets internos.",
    "Razonabilidad": "Valores lógicos para el dominio, como montos no negativos.",
    "Oportunidad": "Vigencia temporal del dato respecto al periodo de análisis.",
    "Unicidad": "Ausencia de duplicados y claves candidatas únicas.",
    "Validez": "Formatos, catálogos y tipos de datos permitidos.",
}

COLUMN_LABELS = {
    "dataset": "Dataset evaluado",
    "dimension": "Dimensión de calidad",
    "rule_id": "Identificador de regla",
    "rule_name": "Regla aplicada",
    "column_name": "Columna evaluada",
    "rule_type": "Tipo de regla",
    "expected_value": "Valor esperado",
    "evaluated_rows": "Filas evaluadas",
    "passed_rows": "Filas que cumplen",
    "failed_rows": "Filas observadas",
    "score": "Porcentaje de cumplimiento",
    "severity": "Severidad",
    "source": "Fuente de la regla",
    "status": "Estado",
    "observation": "Descripción de la observación",
    "sample_failed_rows": "Filas de ejemplo",
    "rules_evaluated": "Reglas evaluadas",
    "rules_with_observations": "Reglas con observación",
    "observations_total": "Total de observaciones",
    "evaluated_rows_total": "Total de filas evaluadas",
    "score_general": "Score general",
    "estado_descriptivo": "Estado descriptivo",
}


def _css() -> str:
    return """
    body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
    header{background:#1f2430;color:white;padding:24px 32px} header h1{margin:0;font-size:25px} header p{margin:8px 0 0;color:#cbd5e1}
    main{padding:24px 32px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .quality-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .card,.qcard{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #3b82f6}
    .qcard{min-height:180px}.qcard h3{margin:0 0 6px;font-size:15px}.desc{font-size:12px;color:#6b7280;min-height:34px}
    .label{color:#6b7280;text-transform:uppercase;font-size:12px;letter-spacing:.05em}.value{font-size:28px;font-weight:bold;margin-top:8px;color:#2f7d46}
    .score-big{font-size:26px;font-weight:bold;color:#2f7d46;float:right}.small{font-size:12px;color:#6b7280}
    .section-title{color:#4b5563;text-transform:uppercase;letter-spacing:.08em;font-size:13px;margin:24px 0 12px}
    table{border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:middle}th{background:#111827;color:white}tr:nth-child(even){background:#fafafa}
    .bar-wrap{width:100%;height:7px;background:#e5e7eb;border-radius:999px;margin:10px 0}.bar-fill{height:7px;background:#2f7d46;border-radius:999px}
    .inline-bar{display:inline-block;width:140px;height:8px;background:#e5e7eb;border-radius:999px;vertical-align:middle;margin-right:8px}.inline-fill{height:8px;background:#2f7d46;border-radius:999px}.bar-label{color:#2f7d46;font-weight:bold}
    .badge{padding:5px 10px;border-radius:999px;font-weight:bold;font-size:12px;white-space:nowrap}.sin{background:#d9eadf;color:#245c36}.con{background:#fff3cd;color:#7a5b00}.noeval{background:#e5e7eb;color:#374151}
    .alert-card{background:white;border-left:4px solid #c24141;border-radius:10px;padding:12px 16px;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .note{background:#eef2ff;border-left:4px solid #3b82f6;border-radius:10px;padding:14px 16px;margin-bottom:20px}
    code{background:#eef2ff;padding:2px 4px;border-radius:4px}
    """


def _score_value(score) -> str:
    if pd.isna(score):
        return "N/E"
    return f"{float(score):.1f}%"


def _score_width(score) -> float:
    if pd.isna(score):
        return 0
    return max(0, min(100, float(score)))


def _bar(score) -> str:
    width = _score_width(score)
    return f'<span class="inline-bar"><span class="inline-fill" style="display:block;width:{width:.2f}%"></span></span><span class="bar-label">{_score_value(score)}</span>'


def status_badge(status: str) -> str:
    cls = "sin" if status == "Sin observaciones" else ("noeval" if status == "No evaluable" else "con")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _rename_for_html(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas técnicas a nombres entendibles para el HTML.
    El CSV conserva nombres técnicos para trazabilidad.
    """
    return df.rename(columns={c: COLUMN_LABELS.get(c, c) for c in df.columns})


def _glossary_html() -> str:
    rows = []
    explanations = {
        "dataset": "Archivo o tabla analizada.",
        "dimension": "Criterio de calidad evaluado.",
        "rule_id": "Código único de la regla aplicada.",
        "rule_name": "Descripción breve de la validación.",
        "column_name": "Campo específico revisado.",
        "evaluated_rows": "Cantidad de registros revisados por la regla.",
        "passed_rows": "Registros que cumplieron la regla.",
        "failed_rows": "Registros que no cumplieron o quedaron observados.",
        "score": "Porcentaje de cumplimiento de la regla.",
        "severity": "Nivel de importancia de la observación.",
        "source": "Origen que justifica la regla: diccionario, profiling o inferencia documentada.",
        "status": "Resultado de la regla.",
        "sample_failed_rows": "Índices de filas usados como ejemplo de evidencia.",
    }
    for tech, label in COLUMN_LABELS.items():
        if tech in explanations:
            rows.append(f"<tr><td><code>{html.escape(tech)}</code></td><td>{html.escape(label)}</td><td>{html.escape(explanations[tech])}</td></tr>")
    return f"""
    <h2 class="section-title">Glosario de columnas del reporte</h2>
    <table><thead><tr><th>Columna técnica</th><th>Nombre en español</th><th>Explicación</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    """


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Construye resumen por dataset y dimensión."""
    rows = []
    for dataset, ddf in detail_df.groupby("dataset"):
        row = {
            "dataset": dataset,
            "rules_evaluated": int(len(ddf)),
            "rules_with_observations": int((ddf["failed_rows"] > 0).sum()),
            "observations_total": int(ddf["failed_rows"].sum()),
            "evaluated_rows_total": int(ddf["evaluated_rows"].sum()),
        }
        scores = []
        for dim in DIMENSIONS:
            dim_df = ddf[ddf["dimension"] == dim]
            if dim_df.empty:
                row[f"score_{dim.lower()}"] = None
            else:
                score = dim_df["score"].dropna().mean()
                row[f"score_{dim.lower()}"] = round(float(score), 4) if not pd.isna(score) else None
                if not pd.isna(score):
                    scores.append(score)

        row["score_general"] = round(float(pd.Series(scores).mean()), 4) if scores else None
        row["estado_descriptivo"] = "Sin observaciones" if row["rules_with_observations"] == 0 else "Con observaciones"
        rows.append(row)

    return pd.DataFrame(rows)


def severity_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Resume observaciones por severidad."""
    observed = detail_df[detail_df["failed_rows"] > 0]
    if observed.empty:
        return pd.DataFrame(columns=["severity", "rules_with_observations", "observations_total"])
    return (
        observed.groupby("severity", dropna=False)
        .agg(rules_with_observations=("rule_id", "count"), observations_total=("failed_rows", "sum"))
        .reset_index()
        .sort_values("observations_total", ascending=False)
    )


def _dimension_cards(dataset_df: pd.DataFrame) -> str:
    cards = []
    for dim in DIMENSIONS:
        dim_df = dataset_df[dataset_df["dimension"] == dim]
        if dim_df.empty:
            score = None
            rules = 0
            obs = 0
            status = "No evaluable"
        else:
            score = dim_df["score"].dropna().mean()
            score = round(float(score), 4) if not pd.isna(score) else None
            rules = len(dim_df)
            obs = int(dim_df["failed_rows"].sum())
            status = "Sin observaciones" if obs == 0 else "Con observaciones"

        width = _score_width(score)
        cards.append(f"""
        <div class="qcard">
          <h3>{html.escape(dim)} <span class="score-big">{_score_value(score)}</span></h3>
          <div class="desc">{html.escape(DIMENSION_DESCRIPTIONS.get(dim, ""))}</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:{width:.2f}%"></div></div>
          <div class="small">Reglas: <b>{rules}</b> · Observaciones: <b>{obs:,}</b></div>
          <p>{status_badge(status)}</p>
        </div>
        """)

    return "".join(cards)


def write_dataset_html(dataset: str, dataset_df: pd.DataFrame, summary_row: pd.Series, output_path: Path) -> None:
    """Genera reporte HTML individual por dataset."""
    alerts = dataset_df[dataset_df["failed_rows"] > 0].sort_values("failed_rows", ascending=False).head(8)
    if alerts.empty:
        alerts_html = '<div class="alert-card">No se detectaron observaciones en las reglas evaluadas.</div>'
    else:
        alerts_html = "".join([
            f"<div class='alert-card'><b>{html.escape(str(r['dimension']))}</b> — {html.escape(str(r['rule_name']))}<br>Columna: <code>{html.escape(str(r['column_name']))}</code> | Filas observadas: {int(r['failed_rows']):,} | Cumplimiento: {_score_value(r['score'])} | Ejemplos: <code>{html.escape(str(r.get('sample_failed_rows','[]')))}</code></div>"
            for _, r in alerts.iterrows()
        ])

    sev_df = severity_summary(dataset_df)
    sev_html = _rename_for_html(sev_df).to_html(index=False, escape=False) if not sev_df.empty else '<div class="alert-card">No hay observaciones por severidad.</div>'

    detail_display = _rename_for_html(dataset_df)

    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Calidad SISMEPRE - {html.escape(dataset)}</title><style>{_css()}</style></head><body>
    <header><h1>Calidad de Datos — {html.escape(dataset)}</h1><p>Capa evaluada: Bronze | Este reporte no modifica datos</p></header><main>
    <div class="note">Lectura ejecutiva: este reporte muestra el cumplimiento de reglas de calidad por dimensión. Una observación no siempre significa error crítico; su importancia depende de la severidad y de la fuente de la regla.</div>
    <div class="cards">
      <div class="card"><div class="label">Reglas evaluadas</div><div class="value">{int(summary_row['rules_evaluated']):,}</div></div>
      <div class="card"><div class="label">Reglas con observación</div><div class="value">{int(summary_row['rules_with_observations']):,}</div></div>
      <div class="card"><div class="label">Observaciones</div><div class="value">{int(summary_row['observations_total']):,}</div></div>
      <div class="card"><div class="label">Score general</div><div class="value">{_score_value(summary_row['score_general'])}</div></div>
    </div>
    <h2 class="section-title">Resumen ejecutivo por dimensión</h2>
    <div class="quality-grid">{_dimension_cards(dataset_df)}</div>
    <h2 class="section-title">Observaciones por severidad</h2>{sev_html}
    <h2 class="section-title">Principales observaciones</h2>{alerts_html}
    {_glossary_html()}
    <h2 class="section-title">Detalle completo de reglas</h2>{detail_display.to_html(index=False, escape=False)}
    </main></body></html>"""
    output_path.write_text(doc, encoding="utf-8")


def write_dashboard_html(summary_df: pd.DataFrame, detail_df: pd.DataFrame, output_path: Path) -> None:
    """Genera dashboard HTML ejecutivo general."""
    total_datasets = len(summary_df)
    total_rules = int(summary_df["rules_evaluated"].sum())
    total_obs_rules = int(summary_df["rules_with_observations"].sum())
    total_observations = int(summary_df["observations_total"].sum())
    avg_score = summary_df["score_general"].dropna().mean()

    dim_cards = _dimension_cards(detail_df)

    ds_rows = []
    for _, r in summary_df.iterrows():
        ds_rows.append(f"""
        <tr>
          <td><b>{html.escape(str(r['dataset']))}</b></td>
          <td>{int(r['rules_evaluated'])}</td>
          <td>{int(r['rules_with_observations'])}</td>
          <td>{int(r['observations_total']):,}</td>
          <td>{_bar(r['score_general'])}</td>
          <td>{status_badge(str(r['estado_descriptivo']))}</td>
        </tr>
        """)

    top_alerts = detail_df[detail_df["failed_rows"] > 0].sort_values("failed_rows", ascending=False).head(10)
    alerts_html = "".join([
        f"<div class='alert-card'><b>{html.escape(str(r['dataset']))}</b> · {html.escape(str(r['dimension']))} — {html.escape(str(r['rule_name']))}<br>Columna: <code>{html.escape(str(r['column_name']))}</code> | Filas observadas: {int(r['failed_rows']):,} | Cumplimiento: {_score_value(r['score'])} | Severidad: {html.escape(str(r['severity']))}</div>"
        for _, r in top_alerts.iterrows()
    ]) or '<div class="alert-card">No se detectaron observaciones.</div>'

    sev_df = severity_summary(detail_df)
    sev_html = _rename_for_html(sev_df).to_html(index=False, escape=False) if not sev_df.empty else '<div class="alert-card">No hay observaciones por severidad.</div>'

    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Dashboard Calidad SISMEPRE</title><style>{_css()}</style></head><body>
    <header><h1>Dashboard de Calidad de Datos — SISMEPRE Bronze</h1><p>Evaluación con 8 dimensiones: completitud, exactitud, consistencia, integridad, razonabilidad, oportunidad, unicidad y validez</p></header><main>
    <div class="note">Este dashboard evalúa la calidad interna de SISMEPRE. Las validaciones cruzadas con SIAF y RENAMU se deben incorporar después de ingesta, validación y profiling de esas fuentes.</div>
    <div class="cards">
      <div class="card"><div class="label">Datasets evaluados</div><div class="value">{total_datasets}</div></div>
      <div class="card"><div class="label">Reglas evaluadas</div><div class="value">{total_rules}</div></div>
      <div class="card"><div class="label">Reglas con observación</div><div class="value">{total_obs_rules}</div></div>
      <div class="card"><div class="label">Score promedio</div><div class="value">{_score_value(avg_score)}</div></div>
    </div>
    <h2 class="section-title">Resumen ejecutivo por dimensión</h2>
    <div class="quality-grid">{dim_cards}</div>
    <h2 class="section-title">Observaciones por severidad</h2>{sev_html}
    <h2 class="section-title">Resumen por dataset</h2>
    <table><thead><tr><th>Dataset evaluado</th><th>Reglas</th><th>Reglas con observación</th><th>Total de observaciones</th><th>Score general</th><th>Estado</th></tr></thead><tbody>{''.join(ds_rows)}</tbody></table>
    <h2 class="section-title">Principales observaciones</h2>{alerts_html}
    {_glossary_html()}
    <h2 class="section-title">Detalle completo</h2>{_rename_for_html(detail_df).to_html(index=False, escape=False)}
    </main></body></html>"""
    output_path.write_text(doc, encoding="utf-8")
