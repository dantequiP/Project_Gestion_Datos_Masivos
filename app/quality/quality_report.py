from __future__ import annotations

import html
from pathlib import Path
import pandas as pd

DIMENSIONS = ["Completitud", "Exactitud", "Consistencia", "Integridad", "Razonabilidad", "Oportunidad", "Unicidad", "Validez"]

def _bar(score) -> str:
    if pd.isna(score):
        label, width = "N/E", 0
    else:
        label, width = f"{float(score):.1f}%", max(0, min(100, float(score)))
    return f'<div class="bar-wrap"><div class="bar-fill" style="width:{width:.2f}%"></div></div><span class="bar-label">{label}</span>'

def _css() -> str:
    return """
    body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
    header{background:#1f2430;color:white;padding:24px 32px} header h1{margin:0;font-size:24px} header p{margin:8px 0 0;color:#cbd5e1}
    main{padding:24px 32px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #3b82f6}
    .label{color:#6b7280;text-transform:uppercase;font-size:12px;letter-spacing:.05em}.value{font-size:28px;font-weight:bold;margin-top:8px;color:#2f7d46}
    .section-title{color:#4b5563;text-transform:uppercase;letter-spacing:.08em;font-size:13px;margin:24px 0 12px}
    table{border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:middle}th{background:#111827;color:white}tr:nth-child(even){background:#fafafa}
    .bar-wrap{display:inline-block;width:140px;height:8px;background:#e5e7eb;border-radius:999px;vertical-align:middle;margin-right:8px}.bar-fill{height:8px;background:#2f7d46;border-radius:999px}.bar-label{color:#2f7d46;font-weight:bold}
    .badge{padding:5px 10px;border-radius:999px;font-weight:bold;font-size:12px;white-space:nowrap}.sin{background:#d9eadf;color:#245c36}.con{background:#fff3cd;color:#7a5b00}.noeval{background:#e5e7eb;color:#374151}
    .alert-card{background:white;border-left:4px solid #c24141;border-radius:10px;padding:12px 16px;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}
    """

def status_badge(status: str) -> str:
    cls = "sin" if status == "Sin observaciones" else ("noeval" if status == "No evaluable" else "con")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'

def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
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

def write_dataset_html(dataset: str, dataset_df: pd.DataFrame, summary_row: pd.Series, output_path: Path) -> None:
    dim_rows = []
    for dim in DIMENSIONS:
        dim_df = dataset_df[dataset_df["dimension"] == dim]
        score = dim_df["score"].dropna().mean() if not dim_df.empty else None
        score = round(float(score), 4) if score is not None and not pd.isna(score) else None
        obs = int(dim_df["failed_rows"].sum()) if not dim_df.empty else 0
        status = "No evaluable" if dim_df.empty else ("Sin observaciones" if obs == 0 else "Con observaciones")
        dim_rows.append(f"<tr><td><b>{dim}</b></td><td>{_bar(score)}</td><td>{len(dim_df)}</td><td>{obs:,}</td><td>{status_badge(status)}</td></tr>")

    alerts = dataset_df[dataset_df["failed_rows"] > 0].sort_values("failed_rows", ascending=False).head(6)
    if alerts.empty:
        alerts_html = '<div class="alert-card">No se detectaron observaciones en las reglas evaluadas.</div>'
    else:
        alerts_html = "".join([
            f"<div class='alert-card'><b>{html.escape(str(r['dimension']))}</b> — {html.escape(str(r['rule_name']))}<br>Columna: <code>{html.escape(str(r['column_name']))}</code> | Fallidos: {int(r['failed_rows']):,} | Severidad: {html.escape(str(r['severity']))}</div>"
            for _, r in alerts.iterrows()
        ])

    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Calidad SISMEPRE - {html.escape(dataset)}</title><style>{_css()}</style></head><body>
    <header><h1>Calidad de Datos — {html.escape(dataset)}</h1><p>Capa evaluada: Bronze | Este reporte no modifica datos</p></header><main>
    <div class="cards">
    <div class="card"><div class="label">Reglas evaluadas</div><div class="value">{int(summary_row['rules_evaluated']):,}</div></div>
    <div class="card"><div class="label">Reglas con observación</div><div class="value">{int(summary_row['rules_with_observations']):,}</div></div>
    <div class="card"><div class="label">Observaciones</div><div class="value">{int(summary_row['observations_total']):,}</div></div>
    <div class="card"><div class="label">Score general</div><div class="value">{'' if pd.isna(summary_row['score_general']) else f"{float(summary_row['score_general']):.1f}%"}</div></div>
    </div><h2 class="section-title">Calidad por dimensión</h2><table><thead><tr><th>Dimensión</th><th>Score</th><th>Reglas</th><th>Observaciones</th><th>Estado</th></tr></thead><tbody>{''.join(dim_rows)}</tbody></table>
    <h2 class="section-title">Principales observaciones</h2>{alerts_html}
    <h2 class="section-title">Detalle de reglas</h2>{dataset_df.to_html(index=False, escape=False)}
    </main></body></html>"""
    output_path.write_text(doc, encoding="utf-8")

def write_dashboard_html(summary_df: pd.DataFrame, detail_df: pd.DataFrame, output_path: Path) -> None:
    total_datasets = len(summary_df); total_rules = int(summary_df["rules_evaluated"].sum()); total_obs_rules = int(summary_df["rules_with_observations"].sum())
    avg_score = summary_df["score_general"].dropna().mean()
    avg_text = "N/E" if pd.isna(avg_score) else f"{float(avg_score):.1f}%"
    dim_rows = []
    for dim in DIMENSIONS:
        dim_df = detail_df[detail_df["dimension"] == dim]
        score = dim_df["score"].dropna().mean() if not dim_df.empty else None
        score = round(float(score), 4) if score is not None and not pd.isna(score) else None
        dim_rows.append(f"<tr><td><b>{dim}</b></td><td>{_bar(score)}</td><td>{len(dim_df)}</td><td>{int(dim_df['failed_rows'].sum()) if not dim_df.empty else 0:,}</td></tr>")
    ds_rows = []
    for _, r in summary_df.iterrows():
        ds_rows.append(f"<tr><td><b>{html.escape(str(r['dataset']))}</b></td><td>{int(r['rules_evaluated'])}</td><td>{int(r['rules_with_observations'])}</td><td>{int(r['observations_total']):,}</td><td>{_bar(r['score_general'])}</td><td>{status_badge(str(r['estado_descriptivo']))}</td></tr>")
    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><title>Dashboard Calidad SISMEPRE</title><style>{_css()}</style></head><body>
    <header><h1>Dashboard de Calidad de Datos — SISMEPRE Bronze</h1><p>Evaluación con 8 dimensiones: completitud, exactitud, consistencia, integridad, razonabilidad, oportunidad, unicidad y validez</p></header><main>
    <div class="cards"><div class="card"><div class="label">Datasets evaluados</div><div class="value">{total_datasets}</div></div><div class="card"><div class="label">Reglas evaluadas</div><div class="value">{total_rules}</div></div><div class="card"><div class="label">Reglas con observación</div><div class="value">{total_obs_rules}</div></div><div class="card"><div class="label">Score promedio</div><div class="value">{avg_text}</div></div></div>
    <h2 class="section-title">Resumen por dimensión</h2><table><thead><tr><th>Dimensión</th><th>Score promedio</th><th>Reglas</th><th>Observaciones</th></tr></thead><tbody>{''.join(dim_rows)}</tbody></table>
    <h2 class="section-title">Resumen por dataset</h2><table><thead><tr><th>Dataset</th><th>Reglas</th><th>Reglas con observación</th><th>Observaciones</th><th>Score general</th><th>Estado</th></tr></thead><tbody>{''.join(ds_rows)}</tbody></table>
    <h2 class="section-title">Detalle completo</h2>{detail_df.to_html(index=False, escape=False)}
    </main></body></html>"""
    output_path.write_text(doc, encoding="utf-8")