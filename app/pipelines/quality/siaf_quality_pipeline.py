"""
siaf_quality_pipeline.py

Pipeline de calidad de datos para SIAF Ingresos.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.quality.quality_report import (
    DIMENSIONS,
    DIMENSION_DESCRIPTIONS,
    _css,
    _bar,
    _score_value,
    _score_width,
    status_badge,
    severity_summary,
    _rename_for_html,
    _glossary_html,
)
from app.schema_validation.siaf_quality_checker import SiafQualityChecker, load_quality_rules
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


# ──────────────────────────────────────────────────────────────────────────────
# Construcción de resumen y auditoría
# ──────────────────────────────────────────────────────────────────────────────

def build_siaf_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Construye resumen por dataset y dimensión para SIAF."""
    rows = []
    for dataset, ddf in detail_df.groupby("dataset"):
        row: dict[str, Any] = {
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


def _dimension_cards_siaf(dataset_df: pd.DataFrame) -> str:
    """Genera tarjetas HTML por dimensión de calidad para SIAF."""
    import html as _html
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
          <h3>{_html.escape(dim)} <span class="score-big">{_score_value(score)}</span></h3>
          <div class="desc">{_html.escape(DIMENSION_DESCRIPTIONS.get(dim, ""))}</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:{width:.2f}%"></div></div>
          <div class="small">Reglas: <b>{rules}</b> · Observaciones: <b>{obs:,}</b></div>
          <p>{status_badge(status)}</p>
        </div>
        """)
    return "".join(cards)


def write_siaf_quality_dashboard(summary_df: pd.DataFrame, detail_df: pd.DataFrame, output_path: Path) -> None:
    """Genera dashboard HTML ejecutivo de calidad SIAF."""
    import html as _html

    total_datasets = len(summary_df)
    total_rules = int(summary_df["rules_evaluated"].sum())
    total_obs_rules = int(summary_df["rules_with_observations"].sum())
    total_observations = int(summary_df["observations_total"].sum())
    avg_score = summary_df["score_general"].dropna().mean()

    dim_cards = _dimension_cards_siaf(detail_df)

    ds_rows = []
    for _, r in summary_df.iterrows():
        ds_rows.append(f"""
        <tr>
          <td><b>{_html.escape(str(r['dataset']))}</b></td>
          <td>{int(r['rules_evaluated'])}</td>
          <td>{int(r['rules_with_observations'])}</td>
          <td>{int(r['observations_total']):,}</td>
          <td>{_bar(r['score_general'])}</td>
          <td>{status_badge(str(r['estado_descriptivo']))}</td>
        </tr>
        """)

    top_alerts = detail_df[detail_df["failed_rows"] > 0].sort_values("failed_rows", ascending=False).head(10)
    alerts_html = "".join([
        f"<div class='alert-card'><b>{_html.escape(str(r['dataset']))}</b> · "
        f"{_html.escape(str(r['dimension']))} — {_html.escape(str(r['rule_name']))}<br>"
        f"Columna: <code>{_html.escape(str(r['column_name']))}</code> | "
        f"Filas observadas: {int(r['failed_rows']):,} | "
        f"Cumplimiento: {_score_value(r['score'])} | "
        f"Severidad: {_html.escape(str(r['severity']))}</div>"
        for _, r in top_alerts.iterrows()
    ]) or '<div class="alert-card">No se detectaron observaciones.</div>'

    sev_df = severity_summary(detail_df)
    sev_html = _rename_for_html(sev_df).to_html(index=False, escape=False) if not sev_df.empty else '<div class="alert-card">No hay observaciones por severidad.</div>'

    doc = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Dashboard Calidad SIAF</title><style>{_css()}</style></head><body>
<header>
  <h1>Dashboard de Calidad de Datos — SIAF Ingresos Bronze</h1>
  <p>Evaluación con 8 dimensiones: completitud, validez, exactitud, consistencia, unicidad, integridad, oportunidad y conformidad</p>
</header><main>
<div class="note">
  Este dashboard evalúa la calidad interna de SIAF Ingresos.
  Las validaciones cruzadas con SISMEPRE y RENAMU se realizarán en la capa Gold.
</div>
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
<table><thead><tr>
  <th>Dataset</th><th>Reglas</th><th>Reglas con observación</th>
  <th>Total observaciones</th><th>Score general</th><th>Estado</th>
</tr></thead><tbody>{''.join(ds_rows)}</tbody></table>
<h2 class="section-title">Principales observaciones</h2>{alerts_html}
{_glossary_html()}
<h2 class="section-title">Detalle completo</h2>{_rename_for_html(detail_df).to_html(index=False, escape=False)}
</main></body></html>"""

    output_path.write_text(doc, encoding="utf-8")


def write_siaf_quality_audit(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    failed_samples_df: pd.DataFrame,
    audit_root: Path,
    reports_root: Path,
    data_quality_root: Path,
    started_at: datetime,
) -> Path:
    """Escribe auditoría JSON de la corrida de calidad SIAF."""
    finished_at = datetime.now()
    record = {
        "pipeline_name": "quality_siaf",
        "status": "success",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "datasets_evaluated": int(len(summary_df)),
        "rules_evaluated": int(len(detail_df)),
        "rules_with_observations": int((detail_df["failed_rows"] > 0).sum()),
        "observations_total": int(detail_df["failed_rows"].sum()),
        "failed_samples_total": int(len(failed_samples_df)),
        "outputs": {
            "summary_csv": str(reports_root / "siaf_quality_summary.csv"),
            "detail_csv": str(reports_root / "siaf_quality_detail.csv"),
            "failed_samples_csv": str(reports_root / "siaf_quality_failed_samples.csv"),
            "dashboard_html": str(reports_root / "siaf_quality_dashboard.html"),
            "summary_parquet": str(data_quality_root / "siaf_quality_summary.parquet"),
            "detail_parquet": str(data_quality_root / "siaf_quality_detail.parquet"),
            "failed_samples_parquet": str(data_quality_root / "siaf_quality_failed_samples.parquet"),
        },
    }

    audit_dir = audit_root / finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"siaf_quality_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_siaf_quality_pipeline() -> dict[str, str]:
    """
    Pipeline principal de calidad para SIAF Ingresos.

    Flujo:
    1. Carga reglas desde quality_rules_siaf.yaml.
    2. Lee Parquet Bronze (combina todos los años disponibles).
    3. Evalúa reglas por las 8 dimensiones de calidad.
    4. Genera CSV, Parquet, HTML y auditoría.
    5. No modifica archivos Bronze.
    """
    started_at = datetime.now()
    logger.info("=== Inicio pipeline de calidad SIAF ===")

    config = load_quality_rules(Path("app/config/quality_rules_siaf.yaml"))
    settings = config["settings"]
    reports_root = Path(settings.get("reports_root", "reports/quality/siaf"))
    data_quality_root = Path(settings.get("data_quality_root", "data/quality/siaf"))
    audit_root = Path(settings.get("audit_root", "data/audit/quality"))

    reports_root.mkdir(parents=True, exist_ok=True)
    data_quality_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    checker = SiafQualityChecker(config)
    detail_df, failed_samples_df = checker.run()
    summary_df = build_siaf_summary(detail_df)

    # ── Salidas tabulares ────────────────────────────────────────────────────
    detail_csv = reports_root / "siaf_quality_detail.csv"
    summary_csv = reports_root / "siaf_quality_summary.csv"
    failed_samples_csv = reports_root / "siaf_quality_failed_samples.csv"
    dashboard_html = reports_root / "siaf_quality_dashboard.html"
    detail_parquet = data_quality_root / "siaf_quality_detail.parquet"
    summary_parquet = data_quality_root / "siaf_quality_summary.parquet"
    failed_samples_parquet = data_quality_root / "siaf_quality_failed_samples.parquet"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    failed_samples_df.to_csv(failed_samples_csv, index=False, encoding="utf-8-sig")

    detail_df.to_parquet(detail_parquet, index=False)
    summary_df.to_parquet(summary_parquet, index=False)
    if not failed_samples_df.empty:
        failed_samples_df.to_parquet(failed_samples_parquet, index=False)

    # ── Dashboard HTML ejecutivo ──────────────────────────────────────────────
    write_siaf_quality_dashboard(summary_df, detail_df, dashboard_html)

    # ── Reporte individual por dataset ────────────────────────────────────────
    # SIAF tiene un único dataset ('ingresos'), pero se genera igual que SISMEPRE
    # para consistencia de estructura de carpetas y reportes.
    from app.quality.quality_report import write_dataset_html as _write_ds_html
    for _, row in summary_df.iterrows():
        dataset = row["dataset"]
        _write_ds_html(
            dataset,
            detail_df[detail_df["dataset"] == dataset].copy(),
            row,
            reports_root / f"{dataset}_quality.html",
        )

    # ── Auditoría ─────────────────────────────────────────────────────────────
    audit_path = write_siaf_quality_audit(
        summary_df, detail_df, failed_samples_df,
        audit_root, reports_root, data_quality_root, started_at,
    )

    duration = (datetime.now() - started_at).total_seconds()
    logger.info("=== Pipeline de calidad SIAF completado en %.1fs ===", duration)
    logger.info("Dashboard HTML: %s", dashboard_html)
    logger.info("Auditoría: %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "failed_samples_csv": str(failed_samples_csv),
        "dashboard_html": str(dashboard_html),
        "summary_parquet": str(summary_parquet),
        "detail_parquet": str(detail_parquet),
        "failed_samples_parquet": str(failed_samples_parquet),
        "reports_dir": str(reports_root),
        "audit_path": str(audit_path),
    }


if __name__ == "__main__":
    run_siaf_quality_pipeline()