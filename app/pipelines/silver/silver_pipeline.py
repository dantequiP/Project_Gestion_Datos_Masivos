from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.silver.silver_audit import write_silver_audit
from app.silver.silver_report import write_dashboard_html, write_dataset_html
from app.silver.silver_transformer import SilverTransformer, load_silver_rules

try:
    from app.utils.logger import get_logger
    logger = get_logger(__name__, log_dir=Path("logs"))
except Exception:  # pragma: no cover - fallback si el proyecto no tiene logger configurado.
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)


def run_silver_pipeline(source: str = "sismepre") -> dict[str, str]:
    """
    Pipeline Silver para SISMEPRE.

    Flujo:
    1. Carga reglas desde app/config/silver_rules_sismepre.yaml.
    2. Lee Parquet Bronze por dataset.
    3. Normaliza vacíos reales a null.
    4. Aplica conversiones de tipos y tratamientos especiales.
    5. Agrega columnas técnicas de trazabilidad.
    6. Guarda Parquet Silver.
    7. Genera CSV, HTML y auditoría JSON.
    """
    if source != "sismepre":
        raise ValueError("Por ahora solo está implementado Silver para source='sismepre'.")

    started_at = datetime.now()
    logger.info("Inicio del pipeline Silver | source=%s", source)

    config_path = Path("app/config/silver_rules_sismepre.yaml")
    config = load_silver_rules(config_path)
    defaults = config.get("silver_defaults", {})

    reports_root = Path(defaults.get("reports_base_path", "reports/silver/sismepre"))
    silver_root = Path(defaults.get("silver_base_path", "data/silver/sismepre"))
    audit_root = Path(defaults.get("audit_base_path", "data/audit/silver"))

    reports_root.mkdir(parents=True, exist_ok=True)
    silver_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    transformer = SilverTransformer(config)
    results, summary_df, detail_df, dictionary_df = transformer.run(processed_at=started_at)

    summary_csv = reports_root / "sismepre_silver_summary.csv"
    detail_csv = reports_root / "sismepre_silver_detail.csv"
    dictionary_csv = reports_root / "sismepre_silver_data_dictionary.csv"
    dashboard_html = reports_root / "sismepre_silver_dashboard.html"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    dictionary_df.to_csv(dictionary_csv, index=False, encoding="utf-8-sig")

    write_dashboard_html(summary_df, detail_df, dictionary_df, dashboard_html)

    for _, summary_row in summary_df.iterrows():
        dataset = summary_row["dataset"]
        dataset_detail = detail_df[detail_df["dataset"] == dataset].copy()
        write_dataset_html(
            dataset=dataset,
            summary_row=summary_row,
            detail_df=dataset_detail,
            dictionary_df=dictionary_df,
            output_path=reports_root / f"{dataset}_silver.html",
        )

    finished_at = datetime.now()
    audit_path = write_silver_audit(
        summary_df=summary_df,
        detail_df=detail_df,
        reports_root=reports_root,
        silver_root=silver_root,
        audit_root=audit_root,
        started_at=started_at,
        finished_at=finished_at,
    )

    logger.info("Pipeline Silver finalizado.")
    logger.info("Dashboard HTML: %s", dashboard_html)
    logger.info("Auditoría: %s", audit_path)

    outputs = {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "data_dictionary_csv": str(dictionary_csv),
        "dashboard_html": str(dashboard_html),
        "reports_dir": str(reports_root),
        "silver_dir": str(silver_root),
        "audit_path": str(audit_path),
    }

    # Agrega rutas Silver por dataset para revisión rápida.
    for result in results:
        outputs[f"silver_{result.dataset}"] = str(result.silver_path)

    return outputs
