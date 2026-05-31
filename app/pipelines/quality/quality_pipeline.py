from datetime import datetime
from pathlib import Path

from app.quality.quality_audit import write_quality_audit
from app.quality.quality_checker import QualityChecker
from app.quality.quality_report import build_summary, write_dashboard_html, write_dataset_html
from app.quality.quality_rules import load_quality_rules
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))

def run_quality_pipeline(source: str = "sismepre") -> dict[str, str]:
    if source != "sismepre":
        raise ValueError("Por ahora solo está implementada calidad para source='sismepre'.")

    started_at = datetime.now()
    logger.info("Inicio del pipeline de calidad | source=%s", source)

    config = load_quality_rules(Path("app/config/quality_rules_sismepre.yaml"))
    settings = config["settings"]
    reports_root = Path(settings.get("reports_root", "reports/quality/sismepre"))
    data_quality_root = Path(settings.get("data_quality_root", "data/quality/sismepre"))
    audit_root = Path(settings.get("audit_root", "data/audit/quality"))

    reports_root.mkdir(parents=True, exist_ok=True)
    data_quality_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    checker = QualityChecker(config)
    detail_df = checker.run()
    summary_df = build_summary(detail_df)

    detail_csv = reports_root / "sismepre_quality_detail.csv"
    summary_csv = reports_root / "sismepre_quality_summary.csv"
    dashboard_html = reports_root / "sismepre_quality_dashboard.html"
    detail_parquet = data_quality_root / "sismepre_quality_detail.parquet"
    summary_parquet = data_quality_root / "sismepre_quality_summary.parquet"

    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    detail_df.to_parquet(detail_parquet, index=False)
    summary_df.to_parquet(summary_parquet, index=False)

    write_dashboard_html(summary_df, detail_df, dashboard_html)

    for _, row in summary_df.iterrows():
        dataset = row["dataset"]
        write_dataset_html(dataset, detail_df[detail_df["dataset"] == dataset].copy(), row, reports_root / f"{dataset}_quality.html")

    audit_path = write_quality_audit(summary_df, detail_df, audit_root, reports_root, data_quality_root, started_at)

    logger.info("Pipeline de calidad finalizado.")
    logger.info("Dashboard HTML: %s", dashboard_html)
    logger.info("Auditoría: %s", audit_path)

    return {
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "dashboard_html": str(dashboard_html),
        "summary_parquet": str(summary_parquet),
        "detail_parquet": str(detail_parquet),
        "reports_dir": str(reports_root),
        "audit_path": str(audit_path),
    }