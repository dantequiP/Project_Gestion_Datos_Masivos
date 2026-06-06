from datetime import datetime
from pathlib import Path
import json
import pandas as pd


def write_quality_audit(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    failed_samples_df: pd.DataFrame,
    audit_root: Path,
    reports_root: Path,
    data_quality_root: Path,
    started_at: datetime,
) -> Path:
    """
    Escribe auditoría JSON de la corrida de calidad.

    La auditoría permite saber cuándo se ejecutó el pipeline,
    cuántas reglas se evaluaron y qué archivos se generaron.
    """
    finished_at = datetime.now()
    record = {
        "pipeline_name": "quality_sismepre",
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
            "summary_csv": str(reports_root / "sismepre_quality_summary.csv"),
            "detail_csv": str(reports_root / "sismepre_quality_detail.csv"),
            "failed_samples_csv": str(reports_root / "sismepre_quality_failed_samples.csv"),
            "dashboard_html": str(reports_root / "sismepre_quality_dashboard.html"),
            "summary_parquet": str(data_quality_root / "sismepre_quality_summary.parquet"),
            "detail_parquet": str(data_quality_root / "sismepre_quality_detail.parquet"),
            "failed_samples_parquet": str(data_quality_root / "sismepre_quality_failed_samples.parquet"),
        },
    }

    audit_dir = audit_root / finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"sismepre_quality_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path
