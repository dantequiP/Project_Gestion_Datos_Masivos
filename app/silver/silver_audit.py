from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def write_silver_audit(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    reports_root: Path,
    silver_root: Path,
    audit_root: Path,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """Escribe auditoría JSON de ejecución Silver."""
    audit_date_dir = audit_root / f"year={finished_at:%Y}" / f"month={finished_at:%m}" / f"day={finished_at:%d}"
    audit_date_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_date_dir / f"sismepre_silver_{finished_at:%Y%m%d_%H%M%S}.json"

    payload: dict[str, Any] = {
        "source": "sismepre",
        "layer": "silver",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "datasets_processed": int(len(summary_df)),
        "total_bronze_rows": int(summary_df["bronze_rows"].sum()) if not summary_df.empty else 0,
        "total_silver_rows": int(summary_df["silver_rows"].sum()) if not summary_df.empty else 0,
        "total_conversion_errors": int(summary_df["conversion_errors"].sum()) if not summary_df.empty else 0,
        "reports_root": str(reports_root),
        "silver_root": str(silver_root),
        "summary": summary_df.to_dict(orient="records"),
        "status_by_column": detail_df[["dataset", "column_name", "status", "conversion_errors", "transformation"]].to_dict(orient="records") if not detail_df.empty else [],
    }

    audit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path
