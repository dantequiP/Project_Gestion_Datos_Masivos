"""
siaf_silver_audit.py

Genera la auditoría JSON de cada ejecución del pipeline Silver SIAF.
Mismo patrón que silver_audit.py de SISMEPRE, adaptado a SIAF.

Outputs:
    data/audit/silver/year=YYYY/month=MM/day=DD/
        siaf_silver_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from app.silver.siaf_silver_transformer import SiafSilverResult


def write_siaf_silver_audit(
    result: "SiafSilverResult",
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    reports_root: Path,
    silver_root: Path,
    audit_root: Path,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """
    Escribe el JSON de auditoría de la ejecución Silver SIAF.

    Registra:
    - Timestamps de inicio y fin.
    - Métricas de filas (Bronze → filtrado → Silver).
    - Columnas transformadas, derivadas y técnicas.
    - Errores de conversión.
    - Rutas de todos los outputs generados.

    Retorna la ruta del archivo JSON creado.
    """
    audit_dir = (
        audit_root
        / f"year={finished_at:%Y}"
        / f"month={finished_at:%m}"
        / f"day={finished_at:%d}"
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_path = audit_dir / f"siaf_silver_{finished_at:%Y%m%d_%H%M%S}.json"

    # Detalle de columnas con error para incluir en el JSON
    error_cols: list[dict[str, Any]] = []
    if not detail_df.empty and "conversion_errors" in detail_df.columns:
        error_df = detail_df[detail_df["conversion_errors"] > 0]
        error_cols = error_df[
            ["dataset", "column_name", "bronze_type", "silver_type", "conversion_errors"]
        ].to_dict(orient="records")

    payload: dict[str, Any] = {
        "source": "siaf",
        "layer": "silver",
        "pipeline": "siaf_silver_pipeline",
        "status": result.status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),

        # Métricas de filas
        "bronze_files_processed": len(result.bronze_paths),
        "bronze_rows": result.bronze_rows,
        "rows_after_filter_nivel_gobierno_m": result.filtered_rows,
        "rows_filtered_out": result.bronze_rows - result.filtered_rows,
        "silver_rows": result.silver_rows,
        "duplicates_removed": result.duplicates_removed,

        # Métricas de columnas
        "bronze_columns": result.bronze_columns,
        "silver_columns": result.silver_columns,
        "derived_columns_added": 7,   # ubigeo_ejecutora, nombre_municipalidad_normalizado,
                                      # sk_tiempo_mensual, anio_mes_key, ano_particion,
                                      # flag_tipo_transaccion, flag_monto_pia_activo
        "technical_columns_added": 5, # source_system, source_dataset,
                                      # silver_processed_at, _audit_loaded_at, record_hash

        # Calidad de la transformación
        "conversion_errors": result.conversion_errors,
        "nulls_before": result.nulls_before,
        "nulls_after": result.nulls_after,
        "columns_with_conversion_errors": error_cols,

        # Notas del proceso
        "notes": result.notes,

        # Rutas de outputs
        "outputs": {
            "silver_parquet_dir": str(silver_root),
            "partition_key": "ano_particion",
            "summary_csv": str(reports_root / "siaf_silver_summary.csv"),
            "detail_csv": str(reports_root / "siaf_silver_detail.csv"),
            "data_dictionary_csv": str(reports_root / "siaf_silver_data_dictionary.csv"),
            "dashboard_html": str(reports_root / "siaf_silver_dashboard.html"),
            "ingresos_html": str(reports_root / "ingresos_silver.html"),
        },

        # Resumen ejecutivo en el JSON (útil para monitoreo automatizado)
        "summary": summary_df.to_dict(orient="records") if not summary_df.empty else [],
    }

    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return audit_path