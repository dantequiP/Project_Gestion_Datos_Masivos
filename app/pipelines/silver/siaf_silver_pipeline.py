"""
siaf_silver_pipeline.py
Pipeline Silver SIAF — orquestador principal.

Lee silver_rules_siaf.yaml, ejecuta el transformer PySpark,
genera reportes CSV + HTML, y escribe la auditoría JSON.

Flujo completo:
  1. Carga silver_rules_siaf.yaml
  2. Crea SparkSession
  3. SiafSilverTransformer.run() →
       filtering → cleaning → imputations → type_casting
       → derived_columns → audit_columns → deduplication
       → write Parquet Silver
  4. Guarda summary.csv, detail.csv, data_dictionary.csv
  5. Genera dashboard HTML + reporte por dataset
  6. Escribe auditoría JSON

Outputs generados:
  reports/silver/siaf/
    siaf_silver_summary.csv
    siaf_silver_detail.csv
    siaf_silver_data_dictionary.csv
    siaf_silver_dashboard.html
    ingresos_silver.html
  data/silver/siaf/
    (Parquet particionado por ano_particion)
  data/audit/silver/year=YYYY/month=MM/day=DD/
    siaf_silver_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

try:
    from app.utils.logger import get_logger
    logger = get_logger(__name__, log_dir=Path("logs"))
except Exception:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

from app.silver.siaf_silver_transformer import (
    SiafSilverTransformer,
    load_silver_rules,
    _build_spark,           # noqa: PLC2701 — utilidad interna compartida
)
from app.silver.siaf_silver_audit import write_siaf_silver_audit
from app.silver.siaf_silver_report import (
    write_dashboard_html,
    write_dataset_html,
)


CONFIG_PATH = Path("app/config/silver_rules_siaf.yaml")


def run_siaf_silver_pipeline() -> dict[str, str]:
    """
    Ejecuta el pipeline Silver completo para SIAF Ingresos.
    Retorna un dict con las rutas de todos los outputs generados.
    """
    started_at = datetime.now()
    logger.info("=" * 60)
    logger.info("Inicio del pipeline Silver SIAF")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("=" * 60)

    # ── Carga de configuración ────────────────────────────────
    config = load_silver_rules(CONFIG_PATH)
    logger.info("silver_rules_siaf.yaml cargado correctamente.")

    reports_root = Path(config.get("reports_base_path", "reports/silver/siaf"))
    silver_root  = Path(config.get("silver_base_path",  "data/silver/siaf"))
    audit_root   = Path(config.get("audit_base_path",   "data/audit/silver"))

    reports_root.mkdir(parents=True, exist_ok=True)
    silver_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    # ── SparkSession ──────────────────────────────────────────
    spark = _build_spark("SIAF Silver Pipeline")
    logger.info("SparkSession iniciada.")

    # ── Transformación ────────────────────────────────────────
    transformer = SiafSilverTransformer(config)

    try:
        result, summary_df, detail_df, dictionary_df = transformer.run(
            spark=spark,
            processed_at=started_at,
        )
    except Exception as exc:
        logger.exception("Error fatal en la transformación Silver SIAF: %s", exc)
        spark.stop()
        sys.exit(1)

    # ── Reportes CSV ──────────────────────────────────────────
    summary_csv    = reports_root / "siaf_silver_summary.csv"
    detail_csv     = reports_root / "siaf_silver_detail.csv"
    dictionary_csv = reports_root / "siaf_silver_data_dictionary.csv"

    summary_df.to_csv(summary_csv,    index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_csv,      index=False, encoding="utf-8-sig")
    dictionary_df.to_csv(dictionary_csv, index=False, encoding="utf-8-sig")

    logger.info("CSVs guardados en: %s", reports_root)

    # ── Reportes HTML ─────────────────────────────────────────
    dashboard_html = reports_root / "siaf_silver_dashboard.html"
    write_dashboard_html(summary_df, detail_df, dictionary_df, dashboard_html)
    logger.info("Dashboard HTML: %s", dashboard_html)

    # Reporte por dataset (solo 'ingresos' en SIAF)
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
    logger.info("Reporte HTML por dataset generado.")

    # ── Auditoría JSON ────────────────────────────────────────
    finished_at = datetime.now()
    audit_path = write_siaf_silver_audit(
        result=result,
        summary_df=summary_df,
        detail_df=detail_df,
        reports_root=reports_root,
        silver_root=silver_root,
        audit_root=audit_root,
        started_at=started_at,
        finished_at=finished_at,
    )
    logger.info("Auditoría JSON: %s", audit_path)

    # ── Resumen final ─────────────────────────────────────────
    duration = (finished_at - started_at).total_seconds()
    logger.info("=" * 60)
    logger.info("Pipeline Silver SIAF FINALIZADO")
    logger.info("  Status       : %s", result.status)
    logger.info("  Filas Bronze : %d", result.bronze_rows)
    logger.info("  Filas filtro : %d (NIVEL_GOBIERNO=M)", result.filtered_rows)
    logger.info("  Filas Silver : %d", result.silver_rows)
    logger.info("  Dupes elim.  : %d", result.duplicates_removed)
    logger.info("  Errores conv.: %d", result.conversion_errors)
    logger.info("  Duración     : %.1f s", duration)
    logger.info("=" * 60)

    spark.stop()

    return {
        "summary_csv":        str(summary_csv),
        "detail_csv":         str(detail_csv),
        "data_dictionary_csv": str(dictionary_csv),
        "dashboard_html":     str(dashboard_html),
        "reports_dir":        str(reports_root),
        "silver_dir":         str(silver_root),
        "audit_path":         str(audit_path),
        "silver_ingresos":    str(result.silver_path),
    }