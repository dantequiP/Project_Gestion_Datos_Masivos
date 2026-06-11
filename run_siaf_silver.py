"""
run_siaf_silver.py

Punto de entrada para el pipeline Silver de SIAF Ingresos.

Uso desde Docker:
    docker exec --user jovyan -it gdm_pyspark_siaf spark-submit run_siaf_silver.py

Outputs generados:
    data/silver/siaf/
        (Parquet particionado por ano_particion)
    reports/silver/siaf/
        siaf_silver_summary.csv
        siaf_silver_detail.csv
        siaf_silver_data_dictionary.csv
        siaf_silver_dashboard.html
        ingresos_silver.html
    data/audit/silver/year=YYYY/month=MM/day=DD/
        siaf_silver_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path

from app.pipelines.silver.siaf_silver_pipeline import run_siaf_silver_pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


def main() -> None:
    print("[INFO] ============================================================")
    print("[INFO]  Pipeline: Silver — SIAF Ingresos")
    print("[INFO]  Bronze conserva → Silver estandariza → Gold modela")
    print("[INFO] ============================================================")

    try:
        outputs = run_siaf_silver_pipeline()

        print("[INFO] ── Archivos generados ─────────────────────────────────")
        for key, path in outputs.items():
            print(f"[INFO]   {key}: {path}")
        print("[INFO] Silver SIAF completado exitosamente.")

    except Exception as exc:
        logger.error("Error fatal en Silver SIAF: %s", exc, exc_info=True)
        print(f"[ERROR] Error fatal: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()