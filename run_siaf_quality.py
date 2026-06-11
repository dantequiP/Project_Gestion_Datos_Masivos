"""
run_siaf_quality.py

Punto de entrada para la evaluación de calidad de SIAF Ingresos.

Uso desde Docker:
    docker exec -it gdm_pyspark_siaf python run_siaf_quality.py

Precondición: el pipeline Bronze debe haber corrido primero.
Esta etapa NO modifica datos Bronze.

Archivos generados en reports/quality/siaf/ y data/quality/siaf/.
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession

from app.pipelines.quality.siaf_quality_pipeline import run_siaf_quality_pipeline
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


def build_spark_session() -> SparkSession:
    """
    SparkSession optimizada para quality sobre 10M+ filas.

    Más memoria que schema validation porque el checker hace
    múltiples pasadas (count, filter, groupBy, distinct).
    """
    logger.info("Inicializando SparkSession para Quality SIAF...")
    spark = (
        SparkSession.builder
        .appName("SIAF_Quality")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("SparkSession inicializada.")
    return spark


def main() -> None:
    logger.info("============================================================")
    logger.info(" Pipeline: Quality — SIAF Ingresos (Bronze)")
    logger.info(" 8 dimensiones | 60+ reglas | Sin modificación de datos")
    logger.info("============================================================")

    spark = build_spark_session()

    try:
        outputs = run_siaf_quality_pipeline(spark=spark)

        logger.info("── Archivos generados ─────────────────────────────────")
        for key, path in outputs.items():
            logger.info("  %s: %s", key, path)

        print("\n[OK] Quality SIAF completada exitosamente.")
        print("Archivos generados:")
        for key, path in outputs.items():
            print(f"  {key}: {path}")

    except FileNotFoundError as exc:
        logger.error("Parquets Bronze no encontrados: %s", exc)
        print(f"\n[ERROR] {exc}")
        print("Verifica que run_siaf_schema_validation.py haya pasado sin FALLA_CRITICA.")
        sys.exit(1)

    except Exception as exc:
        logger.error("Error fatal en Quality SIAF: %s", exc, exc_info=True)
        print(f"\n[ERROR] Error fatal: {exc}")
        sys.exit(1)

    finally:
        spark.stop()
        logger.info("SparkSession detenida.")


if __name__ == "__main__":
    main()