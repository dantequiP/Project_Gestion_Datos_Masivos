"""
run_siaf_profiling.py

Punto de entrada para el Data Profiling de SIAF Ingresos.

Uso desde Docker:
    docker exec -it gdm_pyspark_siaf python run_siaf_profiling.py

Responsabilidades:
    - Inicializar SparkSession con configuración para datasets grandes.
    - Llamar al pipeline orquestador de profiling.
    - Imprimir resumen de salidas al finalizar.

El profiling calcula estadísticas descriptivas masivas en una sola pasada
de agregaciones PySpark. NO modifica datos Bronze.
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession

from app.pipelines.profiling.siaf_profiling_pipeline import (
    run_siaf_profiling_pipeline,
)
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


def build_spark_session() -> SparkSession:
    """
    SparkSession configurada para profiling masivo de SIAF.

    El profiling requiere más memoria que el schema validation porque
    ejecuta múltiples funciones de agregación (.agg) sobre todos los
    registros históricos en una sola pasada.
    """
    print("[INFO] Inicializando SparkSession para Profiling SIAF...")

    spark = (
        SparkSession.builder
        .appName("SIAF_Profiling")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
        # approx_count_distinct usa HyperLogLog; este valor controla precisión vs memoria
        .config("spark.sql.statistics.ndv.maxError", "0.05")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    print("[INFO] SparkSession inicializada correctamente.")
    return spark


def main() -> None:
    print("[INFO] ============================================================")
    print("[INFO]  Pipeline: Data Profiling — SIAF Ingresos")
    print("[INFO]  Capa evaluada: Bronze  |  Sin modificación de datos")
    print("[INFO]  Motor: PySpark (agregaciones en una sola pasada)")
    print("[INFO] ============================================================")

    spark = build_spark_session()

    try:
        outputs = run_siaf_profiling_pipeline(spark=spark)

        print("[INFO] ── Archivos generados ─────────────────────────────────")
        for key, path in outputs.items():
            print(f"[INFO]   {key}: {path}")
        print("[INFO] Profiling SIAF completado exitosamente.")

    except Exception as exc:
        logger.error("Error fatal en Profiling SIAF: %s", exc, exc_info=True)
        print(f"[ERROR] Error fatal: {exc}")
        sys.exit(1)

    finally:
        spark.stop()
        print("[INFO] SparkSession detenida.")


if __name__ == "__main__":
    main()