"""
run_siaf_schema_validation.py

Punto de entrada para la validación estructural de SIAF Ingresos.

Uso desde Docker:
    docker exec -it gdm_pyspark_siaf python run_siaf_schema_validation.py

Responsabilidades:
    - Inicializar SparkSession con configuración adecuada para el volumen de SIAF.
    - Llamar al pipeline orquestador.
    - Imprimir resumen de salidas al finalizar.

Esta etapa NO modifica ningún dato Bronze.
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession

from app.pipelines.schema_validation.siaf_schema_validation_pipeline import (
    run_siaf_schema_validation_pipeline,
)
from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


def build_spark_session() -> SparkSession:
    """
    Construye la SparkSession para el pipeline SIAF.

    Configuración ajustada para:
    - Contenedor Docker con bind mount en /app.
    - Millones de registros históricos (2012-2026).
    - Schema validation no requiere shuffles pesados, pero sí lectura
      eficiente de múltiples Parquets.
    """
    print("[INFO] Inicializando SparkSession para Schema Validation SIAF...")

    spark = (
        SparkSession.builder
        .appName("SIAF_SchemaValidation")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.mergeSchema", "true")   # une schemas distintos entre años
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")   # evita spam en Docker
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    print("[INFO] SparkSession inicializada correctamente.")
    return spark


def main() -> None:
    print("[INFO] ============================================================")
    print("[INFO]  Pipeline: Schema Validation — SIAF Ingresos")
    print("[INFO]  Capa evaluada: Bronze  |  Sin modificación de datos")
    print("[INFO] ============================================================")

    spark = build_spark_session()

    try:
        outputs = run_siaf_schema_validation_pipeline(spark=spark)

        print("[INFO] ── Archivos generados ─────────────────────────────────")
        for key, path in outputs.items():
            print(f"[INFO]   {key}: {path}")
        print("[INFO] Schema Validation SIAF completada exitosamente.")

    except Exception as exc:
        logger.error("Error fatal en Schema Validation SIAF: %s", exc, exc_info=True)
        print(f"[ERROR] Error fatal: {exc}")
        sys.exit(1)

    finally:
        spark.stop()
        print("[INFO] SparkSession detenida.")


if __name__ == "__main__":
    main()