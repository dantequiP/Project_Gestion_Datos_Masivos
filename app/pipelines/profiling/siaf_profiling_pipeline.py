"""
siaf_profiling_pipeline.py

Orquestador del pipeline de Data Profiling para SIAF Ingresos.

Responsabilidades:
  - Descubrir los Parquet Bronze bajo data/bronze/siaf/ con glob,
    OMITIENDO rutas que contengan '_diario' (regla de negocio crítica).
  - Cargar cada Parquet con PySpark y delegarlo al SiafDatasetProfiler.
  - Recolectar todos los resultados y generar dashboard + auditoría.

Regla de ingesta (CRÍTICA):
  Se omiten rutas con '_diario' para no duplicar millones de registros
  que ya están consolidados en los archivos mensuales/anuales.

No modifica datos Bronze.
"""

from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession

from app.profiling.siaf_profiler import run_siaf_profiling
from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))


def _discover_parquet_paths(bronze_root: Path) -> list[Path]:
    """
    Descubre Parquets en data/bronze/siaf/** descartando '_diario'.

    Regla de negocio:
      Los archivos _diario son subconjuntos de los mensuales/anuales.
      Incluirlos duplicaría millones de registros presupuestales.

    Estrategia de deduplicación:
      Por cada subcarpeta de dataset (ej: 2024_ingreso), se toma el
      Parquet ubicado directamente en esa carpeta (no dentro de particiones
      year=/month=/day=) para evitar doble conteo.
    """
    siaf_root = bronze_root / "siaf"
    if not siaf_root.exists():
        logger.warning("No existe el directorio Bronze SIAF: %s", siaf_root)
        return []

    # Búsqueda recursiva de todos los Parquets
    todas: list[str] = glob.glob(str(siaf_root / "**" / "*.parquet"), recursive=True)

    # FILTRO CRÍTICO — excluir '_diario'
    filtradas: list[str] = [r for r in todas if "_diario" not in r.lower()]

    excluidas = len(todas) - len(filtradas)
    if excluidas > 0:
        print(f"[INFO] Rutas excluidas por filtro '_diario': {excluidas}")
        logger.info("Rutas excluidas por filtro '_diario': %d", excluidas)

    # Deduplicar: un Parquet por subcarpeta de dataset
    # Preferencia: archivos en la raíz del dataset (no particionados)
    dataset_best: dict[str, Path] = {}
    for ruta_str in sorted(filtradas):
        ruta = Path(ruta_str)
        try:
            partes = ruta.relative_to(siaf_root).parts
            dataset_key = partes[0]  # ej: "2024_ingreso"
        except ValueError:
            continue

        esta_en_raiz = len(partes) == 2  # siaf/<dataset>/<archivo>.parquet
        if dataset_key not in dataset_best:
            dataset_best[dataset_key] = ruta
        elif esta_en_raiz:
            # Archivo en raíz tiene prioridad sobre los de sub-particiones
            dataset_best[dataset_key] = ruta

    resultado = sorted(dataset_best.values(), key=lambda p: p.name)

    print(f"[INFO] Parquets SIAF a perfilar: {len(resultado)} "
          f"| totales={len(todas)} | excluidos_diario={excluidas}")
    for p in resultado:
        print(f"[INFO]   -> {p}")

    return resultado


def run_siaf_profiling_pipeline(
    spark: SparkSession,
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/profiling/siaf"),
    audit_root: Path = Path("data/audit/profiling"),
) -> dict[str, str]:
    """
    Orquesta el Data Profiling completo SIAF.

    Flujo:
      1. Descubre Parquets Bronze (sin '_diario').
      2. Para cada Parquet: ejecuta las 7 secciones de profiling con PySpark.
      3. Genera reportes HTML individuales, dashboard y CSVs por sección.
      4. Escribe auditoría JSON particionada.

    Returns:
        dict con rutas de los artefactos generados.
    """
    print("[INFO] === Inicio pipeline Data Profiling SIAF ===")
    logger.info("=== Inicio pipeline Data Profiling SIAF ===")
    started_at = datetime.now()

    parquet_paths = _discover_parquet_paths(Path(bronze_root))

    if not parquet_paths:
        msg = f"No se encontraron Parquets Bronze bajo {bronze_root}/siaf/"
        logger.warning(msg)
        print(f"[WARNING] {msg}")
        return {}

    outputs = run_siaf_profiling(
        spark=spark,
        parquet_paths=parquet_paths,
        bronze_root=Path(bronze_root),
        reports_root=Path(reports_root),
        audit_root=Path(audit_root),
    )

    duration = (datetime.now() - started_at).total_seconds()
    print(f"[INFO] === Pipeline Data Profiling SIAF completado en {duration:.1f}s ===")
    logger.info("=== Pipeline Data Profiling SIAF completado en %.1fs ===", duration)

    return outputs


if __name__ == "__main__":
    from pyspark.sql import SparkSession as _SparkSession
    _spark = (
        _SparkSession.builder
        .appName("SIAF_Profiling_Direct")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    _spark.sparkContext.setLogLevel("ERROR")
    run_siaf_profiling_pipeline(spark=_spark)
    _spark.stop()