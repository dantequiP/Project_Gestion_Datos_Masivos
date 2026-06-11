"""
siaf_dataset_profiler.py

Clase SiafDatasetProfiler: genera estadísticas descriptivas masivas sobre
los Parquet Bronze de SIAF usando PySpark en una sola pasada de .agg().

Estrategia de rendimiento:
    - Construye TODAS las expresiones de agregación en una lista Python.
    - Las ejecuta en un único df_spark.agg(*expresiones).collect()[0].
    - Esto evita múltiples jobs Spark y es la estrategia óptima para
      datasets de millones de filas en un contenedor Docker local.
    - El resultado único de .collect()[0] se transforma a Pandas vertical.

Métricas calculadas por columna:
    - Nulos (count de nulos reales)
    - Porcentaje_Nulos
    - Cardinalidad_Aprox (approx_count_distinct — HyperLogLog, ~5% error)
    - Min (solo columnas numéricas y de texto donde tiene sentido)
    - Max

Métricas adicionales del dataset completo:
    - Total filas (count() — un job separado, necesario como denominador)
    - Total columnas
    - Columnas con nulos

NO modifica datos Bronze.
NO filtra filas (regla de negocio: sin filtro por municipalidades).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)


class SiafDatasetProfiler:
    """
    Profiler de datos SIAF basado en PySpark.

    Toda la computación masiva ocurre dentro de Spark.
    El resultado final se entrega como Pandas DataFrame para
    compatibilidad con las clases generadoras de reportes HTML.
    """

    # Precisión de approx_count_distinct (HyperLogLog).
    # 0.05 = ~5% error, buen balance entre precisión y memoria.
    APPROX_PRECISION: float = 0.05

    def profile(self, df_spark: DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Ejecuta el profiling completo del DataFrame Spark en una sola pasada.

        Parameters
        ----------
        df_spark : DataFrame
            DataFrame PySpark con todos los Parquets Bronze SIAF cargados.
            Puede tener esquemas combinados de múltiples años.

        Returns
        -------
        profile_df : pd.DataFrame
            DataFrame Pandas vertical con columnas:
            Columna | Tipo_Spark | Nulos | Porcentaje_Nulos |
            Cardinalidad_Aprox | Min | Max

        summary : dict
            Métricas globales del dataset.
        """
        print("[INFO] Iniciando profiling SIAF con PySpark...")
        print(f"[INFO] Columnas a perfilar: {len(df_spark.columns)}")

        # ── Conteo total de filas (job separado, necesario como denominador) ──
        print("[INFO] Contando filas totales (puede tomar unos segundos)...")
        total_rows: int = df_spark.count()
        print(f"[INFO] Total filas Bronze SIAF: {total_rows:,}")

        if total_rows == 0:
            raise ValueError("El DataFrame Bronze SIAF está vacío. No hay datos para perfilar.")

        columnas = df_spark.columns
        schema_map = {f.name: f.dataType for f in df_spark.schema.fields}

        # ── Construcción de expresiones en una sola lista ─────────────────────
        # Esto es el núcleo del profiler: una expresión por métrica por columna,
        # todo ejecutado en un único job Spark con .agg(*expresiones).
        expresiones = []
        alias_map: list[dict[str, str]] = []   # mapeo de aliases para extraer resultados

        for col in columnas:
            dtype = schema_map.get(col)
            safe_col = F.col(f"`{col}`")   # backticks por si el nombre tiene espacios

            # Nulos reales (nulls nativos del Parquet)
            null_alias = f"__nulos__{col}"
            expresiones.append(
                F.count(F.when(safe_col.isNull(), 1)).alias(null_alias)
            )

            # Cardinalidad aproximada (HyperLogLog — mucho más eficiente que countDistinct)
            card_alias = f"__card__{col}"
            expresiones.append(
                F.approx_count_distinct(safe_col, rsd=self.APPROX_PRECISION).alias(card_alias)
            )

            # Min / Max — solo para tipos que tienen orden natural
            is_numeric = isinstance(dtype, (IntegerType, LongType, ShortType, FloatType, DoubleType))
            is_string = isinstance(dtype, StringType)
            is_timestamp = isinstance(dtype, TimestampType)

            if is_numeric or is_string or is_timestamp:
                min_alias = f"__min__{col}"
                max_alias = f"__max__{col}"
                expresiones.append(F.min(safe_col).cast("string").alias(min_alias))
                expresiones.append(F.max(safe_col).cast("string").alias(max_alias))
            else:
                min_alias = None
                max_alias = None

            alias_map.append({
                "columna": col,
                "null_alias": null_alias,
                "card_alias": card_alias,
                "min_alias": min_alias,
                "max_alias": max_alias,
                "dtype": str(dtype),
            })

        # ── Ejecución: UNA SOLA PASADA sobre todos los datos ─────────────────
        print(f"[INFO] Ejecutando {len(expresiones)} expresiones de agregación en una sola pasada...")
        try:
            resultado = df_spark.agg(*expresiones).collect()[0]
        except Exception as exc:
            raise RuntimeError(f"Error en la agregación PySpark del profiling SIAF: {exc}") from exc

        print("[INFO] Agregaciones completadas. Construyendo DataFrame Pandas...")

        # ── Transformación a Pandas vertical ─────────────────────────────────
        rows: list[dict[str, Any]] = []
        for meta in alias_map:
            col = meta["columna"]

            nulos = int(resultado[meta["null_alias"]] or 0)
            cardinalidad = int(resultado[meta["card_alias"]] or 0)
            pct_nulos = round((nulos / total_rows) * 100, 4) if total_rows > 0 else 0.0

            min_val = ""
            max_val = ""
            if meta["min_alias"] is not None:
                min_val = str(resultado[meta["min_alias"]] or "")
                max_val = str(resultado[meta["max_alias"]] or "")

            rows.append({
                "Columna": col,
                "Tipo_Spark": meta["dtype"],
                "Nulos": nulos,
                "Porcentaje_Nulos": pct_nulos,
                "Cardinalidad_Aprox": cardinalidad,
                "Min": min_val,
                "Max": max_val,
            })

        profile_df = pd.DataFrame(rows)

        # ── Resumen global ────────────────────────────────────────────────────
        cols_con_nulos = int((profile_df["Nulos"] > 0).sum())
        cols_sin_nulos = len(columnas) - cols_con_nulos
        pct_completitud = round((cols_sin_nulos / len(columnas)) * 100, 2) if columnas else 0.0

        summary: dict[str, Any] = {
            "fuente": "SIAF",
            "dataset": "ingresos",
            "total_filas": total_rows,
            "total_columnas": len(columnas),
            "columnas_con_nulos": cols_con_nulos,
            "columnas_sin_nulos": cols_sin_nulos,
            "completitud_columnas_pct": pct_completitud,
            "cardinalidad_promedio": int(profile_df["Cardinalidad_Aprox"].mean()),
            "precision_approx_count_distinct": f"~{int(self.APPROX_PRECISION * 100)}% error (HyperLogLog)",
            "nota": "Min/Max solo para columnas numéricas, string y timestamp.",
        }

        print(f"[INFO] Profiling completado:")
        print(f"[INFO]   Filas: {total_rows:,}")
        print(f"[INFO]   Columnas: {len(columnas)}")
        print(f"[INFO]   Columnas con nulos: {cols_con_nulos}")
        print(f"[INFO]   Completitud columnas: {pct_completitud}%")

        return profile_df, summary