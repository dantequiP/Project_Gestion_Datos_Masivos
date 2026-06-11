"""
siaf_profiler.py

Clase de Data Profiling para SIAF Ingresos con PySpark.

Análisis implementados (pensados para informar la capa Silver y Quality):
  1. Nulos y vacíos           — por columna, conteo y % (todas las columnas)
  2. Duplicados               — filas completamente duplicadas y duplicados por
                                clave de negocio (ANO_DOC + MES_DOC + SEC_EJEC +
                                FUENTE_FINANCIAMIENTO + ESPECIFICA_DET)
  3. Columnas categóricas     — cardinalidad, top-10 valores, distribución por
                                columnas identificadas como categoría en el YAML
  4. Variables temporales     — ANO_DOC / MES_DOC: rango, gaps de años, meses
                                sin datos, combinaciones presentes, evolución de
                                registros y montos por año
  5. Cardinalidad general     — approx_count_distinct (HyperLogLog) + clase
                                (identificador / alta / media / baja / binaria)
  6. Consistencia             — cruce de códigos vs nombres (PLIEGO/PLIEGO_NOMBRE,
                                EJECUTORA/EJECUTORA_NOMBRE, etc.)
  7. Valores negativos y cero — solo columnas de montos (MONTO_PIA, MONTO_PIM,
                                MONTO_RECAUDADO): conteo de negativos, ceros y
                                positivos con sus % respectivos

Regla de arquitectura:
  - PySpark hace TODO el procesamiento masivo: lecturas, .agg(), .filter(),
    .groupBy(), approx_count_distinct().
  - El resultado de cada sección se convierte a Pandas SOLO al final,
    para compatibilidad con los generadores de reportes HTML.

Esta etapa NO modifica datos Bronze y NO genera Silver.
"""

from __future__ import annotations

import html as _html_lib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDF
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))

# ──────────────────────────────────────────────────────────────────────────────
# Constantes del dominio SIAF
# ──────────────────────────────────────────────────────────────────────────────

# Columnas de montos presupuestales (análisis de negativos/ceros)
MONTO_COLS: list[str] = ["MONTO_PIA", "MONTO_PIM", "MONTO_RECAUDADO"]

# Columnas que deben ser categorías de baja cardinalidad
CATEGORICAL_COLS: list[str] = [
    "NIVEL_GOBIERNO", "NIVEL_GOBIERNO_NOMBRE",
    "SECTOR", "SECTOR_NOMBRE",
    "FUENTE_FINANCIAMIENTO", "FUENTE_FINANCIAMIENTO_NOMBRE",
    "RUBRO", "RUBRO_NOMBRE",
    "TIPO_RECURSO", "TIPO_RECURSO_NOMBRE",
    "GENERICA", "GENERICA_NOMBRE",
    "SUBGENERICA", "SUBGENERICA_NOMBRE",
    "SUBGENERICA_DET", "SUBGENERICA_DET_NOMBRE",
    "ESPECIFICA", "ESPECIFICA_NOMBRE",
    "ESPECIFICA_DET", "ESPECIFICA_DET_NOMBRE",
]

# Pares código → nombre para análisis de consistencia
CONSISTENCY_PAIRS: list[tuple[str, str]] = [
    ("PLIEGO", "PLIEGO_NOMBRE"),
    ("SEC_EJEC", "EJECUTORA_NOMBRE"),
    ("EJECUTORA", "EJECUTORA_NOMBRE"),
    ("DEPARTAMENTO_EJECUTORA", "DEPARTAMENTO_EJECUTORA_NOMBRE"),
    ("PROVINCIA_EJECUTORA", "PROVINCIA_EJECUTORA_NOMBRE"),
    ("DISTRITO_EJECUTORA", "DISTRITO_EJECUTORA_NOMBRE"),
    ("FUENTE_FINANCIAMIENTO", "FUENTE_FINANCIAMIENTO_NOMBRE"),
    ("RUBRO", "RUBRO_NOMBRE"),
    ("GENERICA", "GENERICA_NOMBRE"),
    ("SUBGENERICA", "SUBGENERICA_NOMBRE"),
    ("ESPECIFICA", "ESPECIFICA_NOMBRE"),
    ("ESPECIFICA_DET", "ESPECIFICA_DET_NOMBRE"),
]

# Clave de negocio para duplicados lógicos
BUSINESS_KEY: list[str] = [
    "ANO_DOC", "MES_DOC", "SEC_EJEC",
    "FUENTE_FINANCIAMIENTO", "ESPECIFICA_DET",
]

# Meses válidos
MESES_VALIDOS: set[int] = set(range(1, 13))

# Años esperados en el dataset histórico
ANO_MIN_ESPERADO: int = 2012
ANO_MAX_ESPERADO: int = 2026


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de rutas
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfilingPaths:
    bronze_root: Path = Path("data/bronze")
    reports_root: Path = Path("reports/profiling/siaf")
    audit_root: Path = Path("data/audit/profiling")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────────────────────

def _classify_cardinality(unique_count: int, row_count: int) -> str:
    """Clasifica la cardinalidad de una columna en función del ratio único/total."""
    if row_count == 0:
        return "sin_datos"
    pct = (unique_count / row_count) * 100
    if unique_count <= 2:
        return "binaria"
    if pct > 90:
        return "identificador"
    if pct > 50:
        return "alta"
    if pct > 5:
        return "media"
    return "baja_categorica"


def _safe_collect_value(row: Any, key: str, default: Any = None) -> Any:
    """Extrae un valor de una Row de Spark de forma segura."""
    try:
        v = getattr(row, key, default)
        return v if v is not None else default
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal de profiling
# ──────────────────────────────────────────────────────────────────────────────

class SiafDatasetProfiler:
    """
    Ejecuta un profiling exhaustivo sobre un DataFrame SIAF cargado con PySpark.

    El profiling se organiza en 7 secciones independientes. Cada sección devuelve
    un Pandas DataFrame listo para ser renderizado en el reporte HTML.
    """

    def __init__(self, spark: SparkSession, paths: ProfilingPaths) -> None:
        self.spark = spark
        self.paths = paths

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 1 — Nulos y vacíos (PySpark .agg en una sola pasada)
    # ──────────────────────────────────────────────────────────────────────────

    def profile_nulls(self, df: SparkDF, dataset: str) -> pd.DataFrame:
        """
        Calcula nulos, vacíos y faltantes-like por columna en una sola pasada .agg().

        PySpark construye una lista de expresiones y las ejecuta en un único job,
        lo que es eficiente para datasets con millones de filas.
        """
        print(f"[INFO]   [1/7] Analizando nulos y vacíos...")
        total_rows = df.count()

        expresiones = []
        for col in df.columns:
            expresiones += [
                F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias(f"{col}__nulls"),
                F.sum(F.when(
                    F.col(col).isNull() |
                    (F.trim(F.col(col).cast("string")) == "") |
                    F.trim(F.col(col).cast("string")).isin("nan", "NaN", "None", "NULL", "null", "<NA>"),
                    1
                ).otherwise(0)).alias(f"{col}__missing_like"),
            ]

        row = df.agg(*expresiones).collect()[0]

        records = []
        for col in df.columns:
            null_count = int(_safe_collect_value(row, f"{col}__nulls", 0))
            missing_count = int(_safe_collect_value(row, f"{col}__missing_like", 0))
            records.append({
                "dataset": dataset,
                "columna": col,
                "total_filas": total_rows,
                "nulos_reales": null_count,
                "pct_nulos": round(null_count / total_rows * 100, 4) if total_rows else 0,
                "faltantes_like": missing_count,
                "pct_faltantes": round(missing_count / total_rows * 100, 4) if total_rows else 0,
                "completitud_pct": round((1 - missing_count / total_rows) * 100, 4) if total_rows else 0,
                "alerta": "REVISAR" if missing_count / total_rows > 0.05 else "OK" if total_rows else "SIN_DATOS",
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 2 — Duplicados
    # ──────────────────────────────────────────────────────────────────────────

    def profile_duplicates(self, df: SparkDF, dataset: str) -> pd.DataFrame:
        """
        Detecta duplicados exactos (todas las columnas) y duplicados lógicos
        por clave de negocio. Usa PySpark groupBy + count para el conteo masivo.
        """
        print(f"[INFO]   [2/7] Analizando duplicados...")
        total_rows: int = df.count()
        distinct_rows: int = df.distinct().count()
        exact_dupes: int = total_rows - distinct_rows

        # Duplicados por clave de negocio (solo si todas las columnas existen)
        bk_available = [c for c in BUSINESS_KEY if c in df.columns]
        bk_dupes = 0
        bk_groups_with_dupes = 0
        if len(bk_available) == len(BUSINESS_KEY):
            bk_df = (
                df.groupBy(*bk_available)
                .agg(F.count("*").alias("cnt"))
                .filter(F.col("cnt") > 1)
            )
            bk_agg = bk_df.agg(
                F.sum("cnt").alias("total_extra_rows"),
                F.count("*").alias("groups_with_dupes"),
            ).collect()[0]
            bk_dupes = int(_safe_collect_value(bk_agg, "total_extra_rows", 0) or 0)
            bk_groups_with_dupes = int(_safe_collect_value(bk_agg, "groups_with_dupes", 0) or 0)

        records = [
            {
                "dataset": dataset,
                "tipo_duplicado": "Filas exactas duplicadas",
                "conteo": exact_dupes,
                "pct": round(exact_dupes / total_rows * 100, 4) if total_rows else 0,
                "descripcion": "Filas con todos los campos idénticos",
                "alerta": "REVISAR" if exact_dupes > 0 else "OK",
            },
            {
                "dataset": dataset,
                "tipo_duplicado": "Duplicados por clave de negocio",
                "conteo": bk_dupes,
                "pct": round(bk_dupes / total_rows * 100, 4) if total_rows else 0,
                "descripcion": f"Clave: {' + '.join(bk_available)} | grupos afectados: {bk_groups_with_dupes}",
                "alerta": "REVISAR" if bk_dupes > 0 else "OK",
            },
        ]
        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 3 — Columnas categóricas
    # ──────────────────────────────────────────────────────────────────────────

    def profile_categoricals(self, df: SparkDF, dataset: str) -> pd.DataFrame:
        """
        Para cada columna categórica del dominio SIAF calcula:
          - Cardinalidad real (distinct count con PySpark)
          - Top-10 valores más frecuentes con su conteo (groupBy + count)
          - % que representan los top-10 sobre el total
        """
        print(f"[INFO]   [3/7] Analizando columnas categóricas...")
        total_rows: int = df.count()
        cols_presentes = [c for c in CATEGORICAL_COLS if c in df.columns]

        # Cardinalidad en una sola pasada .agg()
        card_exprs = [
            F.approx_count_distinct(F.col(c), rsd=0.05).alias(c)
            for c in cols_presentes
        ]
        card_row = df.agg(*card_exprs).collect()[0] if card_exprs else None

        records = []
        for col in cols_presentes:
            card = int(_safe_collect_value(card_row, col, 0)) if card_row else 0

            # Top-10 valores (job separado por columna, pero solo categóricas)
            top_rows = (
                df.filter(F.col(col).isNotNull())
                .groupBy(col)
                .agg(F.count("*").alias("freq"))
                .orderBy(F.col("freq").desc())
                .limit(10)
                .collect()
            )
            top_dict = {str(r[col]): int(r["freq"]) for r in top_rows}
            top_pct = round(sum(top_dict.values()) / total_rows * 100, 2) if total_rows else 0

            records.append({
                "dataset": dataset,
                "columna": col,
                "cardinalidad_aprox": card,
                "clase_cardinalidad": _classify_cardinality(card, total_rows),
                "top10_valores": json.dumps(top_dict, ensure_ascii=False),
                "cobertura_top10_pct": top_pct,
                "alerta": "REVISAR" if card > 500 else "OK",
                "nota": "Alta cardinalidad inesperada para columna categórica" if card > 500 else "",
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 4 — Variables temporales (ANO_DOC, MES_DOC)
    # ──────────────────────────────────────────────────────────────────────────

    def profile_temporal(self, df: SparkDF, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Análisis temporal para ANO_DOC y MES_DOC:
          - Rango, valores únicos, gaps de años faltantes
          - Meses sin datos (1-12)
          - Evolución de registros y monto_recaudado por año (groupBy PySpark)
          - Combinaciones ANO_DOC x MES_DOC presentes

        Returns:
            (resumen_temporal_df, evolucion_anual_df)
        """
        print(f"[INFO]   [4/7] Analizando variables temporales...")

        has_ano = "ANO_DOC" in df.columns
        has_mes = "MES_DOC" in df.columns
        has_monto = "MONTO_RECAUDADO" in df.columns

        resumen_records: list[dict] = []

        if has_ano:
            ano_stats = df.agg(
                F.min(F.col("ANO_DOC").cast(LongType())).alias("ano_min"),
                F.max(F.col("ANO_DOC").cast(LongType())).alias("ano_max"),
                F.approx_count_distinct("ANO_DOC").alias("anos_distintos"),
                F.sum(F.when(F.col("ANO_DOC").isNull(), 1).otherwise(0)).alias("nulos"),
            ).collect()[0]

            ano_min = int(_safe_collect_value(ano_stats, "ano_min", 0) or 0)
            ano_max = int(_safe_collect_value(ano_stats, "ano_max", 0) or 0)
            anos_distintos = int(_safe_collect_value(ano_stats, "anos_distintos", 0) or 0)
            anos_nulos = int(_safe_collect_value(ano_stats, "nulos", 0) or 0)

            # Años presentes reales
            anos_presentes = {
                int(r["ANO_DOC"])
                for r in df.select("ANO_DOC").distinct().collect()
                if r["ANO_DOC"] is not None
            }
            rango_esperado = set(range(ano_min, ano_max + 1)) if ano_min and ano_max else set()
            gaps = sorted(rango_esperado - anos_presentes)
            fuera_rango = sorted(
                a for a in anos_presentes
                if a < ANO_MIN_ESPERADO or a > ANO_MAX_ESPERADO
            )

            resumen_records.append({
                "dataset": dataset, "variable": "ANO_DOC",
                "valor_minimo": ano_min, "valor_maximo": ano_max,
                "valores_distintos": anos_distintos,
                "nulos": anos_nulos,
                "gaps_en_rango": str(gaps) if gaps else "ninguno",
                "valores_fuera_rango_esperado": str(fuera_rango) if fuera_rango else "ninguno",
                "alerta": "REVISAR" if gaps or fuera_rango or anos_nulos > 0 else "OK",
            })

        if has_mes:
            mes_stats = df.agg(
                F.min(F.col("MES_DOC").cast(LongType())).alias("mes_min"),
                F.max(F.col("MES_DOC").cast(LongType())).alias("mes_max"),
                F.approx_count_distinct("MES_DOC").alias("meses_distintos"),
                F.sum(F.when(F.col("MES_DOC").isNull(), 1).otherwise(0)).alias("nulos"),
            ).collect()[0]

            mes_min = int(_safe_collect_value(mes_stats, "mes_min", 0) or 0)
            mes_max = int(_safe_collect_value(mes_stats, "mes_max", 0) or 0)
            meses_distintos = int(_safe_collect_value(mes_stats, "meses_distintos", 0) or 0)
            mes_nulos = int(_safe_collect_value(mes_stats, "nulos", 0) or 0)

            meses_presentes = {
                int(r["MES_DOC"])
                for r in df.select("MES_DOC").distinct().collect()
                if r["MES_DOC"] is not None
            }
            meses_faltantes = sorted(MESES_VALIDOS - meses_presentes)
            meses_invalidos = sorted(m for m in meses_presentes if m not in MESES_VALIDOS)

            resumen_records.append({
                "dataset": dataset, "variable": "MES_DOC",
                "valor_minimo": mes_min, "valor_maximo": mes_max,
                "valores_distintos": meses_distintos,
                "nulos": mes_nulos,
                "gaps_en_rango": str(meses_faltantes) if meses_faltantes else "ninguno",
                "valores_fuera_rango_esperado": str(meses_invalidos) if meses_invalidos else "ninguno",
                "alerta": "REVISAR" if meses_faltantes or meses_invalidos or mes_nulos > 0 else "OK",
            })

        # Evolución anual (groupBy PySpark)
        evolucion_records: list[dict] = []
        if has_ano:
            agg_exprs = [F.count("*").alias("registros")]
            if has_monto:
                agg_exprs += [
                    F.sum(F.col("MONTO_RECAUDADO").cast(DoubleType())).alias("sum_recaudado"),
                    F.avg(F.col("MONTO_RECAUDADO").cast(DoubleType())).alias("avg_recaudado"),
                ]
            if has_mes:
                agg_exprs.append(F.approx_count_distinct("MES_DOC").alias("meses_con_datos"))

            evol_rows = (
                df.groupBy(F.col("ANO_DOC").cast(LongType()).alias("ano"))
                .agg(*agg_exprs)
                .orderBy("ano")
                .collect()
            )
            for r in evol_rows:
                rec: dict = {
                    "dataset": dataset,
                    "ano": int(_safe_collect_value(r, "ano", 0) or 0),
                    "registros": int(_safe_collect_value(r, "registros", 0) or 0),
                }
                if has_monto:
                    rec["sum_monto_recaudado"] = round(float(_safe_collect_value(r, "sum_recaudado", 0) or 0), 2)
                    rec["avg_monto_recaudado"] = round(float(_safe_collect_value(r, "avg_recaudado", 0) or 0), 2)
                if has_mes:
                    rec["meses_con_datos"] = int(_safe_collect_value(r, "meses_con_datos", 0) or 0)
                evolucion_records.append(rec)

        return pd.DataFrame(resumen_records), pd.DataFrame(evolucion_records)

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 5 — Cardinalidad general (todas las columnas)
    # ──────────────────────────────────────────────────────────────────────────

    def profile_cardinality(self, df: SparkDF, dataset: str) -> pd.DataFrame:
        """
        Cardinalidad aproximada para TODAS las columnas en una sola pasada .agg().

        Usa approx_count_distinct (HyperLogLog, rsd=0.05) de PySpark,
        que es O(n) pero con uso de memoria muy bajo y distribución eficiente.
        """
        print(f"[INFO]   [5/7] Calculando cardinalidad...")
        total_rows: int = df.count()

        exprs = [
            F.approx_count_distinct(F.col(c), rsd=0.05).alias(c)
            for c in df.columns
        ]
        card_row = df.agg(*exprs).collect()[0]

        records = []
        for col in df.columns:
            card = int(_safe_collect_value(card_row, col, 0) or 0)
            records.append({
                "dataset": dataset,
                "columna": col,
                "cardinalidad_aprox": card,
                "pct_unicidad": round(card / total_rows * 100, 4) if total_rows else 0,
                "clase": _classify_cardinality(card, total_rows),
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 6 — Consistencia código → nombre
    # ──────────────────────────────────────────────────────────────────────────

    def profile_consistency(self, df: SparkDF, dataset: str) -> pd.DataFrame:
        """
        Para cada par (CODIGO, NOMBRE) del dominio SIAF verifica:
          - Cuántos valores distintos tiene CODIGO y NOMBRE
          - Si hay códigos que mapean a más de un nombre (1→N: inconsistente)
          - Si hay nombres que mapean a más de un código (N→1: puede ser válido)

        Todo con PySpark groupBy + count.
        """
        print(f"[INFO]   [6/7] Verificando consistencia código-nombre...")
        records = []

        for cod_col, nom_col in CONSISTENCY_PAIRS:
            if cod_col not in df.columns or nom_col not in df.columns:
                continue

            # Cardinalidad de ambas columnas en una sola pasada
            stats = df.agg(
                F.approx_count_distinct(cod_col).alias("card_cod"),
                F.approx_count_distinct(nom_col).alias("card_nom"),
            ).collect()[0]
            card_cod = int(_safe_collect_value(stats, "card_cod", 0) or 0)
            card_nom = int(_safe_collect_value(stats, "card_nom", 0) or 0)

            # Códigos que mapean a más de 1 nombre (inconsistencia real)
            cod_multi_nom = (
                df.filter(F.col(cod_col).isNotNull() & F.col(nom_col).isNotNull())
                .groupBy(cod_col)
                .agg(F.approx_count_distinct(nom_col).alias("n_nombres"))
                .filter(F.col("n_nombres") > 1)
                .count()
            )

            records.append({
                "dataset": dataset,
                "columna_codigo": cod_col,
                "columna_nombre": nom_col,
                "cardinalidad_codigo": card_cod,
                "cardinalidad_nombre": card_nom,
                "codigos_con_multiples_nombres": int(cod_multi_nom),
                "consistencia": "INCONSISTENTE" if cod_multi_nom > 0 else "OK",
                "nota": (
                    f"{cod_multi_nom} código(s) tienen más de un nombre asociado"
                    if cod_multi_nom > 0 else "Relación 1→1 correcta"
                ),
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    # SECCIÓN 7 — Valores negativos y cero (montos)
    # ──────────────────────────────────────────────────────────────────────────

    def profile_monto_signs(self, df: SparkDF, dataset: str) -> pd.DataFrame:
        """
        Para MONTO_PIA, MONTO_PIM, MONTO_RECAUDADO calcula en una sola pasada:
          - Conteo y % de valores negativos
          - Conteo y % de ceros exactos
          - Conteo y % de positivos
          - Min, Max, Suma, Media
          - Alerta si hay negativos (inusual en ingresos presupuestados)
        """
        print(f"[INFO]   [7/7] Analizando signos en columnas de montos...")
        total_rows: int = df.count()
        cols_presentes = [c for c in MONTO_COLS if c in df.columns]

        # Construir todas las expresiones en una sola pasada .agg()
        exprs = []
        for col in cols_presentes:
            c = F.col(col).cast(DoubleType())
            exprs += [
                F.sum(F.when(c < 0, 1).otherwise(0)).alias(f"{col}__neg"),
                F.sum(F.when(c == 0, 1).otherwise(0)).alias(f"{col}__cero"),
                F.sum(F.when(c > 0, 1).otherwise(0)).alias(f"{col}__pos"),
                F.sum(F.when(c.isNull(), 1).otherwise(0)).alias(f"{col}__null"),
                F.min(c).alias(f"{col}__min"),
                F.max(c).alias(f"{col}__max"),
                F.sum(c).alias(f"{col}__sum"),
                F.avg(c).alias(f"{col}__avg"),
            ]

        row = df.agg(*exprs).collect()[0] if exprs else None

        records = []
        for col in cols_presentes:
            neg = int(_safe_collect_value(row, f"{col}__neg", 0) or 0)
            cero = int(_safe_collect_value(row, f"{col}__cero", 0) or 0)
            pos = int(_safe_collect_value(row, f"{col}__pos", 0) or 0)
            null = int(_safe_collect_value(row, f"{col}__null", 0) or 0)
            vmin = _safe_collect_value(row, f"{col}__min", None)
            vmax = _safe_collect_value(row, f"{col}__max", None)
            vsum = _safe_collect_value(row, f"{col}__sum", None)
            vavg = _safe_collect_value(row, f"{col}__avg", None)

            records.append({
                "dataset": dataset,
                "columna": col,
                "total_filas": total_rows,
                "negativos": neg,
                "pct_negativos": round(neg / total_rows * 100, 4) if total_rows else 0,
                "ceros": cero,
                "pct_ceros": round(cero / total_rows * 100, 4) if total_rows else 0,
                "positivos": pos,
                "pct_positivos": round(pos / total_rows * 100, 4) if total_rows else 0,
                "nulos": null,
                "valor_minimo": round(float(vmin), 2) if vmin is not None else None,
                "valor_maximo": round(float(vmax), 2) if vmax is not None else None,
                "suma_total": round(float(vsum), 2) if vsum is not None else None,
                "promedio": round(float(vavg), 4) if vavg is not None else None,
                "alerta_negativos": "REVISAR" if neg > 0 else "OK",
                "alerta_ceros": "REVISAR" if cero / total_rows > 0.30 else "OK" if total_rows else "SIN_DATOS",
                "nota": (
                    f"{neg:,} valores negativos detectados — ingresos negativos son inusuales"
                    if neg > 0 else "Sin valores negativos"
                ),
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────────
    # Orquestador del perfil completo por dataset
    # ──────────────────────────────────────────────────────────────────────────

    def profile_dataset(
        self,
        dataset: str,
        parquet_paths: list[Path],
    ) -> dict[str, pd.DataFrame]:
        """
        Ejecuta las 7 secciones de profiling sobre el DataFrame CONSOLIDADO
        de todos los Parquets Bronze SIAF recibidos.

        A diferencia de hacerlo año por año, aquí se carga un único DataFrame
        con mergeSchema=True para que el profiling refleje la realidad global
        del dataset histórico completo (duplicados entre años, cobertura
        temporal real, cardinalidad acumulada, etc.).

        Args:
            dataset:       Nombre lógico del dataset (ej: "siaf_ingresos").
            parquet_paths: Lista de rutas Parquet a combinar (sin _diario).

        Returns:
            dict con clave = nombre_sección, valor = Pandas DataFrame
        """
        if not parquet_paths:
            raise ValueError("Se requiere al menos un Parquet para el profiling.")

        # Validar existencia de cada archivo
        faltantes = [p for p in parquet_paths if not p.exists()]
        if faltantes:
            raise FileNotFoundError(
                f"Parquets no encontrados: {[str(p) for p in faltantes]}"
            )

        print(f"[INFO] Iniciando profiling consolidado: {dataset}")
        print(f"[INFO]   Archivos a combinar: {len(parquet_paths)}")
        for p in parquet_paths:
            print(f"[INFO]     → {p}")
        logger.info(
            "Iniciando profiling SIAF consolidado: %s | archivos=%d",
            dataset, len(parquet_paths),
        )

        # PySpark combina todos los Parquets en un único DataFrame distribuido.
        # mergeSchema=True une columnas de años con esquemas ligeramente distintos
        # sin errores, igual que en schema_validation y quality pipelines.
        str_paths = [str(p) for p in parquet_paths]
        df: SparkDF = (
            self.spark.read
            .option("mergeSchema", "true")
            .parquet(*str_paths)
        )

        total_rows = df.count()
        total_cols = len(df.columns)

        print(f"[INFO]   DataFrame consolidado: {total_rows:,} filas × {total_cols} columnas")
        logger.info(
            "DataFrame consolidado %s | filas=%d | columnas=%d | archivos=%d",
            dataset, total_rows, total_cols, len(parquet_paths),
        )

        secciones: dict[str, pd.DataFrame] = {}

        try:
            secciones["nulos"] = self.profile_nulls(df, dataset)
        except Exception as e:
            logger.error("Sección nulos falló en %s: %s", dataset, e)
            secciones["nulos"] = pd.DataFrame()

        try:
            secciones["duplicados"] = self.profile_duplicates(df, dataset)
        except Exception as e:
            logger.error("Sección duplicados falló en %s: %s", dataset, e)
            secciones["duplicados"] = pd.DataFrame()

        try:
            secciones["categoricas"] = self.profile_categoricals(df, dataset)
        except Exception as e:
            logger.error("Sección categóricas falló en %s: %s", dataset, e)
            secciones["categoricas"] = pd.DataFrame()

        try:
            temp_resumen, temp_evolucion = self.profile_temporal(df, dataset)
            secciones["temporal_resumen"] = temp_resumen
            secciones["temporal_evolucion"] = temp_evolucion
        except Exception as e:
            logger.error("Sección temporal falló en %s: %s", dataset, e)
            secciones["temporal_resumen"] = pd.DataFrame()
            secciones["temporal_evolucion"] = pd.DataFrame()

        try:
            secciones["cardinalidad"] = self.profile_cardinality(df, dataset)
        except Exception as e:
            logger.error("Sección cardinalidad falló en %s: %s", dataset, e)
            secciones["cardinalidad"] = pd.DataFrame()

        try:
            secciones["consistencia"] = self.profile_consistency(df, dataset)
        except Exception as e:
            logger.error("Sección consistencia falló en %s: %s", dataset, e)
            secciones["consistencia"] = pd.DataFrame()

        try:
            secciones["montos"] = self.profile_monto_signs(df, dataset)
        except Exception as e:
            logger.error("Sección montos falló en %s: %s", dataset, e)
            secciones["montos"] = pd.DataFrame()

        # Metadata del dataset consolidado
        secciones["_meta"] = pd.DataFrame([{
            "dataset": dataset,
            "total_filas": total_rows,
            "total_columnas": total_cols,
            "archivos_combinados": len(parquet_paths),
            "rutas_parquet": " | ".join(str(p) for p in parquet_paths),
            "profiling_status": "OK",
        }])

        logger.info("Profiling completo: %s", dataset)
        return secciones


# ──────────────────────────────────────────────────────────────────────────────
# Generación de reportes HTML (mismo estilo visual que SISMEPRE)
# ──────────────────────────────────────────────────────────────────────────────

def _base_css() -> str:
    return """
    body{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
    header{background:#1f2430;color:white;padding:24px 32px}
    header h1{margin:0;font-size:24px} header p{margin:8px 0 0;color:#cbd5e1}
    main{padding:24px 32px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #3b82f6}
    .label{color:#6b7280;text-transform:uppercase;font-size:12px;letter-spacing:.05em}
    .value{font-size:28px;font-weight:bold;margin-top:8px;color:#2f7d46}
    .section-title{color:#4b5563;text-transform:uppercase;letter-spacing:.08em;font-size:13px;margin:28px 0 10px;
                   border-bottom:2px solid #e5e7eb;padding-bottom:4px}
    table{border-collapse:collapse;width:100%;background:white;border-radius:10px;overflow:hidden;
          font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px}
    th,td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:left;vertical-align:middle}
    th{background:#111827;color:white}
    tr:nth-child(even){background:#fafafa}
    .badge{padding:4px 10px;border-radius:999px;font-weight:bold;font-size:11px;white-space:nowrap}
    .ok{background:#d9eadf;color:#245c36} .revisar{background:#fee2e2;color:#991b1b}
    .note{background:#eef2ff;border-left:4px solid #3b82f6;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px}
    code{background:#f1f5f9;padding:2px 5px;border-radius:4px;font-size:11px}
    """


def _badge(val: str) -> str:
    cls = "ok" if str(val).upper() == "OK" else "revisar"
    return f'<span class="badge {cls}">{_html_lib.escape(str(val))}</span>'


def _df_to_html(df: pd.DataFrame, badge_cols: list[str] | None = None) -> str:
    """Convierte un DataFrame a tabla HTML con badges opcionales en columnas de alerta."""
    if df.empty:
        return "<p><i>Sin datos disponibles para esta sección.</i></p>"
    if badge_cols:
        df = df.copy()
        for col in badge_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: _badge(str(v)))
    return df.to_html(index=False, escape=False, classes="profiling-table")


def write_dataset_html_report(
    dataset: str,
    secciones: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Genera el reporte HTML del profiling consolidado con las 7 secciones."""
    generated_at = datetime.now().isoformat(timespec="seconds")
    meta = secciones.get("_meta", pd.DataFrame())
    total_filas = int(meta["total_filas"].iloc[0]) if not meta.empty else 0
    total_cols = int(meta["total_columnas"].iloc[0]) if not meta.empty else 0
    archivos_combinados = int(meta["archivos_combinados"].iloc[0]) if not meta.empty and "archivos_combinados" in meta.columns else 1

    # Calcular alertas resumen
    nulos_df = secciones.get("nulos", pd.DataFrame())
    cols_con_nulos = int((nulos_df["pct_faltantes"] > 0).sum()) if not nulos_df.empty else 0
    cols_criticas = int((nulos_df["pct_faltantes"] > 5).sum()) if not nulos_df.empty else 0

    dupes_df = secciones.get("duplicados", pd.DataFrame())
    exact_dupes = int(dupes_df[dupes_df["tipo_duplicado"] == "Filas exactas duplicadas"]["conteo"].sum()) if not dupes_df.empty else 0

    montos_df = secciones.get("montos", pd.DataFrame())
    cols_con_neg = int((montos_df["negativos"] > 0).sum()) if not montos_df.empty else 0

    consistencia_df = secciones.get("consistencia", pd.DataFrame())
    inconsistencias = int((consistencia_df["consistencia"] == "INCONSISTENTE").sum()) if not consistencia_df.empty else 0

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Profiling SIAF — {_html_lib.escape(dataset)}</title>
  <style>{_base_css()}</style>
</head>
<body>
  <header>
    <h1>Data Profiling SIAF — {_html_lib.escape(dataset)}</h1>
    <p>Generado: {generated_at} | Capa: Bronze | PySpark | {archivos_combinados} archivos combinados | Sin modificación de datos</p>
  </header>
  <main>
    <div class="note">
      Profiling exhaustivo de 7 secciones sobre Bronze Parquet.
      Resultados orientados a informar decisiones de <b>capa Silver</b> y reglas de <b>Quality</b>.
      Motor: <code>PySpark</code> con agregaciones en pasada única donde es posible.
    </div>

    <div class="cards">
      <div class="card"><div class="label">Filas totales</div><div class="value">{total_filas:,}</div></div>
      <div class="card"><div class="label">Columnas</div><div class="value">{total_cols}</div></div>
      <div class="card"><div class="label">Cols con faltantes</div><div class="value">{cols_con_nulos}</div></div>
      <div class="card"><div class="label">Filas exactas dupl.</div>
        <div class="value" style="color:{'#c24141' if exact_dupes>0 else '#2f7d46'}">{exact_dupes:,}</div></div>
    </div>
    <div class="cards">
      <div class="card"><div class="label">Cols faltantes &gt;5%</div>
        <div class="value" style="color:{'#c24141' if cols_criticas>0 else '#2f7d46'}">{cols_criticas}</div></div>
      <div class="card"><div class="label">Montos con negativos</div>
        <div class="value" style="color:{'#c24141' if cols_con_neg>0 else '#2f7d46'}">{cols_con_neg}</div></div>
      <div class="card"><div class="label">Pares inconsistentes</div>
        <div class="value" style="color:{'#c24141' if inconsistencias>0 else '#2f7d46'}">{inconsistencias}</div></div>
      <div class="card"><div class="label">Dataset</div>
        <div class="value" style="font-size:14px;margin-top:12px">{_html_lib.escape(dataset)}</div></div>
    </div>

    <h2 class="section-title">1 — Análisis de Nulos y Vacíos</h2>
    <div class="note">Conteo de nulos reales (NULL) y faltantes-like (cadenas vacías, "nan", "NULL", etc.)
    por columna. La columna <b>completitud_pct</b> indica qué porcentaje de filas tiene datos válidos.</div>
    {_df_to_html(secciones.get('nulos', pd.DataFrame()), badge_cols=['alerta'])}

    <h2 class="section-title">2 — Análisis de Duplicados</h2>
    <div class="note">Duplicados exactos (todas las columnas idénticas) y duplicados lógicos
    por clave de negocio <code>ANO_DOC + MES_DOC + SEC_EJEC + FUENTE_FINANCIAMIENTO + ESPECIFICA_DET</code>.</div>
    {_df_to_html(secciones.get('duplicados', pd.DataFrame()), badge_cols=['alerta'])}

    <h2 class="section-title">3 — Columnas Categóricas</h2>
    <div class="note">Cardinalidad y top-10 valores para columnas clasificatorias del SIAF.
    Alta cardinalidad en estas columnas puede indicar datos sucios o campos no normalizados.</div>
    {_df_to_html(secciones.get('categoricas', pd.DataFrame()), badge_cols=['alerta'])}

    <h2 class="section-title">4 — Variables Temporales (ANO_DOC / MES_DOC)</h2>
    <div class="note"><code>ANO_DOC</code> y <code>MES_DOC</code> no son campos de fecha pero
    representan la dimensión temporal del presupuesto. Se analiza rango, gaps y cobertura.</div>
    <b>Resumen temporal:</b>
    {_df_to_html(secciones.get('temporal_resumen', pd.DataFrame()), badge_cols=['alerta'])}
    <b>Evolución anual de registros y montos:</b>
    {_df_to_html(secciones.get('temporal_evolucion', pd.DataFrame()))}

    <h2 class="section-title">5 — Cardinalidad General</h2>
    <div class="note">Cardinalidad aproximada (HyperLogLog, rsd=5%) para todas las columnas.
    Útil para decidir tipos de índices y estrategias de particionado en Silver.</div>
    {_df_to_html(secciones.get('cardinalidad', pd.DataFrame()))}

    <h2 class="section-title">6 — Consistencia Código → Nombre</h2>
    <div class="note">Verifica que la relación entre columnas de código y su descripción sea 1→1.
    Un código con múltiples nombres asociados indica datos inconsistentes que deben limpiarse en Silver.</div>
    {_df_to_html(secciones.get('consistencia', pd.DataFrame()), badge_cols=['consistencia'])}

    <h2 class="section-title">7 — Valores Negativos y Cero (Montos)</h2>
    <div class="note">Análisis de signos en <code>MONTO_PIA</code>, <code>MONTO_PIM</code> y
    <code>MONTO_RECAUDADO</code>. Valores negativos en ingresos son inusuales y pueden
    indicar reversiones o errores de carga.</div>
    {_df_to_html(secciones.get('montos', pd.DataFrame()), badge_cols=['alerta_negativos', 'alerta_ceros'])}

  </main>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")
    logger.info("Reporte HTML dataset escrito: %s", output_path)


def write_dashboard_html(
    all_metas: list[dict],
    all_alertas: list[dict],
    output_path: Path,
) -> None:
    """Dashboard consolidado de todos los datasets SIAF procesados."""
    generated_at = datetime.now().isoformat(timespec="seconds")
    total_datasets = len(all_metas)
    total_rows = sum(m.get("total_filas", 0) for m in all_metas)
    total_cols = sum(m.get("total_columnas", 0) for m in all_metas)

    rows_html = []
    for meta, alertas in zip(all_metas, all_alertas):
        status = "REVISAR" if alertas.get("tiene_alertas") else "OK"
        badge_cls = "revisar" if status == "REVISAR" else "ok"
        rows_html.append(
            f"<tr>"
            f"<td><b>{_html_lib.escape(str(meta['dataset']))}</b></td>"
            f"<td>{int(meta['total_filas']):,}</td>"
            f"<td>{int(meta['total_columnas'])}</td>"
            f"<td>{int(alertas.get('exact_dupes', 0)):,}</td>"
            f"<td>{int(alertas.get('cols_nulos_criticos', 0))}</td>"
            f"<td>{int(alertas.get('inconsistencias', 0))}</td>"
            f"<td>{int(alertas.get('cols_con_negativos', 0))}</td>"
            f"<td><span class='badge {badge_cls}'>{status}</span></td>"
            f"</tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dashboard Profiling SIAF — Bronze</title>
  <style>{_base_css()}</style>
</head>
<body>
  <header>
    <h1>Dashboard de Profiling — SIAF Bronze</h1>
    <p>Generado: {generated_at} | Pipeline: profiling_siaf | Motor: PySpark</p>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Datasets analizados</div><div class="value">{total_datasets}</div></div>
      <div class="card"><div class="label">Filas perfiladas</div><div class="value">{total_rows:,}</div></div>
      <div class="card"><div class="label">Columnas perfiladas</div><div class="value">{total_cols}</div></div>
      <div class="card"><div class="label">Secciones por dataset</div><div class="value">7</div></div>
    </div>
    <h2 class="section-title">Resumen de alertas por dataset</h2>
    <table>
      <thead><tr>
        <th>Dataset</th><th>Filas</th><th>Columnas</th>
        <th>Dupl. exactos</th><th>Cols nulos &gt;5%</th>
        <th>Inconsistencias cod→nom</th><th>Montos negativos</th><th>Estado</th>
      </tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
    <div class="note">Ver reportes individuales por dataset para el detalle completo de cada sección.</div>
  </main>
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")
    logger.info("Dashboard HTML escrito: %s", output_path)


def write_audit(
    all_metas: list[dict],
    paths: ProfilingPaths,
    started_at: datetime,
) -> Path:
    finished_at = datetime.now()
    meta = all_metas[0] if all_metas else {}
    record = {
        "pipeline_name": "profiling_siaf",
        "status": "success",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "dataset": meta.get("dataset", "siaf_ingresos"),
        "total_filas": meta.get("total_filas", 0),
        "total_columnas": meta.get("total_columnas", 0),
        "archivos_combinados": meta.get("archivos_combinados", 0),
        "secciones_ejecutadas": 7,
        "secciones": [
            "nulos", "duplicados", "categoricas",
            "temporal_resumen", "temporal_evolucion",
            "cardinalidad", "consistencia", "montos",
        ],
        "outputs": {
            "reports_root": str(paths.reports_root),
            "profile_html": str(paths.reports_root / "siaf_ingresos_profile.html"),
            "dashboard_html": str(paths.reports_root / "siaf_profile_dashboard.html"),
            "summary_csv": str(paths.reports_root / "siaf_profile_summary.csv"),
        },
    }
    partition = finished_at.strftime("year=%Y/month=%m/day=%d")
    audit_dir = paths.audit_root / partition
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"siaf_profiling_{finished_at.strftime('%Y%m%d_%H%M%S')}.json"
    audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Auditoría escrita: %s", audit_path)
    return audit_path


# ──────────────────────────────────────────────────────────────────────────────
# Función principal exportable
# ──────────────────────────────────────────────────────────────────────────────

def run_siaf_profiling(
    spark: SparkSession,
    parquet_paths: list[Path],
    bronze_root: Path = Path("data/bronze"),
    reports_root: Path = Path("reports/profiling/siaf"),
    audit_root: Path = Path("data/audit/profiling"),
) -> dict[str, str]:
    """
    Ejecuta el profiling completo (7 secciones) sobre el dataset SIAF consolidado.

    Todos los Parquets recibidos se combinan en un único DataFrame PySpark
    (mergeSchema=True) antes de ejecutar cualquier análisis. Esto garantiza
    que métricas como duplicados entre años, cobertura temporal completa,
    cardinalidad acumulada y consistencia código→nombre reflejen la realidad
    del dataset histórico global, no la de un año aislado.

    Genera:
      - reports/profiling/siaf/siaf_ingresos_profile.html   → reporte consolidado
      - reports/profiling/siaf/siaf_ingresos_<seccion>.csv  → CSV por sección
      - reports/profiling/siaf/siaf_profile_dashboard.html  → dashboard ejecutivo
      - reports/profiling/siaf/siaf_profile_summary.csv     → resumen tabular
      - data/audit/profiling/.../siaf_profiling_<ts>.json   → auditoría JSON

    Returns:
        dict con rutas de los artefactos generados.
    """
    started_at = datetime.now()
    paths = ProfilingPaths(
        bronze_root=Path(bronze_root),
        reports_root=Path(reports_root),
        audit_root=Path(audit_root),
    )
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    paths.audit_root.mkdir(parents=True, exist_ok=True)

    profiler = SiafDatasetProfiler(spark=spark, paths=paths)

    # Nombre lógico único para el dataset consolidado
    dataset = "siaf_ingresos"

    print(f"\n[INFO] ── Profiling consolidado SIAF ({len(parquet_paths)} archivos) ──")

    try:
        # Una sola llamada con todos los paths: carga combinada + 7 secciones
        secciones = profiler.profile_dataset(dataset, parquet_paths)
        meta = secciones["_meta"].iloc[0].to_dict()

        # Calcular alertas
        nulos_df = secciones.get("nulos", pd.DataFrame())
        dupes_df = secciones.get("duplicados", pd.DataFrame())
        montos_df = secciones.get("montos", pd.DataFrame())
        consist_df = secciones.get("consistencia", pd.DataFrame())

        exact_dupes = int(dupes_df[dupes_df["tipo_duplicado"] == "Filas exactas duplicadas"]["conteo"].sum()) if not dupes_df.empty else 0
        cols_nulos_criticos = int((nulos_df["pct_faltantes"] > 5).sum()) if not nulos_df.empty else 0
        cols_con_negativos = int((montos_df["negativos"] > 0).sum()) if not montos_df.empty else 0
        inconsistencias = int((consist_df["consistencia"] == "INCONSISTENTE").sum()) if not consist_df.empty else 0
        tiene_alertas = any([exact_dupes > 0, cols_nulos_criticos > 0,
                             cols_con_negativos > 0, inconsistencias > 0])

        # ── CSVs por sección ──────────────────────────────────────────────────
        csv_outputs: dict[str, str] = {}
        for nombre_sec, sec_df in secciones.items():
            if nombre_sec.startswith("_") or sec_df.empty:
                continue
            csv_path = paths.reports_root / f"{dataset}_{nombre_sec}.csv"
            sec_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            csv_outputs[f"csv_{nombre_sec}"] = str(csv_path)
            print(f"[INFO]   CSV {nombre_sec}: {csv_path}")

        # ── Reporte HTML del dataset consolidado ──────────────────────────────
        html_path = paths.reports_root / f"{dataset}_profile.html"
        write_dataset_html_report(dataset, secciones, html_path)
        print(f"[INFO]   Reporte HTML: {html_path}")

        # ── Summary CSV (fila única del dataset consolidado) ──────────────────
        summary_record = {
            "dataset": dataset,
            "total_filas": meta["total_filas"],
            "total_columnas": meta["total_columnas"],
            "archivos_combinados": meta.get("archivos_combinados", len(parquet_paths)),
            "exact_dupes": exact_dupes,
            "cols_nulos_criticos": cols_nulos_criticos,
            "cols_con_negativos": cols_con_negativos,
            "inconsistencias_cod_nom": inconsistencias,
            "estado": "REVISAR" if tiene_alertas else "OK",
        }
        summary_csv = paths.reports_root / "siaf_profile_summary.csv"
        pd.DataFrame([summary_record]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

        # ── Dashboard HTML ejecutivo ──────────────────────────────────────────
        dashboard_path = paths.reports_root / "siaf_profile_dashboard.html"
        alertas_dict = {
            "dataset": dataset,
            "exact_dupes": exact_dupes,
            "cols_nulos_criticos": cols_nulos_criticos,
            "cols_con_negativos": cols_con_negativos,
            "inconsistencias": inconsistencias,
            "tiene_alertas": tiene_alertas,
        }
        write_dashboard_html([meta], [alertas_dict], dashboard_path)

        # ── Auditoría JSON ────────────────────────────────────────────────────
        audit_path = write_audit([meta], paths, started_at)

        outputs = {
            "dashboard_html": str(dashboard_path),
            "profile_html": str(html_path),
            "summary_csv": str(summary_csv),
            "reports_dir": str(paths.reports_root),
            "audit_path": str(audit_path),
        }
        outputs.update(csv_outputs)
        return outputs

    except Exception as exc:
        logger.error("Error fatal en profiling consolidado SIAF: %s", exc, exc_info=True)
        print(f"[ERROR] Falló profiling consolidado SIAF: {exc}")
        raise