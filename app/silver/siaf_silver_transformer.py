"""
siaf_silver_transformer.py
Transformador Bronze → Silver para SIAF Ingresos.

Lee las reglas desde silver_rules_siaf.yaml (secciones:
filtering, cleaning, imputations, derived_columns,
audit_columns, deduplication).

Orden de ejecución:
  1. Lectura de Parquet Bronze (excluye rutas "_diario")
  2. filtering       — NIVEL_GOBIERNO = 'M'
  3. cleaning        — null_tokens, global_trim, lpad_codes
  4. imputations     — rellena vacíos estructurales post-filtro
  5. derived_columns — ubigeo_ejecutora, sk_tiempo_mensual,
                       nombre_municipalidad_normalizado,
                       anio_mes_key, ano_particion,
                       flag_tipo_transaccion,
                       flag_monto_pia_activo
  6. audit_columns   — source_system, source_dataset,
                       silver_processed_at, _audit_loaded_at,
                       record_hash (sha2-256)
  7. deduplication   — row_number() por clave de negocio
  8. Escritura Parquet Silver particionado por ano_particion

Filosofía:
  Bronze conserva → Silver estandariza → Gold modela.
  Silver NO elimina registros válidos ni altera significado
  de negocio.  Los negativos en montos son válidos (SIAF).

Entorno:
  Docker con PySpark. bind-mount en /app.
  Ejecución:
    docker exec -it gdm_pyspark_siaf python run_siaf_silver.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDF
    from pyspark.sql import functions as F
    from pyspark.sql import Window
    from pyspark.sql.types import (
        IntegerType, LongType, DoubleType, StringType,
        BooleanType, TimestampType,
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

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


# ─────────────────────────────────────────────────────────────
# Estructuras de datos de resultado
# ─────────────────────────────────────────────────────────────

@dataclass
class SiafSilverResult:
    """Resultado de la transformación Silver SIAF."""

    dataset: str
    bronze_paths: list[Path]
    silver_path: Path
    bronze_rows: int
    filtered_rows: int       # tras filtro NIVEL_GOBIERNO=M
    silver_rows: int         # tras dedup
    bronze_columns: int
    silver_columns: int
    detail_rows: list[dict[str, Any]]
    conversion_errors: int
    duplicates_removed: int
    nulls_before: int
    nulls_after: int
    status: str
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Carga de configuración
# ─────────────────────────────────────────────────────────────

def load_silver_rules(path: Path) -> dict[str, Any]:
    """Carga silver_rules_siaf.yaml con yaml.safe_load."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el YAML de reglas Silver: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError("El YAML de reglas Silver no tiene estructura válida.")
    return config


# ─────────────────────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────────────────────

def _discover_bronze_files(bronze_base: Path) -> list[Path]:
    """
    Descubre todos los Parquet bajo bronze_base_path.
    CRÍTICO: omite cualquier ruta que contenga '_diario'
    para evitar duplicar millones de registros.
    """
    if not bronze_base.exists():
        logger.warning("Ruta Bronze no encontrada: %s", bronze_base)
        return []

    all_parquets = sorted(bronze_base.rglob("*.parquet"))
    filtered = [p for p in all_parquets if "_diario" not in str(p).lower()]

    skipped = len(all_parquets) - len(filtered)
    if skipped:
        logger.warning(
            "Omitidos %d Parquet con '_diario' para evitar duplicados.", skipped
        )

    logger.info("Parquet Bronze encontrados: %d archivos.", len(filtered))
    for p in filtered:
        logger.info("  → %s", p)
    return filtered


def _build_spark(app_name: str = "SIAF Silver Pipeline") -> "SparkSession":
    """Obtiene o crea la SparkSession."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _spark_to_pandas_summary(df: "SparkDF") -> pd.DataFrame:
    """
    Convierte el DataFrame Spark de resumen a Pandas.
    Regla de arquitectura: el resultado final siempre es Pandas
    para compatibilidad con los generadores de reportes HTML.
    """
    return df.toPandas()


# ─────────────────────────────────────────────────────────────
# Pasos del pipeline (cada uno es una función pura)
# ─────────────────────────────────────────────────────────────

def step_read_bronze(spark: "SparkSession", paths: list[Path]) -> "SparkDF":
    """Lee y combina todos los Parquet Bronze en un único DataFrame Spark."""
    str_paths = [str(p) for p in paths]
    df = spark.read.parquet(*str_paths)
    logger.info("Bronze leído: %d filas, %d columnas.", df.count(), len(df.columns))
    return df


def step_filtering(df: "SparkDF", cfg: dict[str, Any]) -> "SparkDF":
    """
    Aplica el filtro maestro definido en filtering.master_filter.
    Por defecto: NIVEL_GOBIERNO = 'M'.
    """
    master = cfg.get("filtering", {}).get("master_filter", {})
    column = master.get("column", "NIVEL_GOBIERNO")
    value = master.get("value", "M")

    before = df.count()
    df = df.filter(F.col(column) == value)
    after = df.count()

    logger.info(
        "Filtro %s='%s': %d → %d filas (-%d).",
        column, value, before, after, before - after,
    )
    return df


def step_cleaning(df: "SparkDF", cfg: dict[str, Any]) -> "SparkDF":
    """
    Cleaning:
      1. Convierte null_tokens a null (solo columnas string)
      2. global_trim en todas las columnas StringType
      3. lpad en campos de código definidos en lpad_codes
    """
    cleaning_cfg = cfg.get("cleaning", {})

    # ── 1. Null tokens ────────────────────────────────────────
    null_tokens_cfg = cleaning_cfg.get("null_tokens", {})
    tokens: list[str] = null_tokens_cfg.get("tokens", [])
    if null_tokens_cfg.get("convert_to_null", True) and tokens:
        string_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType)
        ]
        for col_name in string_cols:
            df = df.withColumn(
                col_name,
                F.when(
                    F.trim(F.col(col_name)).isin(tokens),
                    F.lit(None).cast(StringType()),
                ).otherwise(F.col(col_name)),
            )
        logger.info("Null tokens convertidos a null en %d columnas string.", len(string_cols))

    # ── 2. Global trim ────────────────────────────────────────
    global_trim_cfg = cleaning_cfg.get("global_trim", {})
    if global_trim_cfg.get("apply", True):
        string_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType)
        ]
        for col_name in string_cols:
            df = df.withColumn(col_name, F.trim(F.col(col_name)))
        logger.info("trim() aplicado a %d columnas string.", len(string_cols))

    # ── 3. lpad en códigos ────────────────────────────────────
    lpad_rules: list[dict] = cleaning_cfg.get("lpad_codes", [])
    for rule in lpad_rules:
        col_name: str = rule["column"]
        length: int = rule["length"]
        fill: str = rule.get("fill_char", "0")
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.lpad(F.trim(F.col(col_name).cast(StringType())), length, fill),
            )
            logger.info(
                "lpad('%s', %d, '%s') aplicado.", col_name, length, fill
            )
        else:
            logger.warning("lpad_codes: columna '%s' no existe en el DataFrame.", col_name)

    return df


def step_imputations(df: "SparkDF", cfg: dict[str, Any]) -> "SparkDF":
    """
    Imputa vacíos estructurales post-filtro NIVEL_GOBIERNO=M.

    Secciones leídas del YAML:
      void_codes        → SECTOR, PLIEGO = "00"
      void_names        → SECTOR_NOMBRE, PLIEGO_NOMBRE = "SIN DATO"
      geo_residual_nulls → códigos/nombres geográficos con 0.014% nulos
    """
    imp_cfg = cfg.get("imputations", {})

    # void_codes
    void_codes = imp_cfg.get("void_codes", {})
    code_value: str = void_codes.get("impute_value", "00")
    for col_name in (void_codes.get("columns") or []):
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.coalesce(F.col(col_name), F.lit(code_value)),
            )
            logger.info("Imputado '%s' → '%s'", col_name, code_value)

    # void_names
    void_names = imp_cfg.get("void_names", {})
    name_value: str = void_names.get("impute_value", "SIN DATO")
    for col_name in (void_names.get("columns") or []):
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.coalesce(F.col(col_name), F.lit(name_value)),
            )
            logger.info("Imputado '%s' → '%s'", col_name, name_value)

    # geo_residual_nulls
    geo_cfg = imp_cfg.get("geo_residual_nulls", {})
    geo_code_val: str = geo_cfg.get("void_codes_geo", {}).get("impute_value", "00")
    for col_name in (geo_cfg.get("void_codes_geo", {}).get("columns") or []):
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.coalesce(F.col(col_name), F.lit(geo_code_val)),
            )

    geo_name_val: str = geo_cfg.get("void_names_geo", {}).get("impute_value", "SIN DATO")
    for col_name in (geo_cfg.get("void_names_geo", {}).get("columns") or []):
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.coalesce(F.col(col_name), F.lit(geo_name_val)),
            )

    logger.info("Imputaciones aplicadas.")
    return df


def step_type_casting(df: "SparkDF", cfg: dict[str, Any]) -> tuple["SparkDF", int, list[dict]]:
    """
    Convierte tipos de datos según output_schema del YAML.
    Retorna (df_convertido, total_errores, detail_rows).

    La conversión de tipos se hace DESPUÉS de cleaning e imputations
    para que los valores ya estén limpios.
    """
    schema_cols: list[dict] = cfg.get("output_schema", {}).get("columns", [])
    # Construimos mapa col → tipo para las columnas originales
    type_map = {
        item["name"]: item["type"]
        for item in schema_cols
        if "name" in item and "type" in item
    }

    conversion_errors = 0
    detail_rows: list[dict] = []

    for col_name, target_type in type_map.items():
        if col_name not in df.columns:
            continue

        bronze_type = str(dict(df.dtypes).get(col_name, "unknown"))
        errors = 0

        try:
            if target_type == "integer":
                df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
                transformation = "cast_to_integer"
            elif target_type == "double":
                df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
                transformation = "cast_to_double"
            elif target_type == "boolean":
                # Se construye en derived_columns, no aquí
                transformation = "built_in_derived"
            elif target_type == "string":
                df = df.withColumn(col_name, F.col(col_name).cast(StringType()))
                transformation = "cast_to_string"
            elif target_type == "timestamp":
                # Las columnas timestamp se agregan en audit_columns
                transformation = "built_in_audit"
            else:
                transformation = "unchanged"
        except Exception as exc:
            logger.warning("Error convirtiendo '%s' a %s: %s", col_name, target_type, exc)
            errors += 1
            transformation = "conversion_failed"

        conversion_errors += errors
        detail_rows.append({
            "dataset": "ingresos",
            "column_name": col_name,
            "bronze_type": bronze_type,
            "silver_type": target_type,
            "transformation": transformation,
            "conversion_errors": errors,
            "status": "OK" if errors == 0 else "CONVERSION_ERRORS",
            "reason": "",
        })

    logger.info(
        "Conversión de tipos: %d columnas procesadas, %d errores.",
        len(type_map), conversion_errors,
    )
    return df, conversion_errors, detail_rows


def step_derived_columns(df: "SparkDF", cfg: dict[str, Any]) -> "SparkDF":
    """
    Genera las columnas derivadas definidas en derived_columns.

    Columnas creadas (en orden):
      ubigeo_ejecutora              — lpad(concat(DPTO,PROV,DIST),6,'0')
      nombre_municipalidad_normalizado — upper(trim(regexp_replace(...)))
      sk_tiempo_mensual             — ANO_DOC*100 + MES_DOC
      anio_mes_key                  — YYYY-MM
      ano_particion                 — ANO_DOC (para particionado)
      flag_tipo_transaccion         — NORMAL / REVERSION
      flag_monto_pia_activo         — boolean MONTO_PIA > 0
    """
    # ── ubigeo_ejecutora ──────────────────────────────────────
    df = df.withColumn(
        "ubigeo_ejecutora",
        F.lpad(
            F.concat(
                F.col("DEPARTAMENTO_EJECUTORA"),
                F.col("PROVINCIA_EJECUTORA"),
                F.col("DISTRITO_EJECUTORA"),
            ),
            6, "0",
        ),
    )

    # ── nombre_municipalidad_normalizado ──────────────────────
    df = df.withColumn(
        "nombre_municipalidad_normalizado",
        F.upper(
            F.trim(
                F.regexp_replace(F.col("EJECUTORA_NOMBRE"), r"\s+", " ")
            )
        ),
    )

    # ── sk_tiempo_mensual ─────────────────────────────────────
    df = df.withColumn(
        "sk_tiempo_mensual",
        (F.col("ANO_DOC").cast(IntegerType()) * 100
         + F.col("MES_DOC").cast(IntegerType())).cast(IntegerType()),
    )

    # ── anio_mes_key  (YYYY-MM) ───────────────────────────────
    df = df.withColumn(
        "anio_mes_key",
        F.concat(
            F.col("ANO_DOC").cast(StringType()),
            F.lit("-"),
            F.lpad(F.col("MES_DOC").cast(StringType()), 2, "0"),
        ),
    )

    # ── ano_particion ─────────────────────────────────────────
    df = df.withColumn(
        "ano_particion",
        F.col("ANO_DOC").cast(IntegerType()),
    )

    # ── flag_tipo_transaccion ─────────────────────────────────
    df = df.withColumn(
        "flag_tipo_transaccion",
        F.when(F.col("MONTO_RECAUDADO") >= 0, F.lit("NORMAL"))
         .otherwise(F.lit("REVERSION")),
    )

    # ── flag_monto_pia_activo ─────────────────────────────────
    df = df.withColumn(
        "flag_monto_pia_activo",
        F.col("MONTO_PIA").cast(DoubleType()) > 0,
    )

    logger.info("Columnas derivadas generadas: 7.")
    return df


def step_audit_columns(
    df: "SparkDF",
    cfg: dict[str, Any],
    processed_at: datetime,
) -> "SparkDF":
    """
    Agrega columnas técnicas de trazabilidad.

    Columnas creadas:
      source_system         — literal "siaf"
      source_dataset        — literal "ingresos"
      silver_processed_at   — timestamp UTC del proceso
      _audit_loaded_at      — alias para compatibilidad con silver_audit.py
      record_hash           — sha2(concat_ws('|', ...cols originales...), 256)
    """
    audit_cfg = cfg.get("audit_columns", {})

    # source_system
    sys_value = audit_cfg.get("source_system", {}).get("value", "siaf")
    df = df.withColumn("source_system", F.lit(sys_value))

    # source_dataset
    ds_value = audit_cfg.get("source_dataset", {}).get("value", "ingresos")
    df = df.withColumn("source_dataset", F.lit(ds_value))

    # silver_processed_at y _audit_loaded_at
    ts_lit = F.lit(processed_at.isoformat()).cast(TimestampType())
    df = df.withColumn("silver_processed_at", ts_lit)
    df = df.withColumn("_audit_loaded_at", ts_lit)

    # record_hash — sobre las 36 columnas originales de Bronze
    # (excluye derivadas y técnicas para que represente el dato origen)
    _ORIGINAL_COLS = [
        "ANO_DOC", "MES_DOC", "NIVEL_GOBIERNO", "NIVEL_GOBIERNO_NOMBRE",
        "SECTOR", "SECTOR_NOMBRE", "PLIEGO", "PLIEGO_NOMBRE",
        "SEC_EJEC", "EJECUTORA", "EJECUTORA_NOMBRE",
        "DEPARTAMENTO_EJECUTORA", "DEPARTAMENTO_EJECUTORA_NOMBRE",
        "PROVINCIA_EJECUTORA", "PROVINCIA_EJECUTORA_NOMBRE",
        "DISTRITO_EJECUTORA", "DISTRITO_EJECUTORA_NOMBRE",
        "FUENTE_FINANCIAMIENTO", "FUENTE_FINANCIAMIENTO_NOMBRE",
        "RUBRO", "RUBRO_NOMBRE", "TIPO_RECURSO", "TIPO_RECURSO_NOMBRE",
        "GENERICA", "GENERICA_NOMBRE",
        "SUBGENERICA", "SUBGENERICA_NOMBRE",
        "SUBGENERICA_DET", "SUBGENERICA_DET_NOMBRE",
        "ESPECIFICA", "ESPECIFICA_NOMBRE",
        "ESPECIFICA_DET", "ESPECIFICA_DET_NOMBRE",
        "MONTO_PIA", "MONTO_PIM", "MONTO_RECAUDADO",
    ]
    hash_cols = [
        F.coalesce(F.col(c).cast(StringType()), F.lit("<NULL>"))
        for c in _ORIGINAL_COLS
        if c in df.columns
    ]
    df = df.withColumn(
        "record_hash",
        F.sha2(F.concat_ws("|", *hash_cols), 256),
    )

    logger.info("Columnas de auditoría agregadas: 5.")
    return df


def step_deduplication(df: "SparkDF", cfg: dict[str, Any]) -> tuple["SparkDF", int]:
    """
    Deduplicación por clave de negocio (13 columnas).
    Estrategia keep_first ordenando por ANO_DOC DESC, MES_DOC DESC.
    Retorna (df_deduplicado, filas_eliminadas).
    """
    dedup_cfg = cfg.get("deduplication", {})

    if not dedup_cfg.get("enabled", True):
        logger.info("Deduplicación deshabilitada en YAML.")
        return df, 0

    key_cols: list[str] = dedup_cfg.get("key_columns", [])
    if not key_cols:
        logger.warning("deduplication.key_columns vacío — omitiendo dedup.")
        return df, 0

    window_order_cfg: list[dict] = dedup_cfg.get("window_order", [
        {"column": "ANO_DOC", "direction": "desc"},
        {"column": "MES_DOC", "direction": "desc"},
    ])
    order_exprs = []
    for item in window_order_cfg:
        col_expr = F.col(item["column"])
        order_exprs.append(
            col_expr.desc() if item.get("direction", "asc") == "desc"
            else col_expr.asc()
        )

    # Solo incluir en la partition las key_cols que existen
    existing_keys = [c for c in key_cols if c in df.columns]
    missing_keys = [c for c in key_cols if c not in df.columns]
    if missing_keys:
        logger.warning("Columnas clave de dedup no encontradas: %s", missing_keys)

    window = Window.partitionBy(*existing_keys).orderBy(*order_exprs)
    before = df.count()

    df = (
        df.withColumn("_rn", F.row_number().over(window))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
    )

    after = df.count()
    removed = before - after
    logger.info(
        "Deduplicación: %d → %d filas. Eliminadas: %d duplicados.",
        before, after, removed,
    )
    return df, removed


def step_write_silver(
    df: "SparkDF",
    silver_path: Path,
    partition_cols: list[str],
) -> None:
    """
    Escribe el Parquet Silver particionado por ano_particion.
    Modo overwrite para soportar re-ejecuciones idempotentes.
    """
    silver_path.mkdir(parents=True, exist_ok=True)
    (
        df.write
          .mode("overwrite")
          .partitionBy(*partition_cols)
          .parquet(str(silver_path))
    )
    logger.info("Parquet Silver escrito en: %s", silver_path)


# ─────────────────────────────────────────────────────────────
# Transformer principal
# ─────────────────────────────────────────────────────────────

class SiafSilverTransformer:
    """
    Orquesta todos los pasos Bronze → Silver para SIAF Ingresos.
    Lee las reglas exclusivamente desde silver_rules_siaf.yaml.
    Usa PySpark para el procesamiento masivo.
    El resultado final (resúmenes) se convierte a Pandas para
    compatibilidad con los reportes HTML.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.source_system = config.get("source_system", "SIAF")
        self.bronze_base = Path(config.get("bronze_base_path", "data/bronze/siaf"))
        self.silver_base = Path(config.get("silver_base_path", "data/silver/siaf"))
        self.audit_base = Path(config.get("audit_base_path", "data/audit/silver"))
        self.reports_base = Path(config.get("reports_base_path", "reports/silver/siaf"))

    def run(
        self,
        spark: "SparkSession",
        processed_at: datetime | None = None,
    ) -> tuple[SiafSilverResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta el pipeline completo.
        Retorna (result, summary_df, detail_df, dictionary_df) en Pandas.
        """
        processed_at = processed_at or datetime.now()
        notes: list[str] = []

        # ── 1. Descubrir y leer Bronze ────────────────────────
        bronze_files = _discover_bronze_files(self.bronze_base)
        if not bronze_files:
            raise FileNotFoundError(
                f"No se encontraron Parquet Bronze en {self.bronze_base}"
            )

        df = step_read_bronze(spark, bronze_files)
        bronze_rows = df.count()
        bronze_cols = len(df.columns)

        # Conteo de nulos antes (sobre muestra para no bloquear)
        nulls_before = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()[0].asDict()
        total_nulls_before = sum(nulls_before.values())

        # ── 2. Filtering ─────────────────────────────────────
        df = step_filtering(df, self.config)
        filtered_rows = df.count()

        # ── 3. Cleaning ───────────────────────────────────────
        df = step_cleaning(df, self.config)

        # ── 4. Imputations ────────────────────────────────────
        df = step_imputations(df, self.config)

        # ── 5. Type casting ───────────────────────────────────
        df, conversion_errors, detail_rows = step_type_casting(df, self.config)

        # ── 6. Derived columns ────────────────────────────────
        df = step_derived_columns(df, self.config)

        # ── 7. Audit columns ──────────────────────────────────
        df = step_audit_columns(df, self.config, processed_at)

        # ── 8. Deduplication ──────────────────────────────────
        df, duplicates_removed = step_deduplication(df, self.config)
        silver_rows = df.count()
        silver_cols = len(df.columns)

        # Nulos después
        nulls_after_dict = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
        ).collect()[0].asDict()
        total_nulls_after = sum(nulls_after_dict.values())

        # ── 9. Escritura Parquet Silver ───────────────────────
        partition_cols: list[str] = (
            self.config.get("output_schema", {})
            .get("partition_by", ["ano_particion"])
        )
        step_write_silver(df, self.silver_base, partition_cols)

        status = "OK" if conversion_errors == 0 else "WITH_CONVERSION_ERRORS"

        result = SiafSilverResult(
            dataset="ingresos",
            bronze_paths=bronze_files,
            silver_path=self.silver_base,
            bronze_rows=bronze_rows,
            filtered_rows=filtered_rows,
            silver_rows=silver_rows,
            bronze_columns=bronze_cols,
            silver_columns=silver_cols,
            detail_rows=detail_rows,
            conversion_errors=conversion_errors,
            duplicates_removed=duplicates_removed,
            nulls_before=total_nulls_before,
            nulls_after=total_nulls_after,
            status=status,
            notes=notes,
        )

        # ── Construir DataFrames de reporte (Pandas) ──────────
        summary_df = self._build_summary(result)
        detail_df = self._build_detail(result)
        dictionary_df = self._build_data_dictionary(result, df)

        return result, summary_df, detail_df, dictionary_df

    # ── Constructores de reportes ─────────────────────────────

    def _build_summary(self, result: SiafSilverResult) -> pd.DataFrame:
        """Una fila de resumen ejecutivo del proceso."""
        detail = pd.DataFrame(result.detail_rows)
        converted = int(
            detail[detail["transformation"].str.startswith("cast_to", na=False)].shape[0]
        ) if not detail.empty else 0
        numeric_conv = int(
            detail[detail["transformation"].isin(
                ["cast_to_integer", "cast_to_double"]
            )].shape[0]
        ) if not detail.empty else 0

        row = {
            "dataset":                   result.dataset,
            "bronze_files":              len(result.bronze_paths),
            "bronze_rows":               result.bronze_rows,
            "filtered_rows":             result.filtered_rows,
            "rows_filtered_out":         result.bronze_rows - result.filtered_rows,
            "silver_rows":               result.silver_rows,
            "duplicates_removed":        result.duplicates_removed,
            "bronze_columns":            result.bronze_columns,
            "silver_columns":            result.silver_columns,
            "columns_converted":         converted,
            "numeric_columns_converted": numeric_conv,
            "technical_columns_added":   5,   # source_system,source_dataset,silver_processed_at,_audit_loaded_at,record_hash
            "derived_columns_added":     7,   # ubigeo,nombre_norm,sk_tiempo,anio_mes,ano_part,flag_trans,flag_pia
            "conversion_errors":         result.conversion_errors,
            "nulls_before":              result.nulls_before,
            "nulls_after":               result.nulls_after,
            "status":                    result.status,
            "notes":                     " | ".join(result.notes),
            "silver_path":               str(result.silver_path),
        }
        return pd.DataFrame([row])

    def _build_detail(self, result: SiafSilverResult) -> pd.DataFrame:
        """Una fila por columna con el resultado de la transformación."""
        return pd.DataFrame(result.detail_rows)

    def _build_data_dictionary(
        self,
        result: SiafSilverResult,
        df: "SparkDF",
    ) -> pd.DataFrame:
        """Diccionario Silver: una fila por columna con tipo y descripción."""
        schema_cols: list[dict] = (
            self.config.get("output_schema", {}).get("columns", [])
        )
        schema_map = {item["name"]: item for item in schema_cols if "name" in item}

        rows = []
        spark_dtypes = dict(df.dtypes)

        for col_name, spark_type in spark_dtypes.items():
            schema_info = schema_map.get(col_name, {})
            is_tech = col_name in {
                "source_system", "source_dataset",
                "silver_processed_at", "_audit_loaded_at", "record_hash",
            }
            is_derived = col_name in {
                "ubigeo_ejecutora", "nombre_municipalidad_normalizado",
                "sk_tiempo_mensual", "anio_mes_key", "ano_particion",
                "flag_tipo_transaccion", "flag_monto_pia_activo",
            }

            origin = "bronze_original"
            if is_tech:
                origin = "audit_column"
            elif is_derived:
                origin = "derived_column"

            rows.append({
                "dataset":               result.dataset,
                "column_name":           col_name,
                "silver_type":           spark_type,
                "nullable":              schema_info.get("nullable", True),
                "origin":                origin,
                "imputed_value":         schema_info.get("imputed_value", ""),
                "note":                  schema_info.get("note", ""),
                "source":                "Bronze SIAF + silver_rules_siaf.yaml",
            })

        return pd.DataFrame(rows)