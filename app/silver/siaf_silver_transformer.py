"""
siaf_silver_transformer.py

Transformador Bronze → Silver para SIAF Ingresos.

Aplica exclusivamente las transformaciones definidas en silver_rules_siaf.yaml:
- Convierte tipos de datos (integer, decimal, string, datetime).
- Normaliza tokens vacíos a null.
- Aplica trim() a columnas de texto.
- Genera columna derivada anio_mes_key (YYYY-MM).
- Genera columna derivada ubigeo_ejecutora (6 dígitos).
- Agrega columnas técnicas de trazabilidad.
- Guarda Parquet Silver por año/dataset.

Silver NO limpia agresivamente.
Silver NO elimina registros válidos.
Silver NO altera el significado de negocio.
Bronze conserva → Silver estandariza → Gold modela.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from app.utils.logger import get_logger


logger = get_logger(__name__, log_dir=Path("logs"))


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSilverResult:
    """Resultado de transformación Silver para un dataset SIAF."""
    dataset: str
    bronze_path: Path
    silver_path: Path
    bronze_rows: int
    silver_rows: int
    bronze_columns: int
    silver_columns: int
    detail_rows: list[dict[str, Any]]
    dropped_columns: list[str]
    conversion_errors: int
    nulls_before: int
    nulls_after: int
    status: str
    notes: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Carga de configuración
# ──────────────────────────────────────────────────────────────────────────────

def load_silver_rules(path: Path) -> dict[str, Any]:
    """Carga el archivo YAML con reglas Silver."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de reglas Silver: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("El archivo YAML de reglas Silver no tiene estructura válida.")
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de conversión (idénticas a SISMEPRE)
# ──────────────────────────────────────────────────────────────────────────────

def _as_list(value: Any) -> list[str]:
    """Convierte listas/diccionarios/valores simples a lista de nombres de columnas."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, str):
        return [value]
    return list(value)


def _reason_from_mapping(mapping: Any, column: str) -> str:
    """Obtiene la razón documentada para una columna dentro de una regla YAML."""
    if isinstance(mapping, dict) and column in mapping:
        value = mapping[column]
        if isinstance(value, dict):
            return str(value.get("reason", ""))
        return str(value)
    return ""


def _normalize_missing_tokens(df: pd.DataFrame, missing_tokens: list[str]) -> pd.DataFrame:
    """
    Convierte tokens vacíos o textuales a valores nulos.
    Solo opera sobre columnas object/string para no alterar valores numéricos válidos como 0.
    """
    tokens = set(missing_tokens or [])
    result = df.copy()

    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column]):
            series = result[column]
            stripped = series.astype("string").str.strip()
            mask = stripped.isin(tokens)
            result.loc[mask, column] = pd.NA

    return result


def _convert_to_string(series: pd.Series) -> pd.Series:
    """Convierte una columna a string nullable de pandas, conservando nulos."""
    return series.astype("string")


def _convert_to_integer(series: pd.Series) -> tuple[pd.Series, int]:
    """Convierte una columna a entero nullable y cuenta errores reales de conversión."""
    before_not_null = series.notna()
    converted = pd.to_numeric(series, errors="coerce")
    errors = int((before_not_null & converted.isna()).sum())
    return converted.astype("Int64"), errors


def _convert_to_decimal(series: pd.Series) -> tuple[pd.Series, int]:
    """Convierte una columna a decimal/float y cuenta errores reales de conversión."""
    before_not_null = series.notna()
    converted = pd.to_numeric(series, errors="coerce")
    errors = int((before_not_null & converted.isna()).sum())
    return converted.astype("Float64"), errors


def _convert_to_datetime(series: pd.Series) -> tuple[pd.Series, int]:
    """Convierte una columna a datetime nullable (dayfirst para fuentes MEF)."""
    before_not_null = series.notna()
    converted = pd.to_datetime(series, errors="coerce", dayfirst=True)
    errors = int((before_not_null & converted.isna()).sum())
    return converted, errors


def _format_key_value(value: Any) -> str:
    """Formatea valores para claves derivadas, evitando .0 en enteros nullable."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _get_column_target_type(dataset_rules: dict[str, Any], column: str) -> str:
    """Devuelve el tipo objetivo definido en el YAML para una columna."""
    if column in _as_list(dataset_rules.get("integer_columns")):
        return "integer"
    if column in _as_list(dataset_rules.get("decimal_columns")):
        return "decimal"
    if column in _as_list(dataset_rules.get("datetime_columns")):
        return "datetime"
    if column in _as_list(dataset_rules.get("string_columns")):
        return "string"
    return "unchanged"


def _hash_row(row: pd.Series, columns: list[str]) -> str:
    """Genera SHA-256 estable para una fila usando las columnas indicadas."""
    parts = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            value_text = "<NULL>"
        elif isinstance(value, pd.Timestamp):
            value_text = value.isoformat()
        else:
            value_text = str(value)
        parts.append(f"{column}={value_text}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _add_technical_columns(
    df: pd.DataFrame,
    dataset: str,
    source_system: str,
    processed_at: datetime,
) -> pd.DataFrame:
    """Agrega columnas técnicas de trazabilidad."""
    result = df.copy()
    result["source_system"] = source_system
    result["source_dataset"] = dataset
    result["silver_processed_at"] = processed_at

    hash_columns = [
        column
        for column in result.columns
        if column not in {"source_system", "source_dataset", "silver_processed_at", "record_hash"}
    ]
    result["record_hash"] = result.apply(lambda row: _hash_row(row, hash_columns), axis=1)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Columnas derivadas específicas de SIAF
# ──────────────────────────────────────────────────────────────────────────────

def _add_anio_mes_key(df: pd.DataFrame, dataset_rules: dict[str, Any]) -> pd.DataFrame:
    """
    Genera anio_mes_key en formato YYYY-MM a partir de ANO_DOC y MES_DOC.

    Ejemplo: ANO_DOC=2026, MES_DOC=1 → "2026-01"
    """
    derived = dataset_rules.get("derived_columns", {}) or {}
    key_rule = derived.get("anio_mes_key", {}) or {}
    if not key_rule.get("enabled", False):
        return df

    source_cols = key_rule.get("source_columns", ["ANO_DOC", "MES_DOC"])
    if not all(c in df.columns for c in source_cols):
        df["anio_mes_key"] = pd.NA
        return df

    ano_col, mes_col = source_cols[0], source_cols[1]

    def _build_key(row: pd.Series) -> Any:
        if pd.notna(row[ano_col]) and pd.notna(row[mes_col]):
            try:
                ano = int(float(str(row[ano_col]).replace(".0", "")))
                mes = int(float(str(row[mes_col]).replace(".0", "")))
                return f"{ano:04d}-{mes:02d}"
            except (ValueError, TypeError):
                return pd.NA
        return pd.NA

    df["anio_mes_key"] = df.apply(_build_key, axis=1).astype("string")
    return df


def _add_ubigeo_ejecutora(df: pd.DataFrame, dataset_rules: dict[str, Any]) -> pd.DataFrame:
    """
    Genera ubigeo_ejecutora de 6 dígitos concatenando los tres códigos geográficos.

    Ejemplo: DPTO=15, PROV=01, DIST=31 → "150131"
    Esto facilita el join con RENAMU en la capa Gold.
    """
    derived = dataset_rules.get("derived_columns", {}) or {}
    key_rule = derived.get("ubigeo_ejecutora", {}) or {}
    if not key_rule.get("enabled", False):
        return df

    source_cols = key_rule.get("source_columns", ["DEPARTAMENTO_EJECUTORA", "PROVINCIA_EJECUTORA", "DISTRITO_EJECUTORA"])
    if not all(c in df.columns for c in source_cols):
        df["ubigeo_ejecutora"] = pd.NA
        return df

    dpto_col, prov_col, dist_col = source_cols[0], source_cols[1], source_cols[2]

    def _build_ubigeo(row: pd.Series) -> Any:
        try:
            dpto = str(row[dpto_col]).strip().replace(".0", "").zfill(2)
            prov = str(row[prov_col]).strip().replace(".0", "").zfill(2)
            dist = str(row[dist_col]).strip().replace(".0", "").zfill(2)
            if dpto and prov and dist and dpto != "nan" and prov != "nan" and dist != "nan":
                return f"{dpto}{prov}{dist}"
        except (ValueError, TypeError):
            pass
        return pd.NA

    df["ubigeo_ejecutora"] = df.apply(_build_ubigeo, axis=1).astype("string")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Transformer principal
# ──────────────────────────────────────────────────────────────────────────────

class SiafSilverTransformer:
    """
    Transformador Bronze → Silver para SIAF Ingresos.
    Lee reglas exclusivamente desde silver_rules_siaf.yaml.
    No hardcodea ninguna transformación.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.defaults = config.get("silver_defaults", {})
        self.datasets = config.get("datasets", {})
        self.source_system = config.get("source_system", "SIAF")
        self.bronze_base_path = Path(self.defaults.get("bronze_base_path", "data/bronze/siaf"))
        self.silver_base_path = Path(self.defaults.get("silver_base_path", "data/silver/siaf"))
        self.null_settings = self.defaults.get("null_handling", {}) or {}
        self.row_policy = self.defaults.get("row_policy", {}) or {}
        self.text_norm = self.defaults.get("text_normalization", {}) or {}

    def _discover_bronze_files(self) -> dict[str, Path]:
        """
        Descubre todos los Parquet Bronze disponibles en bronze_base_path.

        Retorna un dict {nombre_dataset: ruta_parquet}.
        """
        discovered: dict[str, Path] = {}
        if not self.bronze_base_path.exists():
            logger.warning("No existe la ruta Bronze SIAF: %s", self.bronze_base_path)
            return discovered

        for subdir in sorted(self.bronze_base_path.iterdir()):
            if not subdir.is_dir():
                continue
            candidates = [
                subdir / f"{subdir.name}.parquet",
                subdir / f"{subdir.name}_raw.parquet",
            ] + sorted(subdir.glob("*.parquet"))
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    discovered[subdir.name] = candidate
                    break

        return discovered

    def run(self, processed_at: datetime | None = None) -> tuple[list[DatasetSilverResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta la transformación Silver para todos los datasets configurados.

        Para SIAF, el dataset configurado en el YAML es 'ingresos', lo que combina
        todos los Parquets Bronze disponibles en una única capa Silver consolidada.
        """
        processed_at = processed_at or datetime.now()
        results: list[DatasetSilverResult] = []

        # Si el YAML configura 'ingresos', lo procesamos como dataset combinado.
        # Esto permite cargas incrementales: cada año nuevo llega como Parquet separado
        # pero Silver los unifica.
        for dataset, dataset_rules in self.datasets.items():
            result = self.transform_dataset(dataset, dataset_rules, processed_at)
            results.append(result)

        summary_df = self.build_summary(results)
        detail_df = self.build_detail(results)
        dictionary_df = self.build_data_dictionary(results)
        return results, summary_df, detail_df, dictionary_df

    def transform_dataset(
        self,
        dataset: str,
        dataset_rules: dict[str, Any],
        processed_at: datetime,
    ) -> DatasetSilverResult:
        """
        Transforma el dataset indicado y guarda el Parquet Silver.

        Para 'ingresos' combina todos los Parquets Bronze disponibles.
        """
        # ── Carga de Bronze ───────────────────────────────────────────────────
        bronze_files = self._discover_bronze_files()

        if not bronze_files:
            raise FileNotFoundError(
                f"No se encontraron Parquet Bronze bajo {self.bronze_base_path}"
            )

        logger.info("Cargando Bronze SIAF: %d archivos encontrados.", len(bronze_files))

        frames = []
        for name, path in bronze_files.items():
            try:
                df_part = pd.read_parquet(path)
                frames.append(df_part)
                logger.info("  Cargado: %s (%d filas)", path.name, len(df_part))
            except Exception as exc:
                logger.warning("  Error leyendo %s: %s", path, exc)

        if not frames:
            raise RuntimeError("No se pudo leer ningún Parquet Bronze SIAF.")

        bronze_df = pd.concat(frames, ignore_index=True)
        logger.info("Bronze combinado: %d filas totales.", len(bronze_df))

        bronze_rows = int(len(bronze_df))
        bronze_columns = int(len(bronze_df.columns))
        nulls_before = int(bronze_df.isna().sum().sum())
        original_dtypes = {col: str(dtype) for col, dtype in bronze_df.dtypes.items()}

        # ── Uso del primer archivo como referencia para bronze_path ───────────
        bronze_path_ref = next(iter(bronze_files.values()))

        df = bronze_df.copy()
        notes: list[str] = []
        detail_rows: list[dict[str, Any]] = []

        # ── 1. Normalización de tokens vacíos a null ──────────────────────────
        if self.null_settings.get("convert_missing_tokens_to_null", True):
            df = _normalize_missing_tokens(df, self.null_settings.get("missing_tokens", []))

        # ── 2. Eliminar columnas declaradas en drop_columns ───────────────────
        dropped_columns: list[str] = []
        for column in _as_list(dataset_rules.get("drop_columns")):
            if column in df.columns:
                dropped_columns.append(column)
                detail_rows.append({
                    "dataset": dataset,
                    "column_name": column,
                    "bronze_type": original_dtypes.get(column, "unknown"),
                    "silver_type": "dropped",
                    "transformation": "drop_column",
                    "nulls_before": int(bronze_df[column].isna().sum()),
                    "nulls_after": None,
                    "conversion_errors": 0,
                    "special_values_handled": 0,
                    "status": "DROPPED",
                    "reason": _reason_from_mapping(dataset_rules.get("drop_columns"), column),
                })
                df = df.drop(columns=[column])

        # ── 3. Eliminar filas completamente vacías ────────────────────────────
        if self.row_policy.get("drop_empty_rows", True):
            before = len(df)
            df = df.dropna(how="all")
            dropped_empty = before - len(df)
            if dropped_empty > 0:
                notes.append(f"Filas completamente vacías eliminadas: {dropped_empty}")
                logger.info("Filas completamente vacías eliminadas: %d", dropped_empty)

        # ── 4. Eliminar filas duplicadas si la política lo indica ─────────────
        if self.row_policy.get("drop_duplicate_rows", False):
            before = len(df)
            df = df.drop_duplicates()
            dropped_dupes = before - len(df)
            if dropped_dupes > 0:
                notes.append(f"Filas duplicadas eliminadas: {dropped_dupes}")

        # ── 5. Normalización de texto (trim) ──────────────────────────────────
        apply_trim = self.text_norm.get("apply_trim", True)
        apply_upper = self.text_norm.get("apply_upper", False)

        if apply_trim or apply_upper:
            for column in df.columns:
                if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                    if apply_trim:
                        df[column] = df[column].astype("string").str.strip()
                    if apply_upper:
                        df[column] = df[column].astype("string").str.upper()

        # ── 6. Conversión de tipos ─────────────────────────────────────────────
        conversion_errors_total = 0
        preserve_as_string = dataset_rules.get("preserve_as_string", {}) or {}
        all_configured_columns = set(
            _as_list(dataset_rules.get("integer_columns"))
            + _as_list(dataset_rules.get("decimal_columns"))
            + _as_list(dataset_rules.get("datetime_columns"))
            + _as_list(dataset_rules.get("string_columns"))
        )

        for column in df.columns:
            if column in {"source_system", "source_dataset", "silver_processed_at", "record_hash",
                          "anio_mes_key", "ubigeo_ejecutora"}:
                continue

            # Columnas en preserve_as_string se tratan como string aunque estén
            # en integer_columns del YAML (esto no aplica aquí ya que el YAML
            # las declara como string; el check es por seguridad extra).
            if column in preserve_as_string:
                target_type = "string"
            else:
                target_type = _get_column_target_type(dataset_rules, column)

            bronze_type = original_dtypes.get(column, str(df[column].dtype))
            nulls_col_before = int(df[column].isna().sum())
            errors = 0
            transformation = "unchanged"

            if target_type == "integer":
                df[column], errors = _convert_to_integer(df[column])
                transformation = "cast_to_integer_nullable"
            elif target_type == "decimal":
                df[column], errors = _convert_to_decimal(df[column])
                transformation = "cast_to_decimal_nullable"
            elif target_type == "datetime":
                df[column], errors = _convert_to_datetime(df[column])
                transformation = "cast_to_datetime_nullable"
            elif target_type == "string":
                df[column] = _convert_to_string(df[column])
                transformation = "cast_to_string_nullable"
            else:
                transformation = "kept_without_explicit_rule"

            conversion_errors_total += errors
            nulls_col_after = int(df[column].isna().sum())
            status = "OK" if errors == 0 else "CONVERSION_ERRORS"

            reason_parts = []
            for section_name in ["documented_null_columns", "preserve_as_string"]:
                reason = _reason_from_mapping(dataset_rules.get(section_name), column)
                if reason:
                    reason_parts.append(f"{section_name}: {reason}")

            detail_rows.append({
                "dataset": dataset,
                "column_name": column,
                "bronze_type": bronze_type,
                "silver_type": str(df[column].dtype),
                "transformation": transformation,
                "nulls_before": nulls_col_before,
                "nulls_after": nulls_col_after,
                "conversion_errors": errors,
                "special_values_handled": 0,
                "status": status,
                "reason": " | ".join(reason_parts),
            })

        # ── 7. Advertir columnas configuradas que no existen en Bronze ─────────
        missing_configured = sorted([
            c for c in all_configured_columns
            if c not in df.columns and c not in dropped_columns
        ])
        for column in missing_configured:
            notes.append(f"Columna configurada no encontrada en Bronze: {column}")
            detail_rows.append({
                "dataset": dataset,
                "column_name": column,
                "bronze_type": "missing",
                "silver_type": "missing",
                "transformation": "configured_column_not_found",
                "nulls_before": None,
                "nulls_after": None,
                "conversion_errors": 0,
                "special_values_handled": 0,
                "status": "WARNING",
                "reason": "Columna declarada en YAML, pero no existe en el Parquet Bronze.",
            })

        # ── 8. Columnas derivadas específicas de SIAF ─────────────────────────
        df = _add_anio_mes_key(df, dataset_rules)
        df = _add_ubigeo_ejecutora(df, dataset_rules)

        # ── 9. Columnas técnicas de trazabilidad ──────────────────────────────
        if self.defaults.get("add_technical_columns", True):
            df = _add_technical_columns(df, dataset, self.source_system, processed_at)

        # ── Guardar Parquet Silver ────────────────────────────────────────────
        silver_dir = self.silver_base_path
        silver_dir.mkdir(parents=True, exist_ok=True)
        silver_path = silver_dir / f"{dataset}_silver.parquet"

        df.to_parquet(silver_path, index=False)
        logger.info("Parquet Silver guardado: %s (%d filas)", silver_path, len(df))

        nulls_after = int(df.isna().sum().sum())
        silver_rows = int(len(df))
        silver_columns = int(len(df.columns))
        status_final = "OK" if conversion_errors_total == 0 else "WITH_CONVERSION_ERRORS"

        return DatasetSilverResult(
            dataset=dataset,
            bronze_path=bronze_path_ref,
            silver_path=silver_path,
            bronze_rows=bronze_rows,
            silver_rows=silver_rows,
            bronze_columns=bronze_columns,
            silver_columns=silver_columns,
            detail_rows=detail_rows,
            dropped_columns=dropped_columns,
            conversion_errors=conversion_errors_total,
            nulls_before=nulls_before,
            nulls_after=nulls_after,
            status=status_final,
            notes=notes,
        )

    def build_summary(self, results: list[DatasetSilverResult]) -> pd.DataFrame:
        """Construye resumen una fila por dataset."""
        rows = []
        for result in results:
            detail = pd.DataFrame(result.detail_rows)
            numeric_conversions = int(detail[detail["transformation"].isin(["cast_to_integer_nullable", "cast_to_decimal_nullable"])].shape[0]) if not detail.empty else 0
            date_conversions = int(detail[detail["transformation"].eq("cast_to_datetime_nullable")].shape[0]) if not detail.empty else 0
            converted = int(detail[detail["transformation"].str.startswith("cast_to", na=False)].shape[0]) if not detail.empty else 0
            rows.append({
                "dataset": result.dataset,
                "bronze_rows": result.bronze_rows,
                "silver_rows": result.silver_rows,
                "bronze_columns": result.bronze_columns,
                "silver_columns": result.silver_columns,
                "columns_converted": converted,
                "date_columns_converted": date_conversions,
                "numeric_columns_converted": numeric_conversions,
                "technical_columns_added": 6,  # source_system, source_dataset, silver_processed_at, record_hash, anio_mes_key, ubigeo_ejecutora
                "columns_dropped": len(result.dropped_columns),
                "dropped_columns": ", ".join(result.dropped_columns),
                "conversion_errors": result.conversion_errors,
                "nulls_before": result.nulls_before,
                "nulls_after": result.nulls_after,
                "status": result.status,
                "notes": " | ".join(result.notes),
                "silver_path": str(result.silver_path),
            })
        return pd.DataFrame(rows)

    def build_detail(self, results: list[DatasetSilverResult]) -> pd.DataFrame:
        """Construye detalle una fila por columna."""
        rows = []
        for result in results:
            rows.extend(result.detail_rows)
        return pd.DataFrame(rows)

    def build_data_dictionary(self, results: list[DatasetSilverResult]) -> pd.DataFrame:
        """Construye el diccionario de datos Silver basado en reglas y resultados."""
        rows: list[dict[str, Any]] = []
        detail_df = self.build_detail(results)

        for _, row in detail_df.iterrows():
            dataset = row["dataset"]
            column = row["column_name"]
            dataset_rules = self.datasets.get(dataset, {})
            is_dropped = row["status"] == "DROPPED"
            nullable = None if is_dropped else (row.get("nulls_after", 0) not in [0, "0", None])

            # Descripción desde columnas técnicas o preserve_as_string
            description = row.get("reason", "")

            rows.append({
                "dataset": dataset,
                "column_name": column,
                "business_name": column,
                "description": description,
                "bronze_type": row.get("bronze_type", ""),
                "silver_type": row.get("silver_type", ""),
                "transformation_applied": row.get("transformation", ""),
                "nullable": nullable,
                "is_technical_column": False,
                "source": "Bronze SIAF + silver_rules_siaf.yaml",
                "notes": row.get("reason", ""),
            })
            _ = dataset_rules  # referencia documentada; puede usarse en Gold

        # Columnas técnicas
        technical_descriptions = self.defaults.get("technical_columns", {}) or {}
        for result in results:
            for tech_col, default_desc in [
                ("source_system", "Sistema de origen de los datos (SIAF Ingresos MEF)."),
                ("source_dataset", "Dataset Bronze del cual proviene el registro."),
                ("silver_processed_at", "Timestamp de procesamiento hacia Silver."),
                ("record_hash", "Hash SHA-256 para trazabilidad y detección de cambios."),
                ("anio_mes_key", "Clave temporal YYYY-MM derivada de ANO_DOC y MES_DOC."),
                ("ubigeo_ejecutora", "UBIGEO de 6 dígitos reconstruido para join con RENAMU en Gold."),
            ]:
                desc = technical_descriptions.get(tech_col, {}).get("description", default_desc)
                rows.append({
                    "dataset": result.dataset,
                    "column_name": tech_col,
                    "business_name": tech_col,
                    "description": desc,
                    "bronze_type": "not_in_bronze",
                    "silver_type": "technical",
                    "transformation_applied": "added_technical_column",
                    "nullable": False if tech_col not in {"anio_mes_key", "ubigeo_ejecutora"} else True,
                    "is_technical_column": True,
                    "source": "silver_pipeline_siaf",
                    "notes": desc,
                })

        return pd.DataFrame(rows)