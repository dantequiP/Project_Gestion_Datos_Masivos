from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class DatasetSilverResult:
    """Resultado de transformación Silver para un dataset."""

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
    notes: list[str]


def load_silver_rules(path: Path) -> dict[str, Any]:
    """Carga el archivo YAML con reglas Silver."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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

    Se aplica especialmente sobre columnas object/string para no alterar valores numéricos válidos como 0.
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


def _apply_special_values_to_null(df: pd.DataFrame, dataset_rules: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convierte valores especiales definidos en YAML a null y devuelve conteos tratados."""
    result = df.copy()
    handled: dict[str, int] = {}
    special_rules = dataset_rules.get("special_values_to_null", {}) or {}

    for column, rule in special_rules.items():
        if column not in result.columns:
            handled[column] = 0
            continue

        values = [str(value) for value in rule.get("values", [])]
        series_as_text = result[column].astype("string").str.strip()
        mask = series_as_text.isin(values)
        count = int(mask.sum())
        result.loc[mask, column] = pd.NA
        handled[column] = count

    return result, handled


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
    """
    Convierte una columna a datetime nullable.

    Se usa dayfirst=True porque las fuentes MEF suelen venir en formato dd/mm/yyyy.
    """
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


def _add_anio_periodo_key(df: pd.DataFrame, dataset_rules: dict[str, Any]) -> pd.DataFrame:
    """Agrega anio_periodo_key cuando está habilitado y existen las columnas fuente."""
    derived = dataset_rules.get("derived_columns", {}) or {}
    key_rule = derived.get("anio_periodo_key", {}) or {}
    if not key_rule.get("enabled", False):
        return df

    source_columns = key_rule.get("source_columns", ["ANO_APLICACION", "PERIODO"])
    if not all(column in df.columns for column in source_columns):
        df["anio_periodo_key"] = pd.NA
        return df

    left, right = source_columns[0], source_columns[1]
    df["anio_periodo_key"] = df.apply(
        lambda row: (
            f"{_format_key_value(row[left])}-{_format_key_value(row[right])}"
            if pd.notna(row[left]) and pd.notna(row[right])
            else pd.NA
        ),
        axis=1,
    ).astype("string")
    return df


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

    # El hash se calcula antes de crear record_hash y excluye columnas técnicas de proceso.
    hash_columns = [
        column
        for column in result.columns
        if column not in {"source_system", "source_dataset", "silver_processed_at", "record_hash"}
    ]
    result["record_hash"] = result.apply(lambda row: _hash_row(row, hash_columns), axis=1)
    return result


def _get_column_target_type(dataset_rules: dict[str, Any], column: str) -> str:
    """Devuelve el tipo objetivo definido para una columna."""
    if column in _as_list(dataset_rules.get("integer_columns")):
        return "integer"
    if column in _as_list(dataset_rules.get("decimal_columns")):
        return "decimal"
    if column in _as_list(dataset_rules.get("datetime_columns")):
        return "datetime"
    if column in _as_list(dataset_rules.get("string_columns")):
        return "string"
    return "unchanged"


class SilverTransformer:
    """Transformador de Bronze a Silver basado en reglas YAML."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.defaults = config.get("silver_defaults", {})
        self.datasets = config.get("datasets", {})
        self.source_system = config.get("source_system", self.defaults.get("technical_columns", {}).get("source_system", {}).get("value", "SISMEPRE"))
        self.bronze_base_path = Path(self.defaults.get("bronze_base_path", "data/bronze/sismepre"))
        self.silver_base_path = Path(self.defaults.get("silver_base_path", "data/silver/sismepre"))
        self.null_settings = self.defaults.get("null_handling", {}) or {}
        self.row_policy = self.defaults.get("row_policy", {}) or {}

    def run(self, processed_at: datetime | None = None) -> tuple[list[DatasetSilverResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Ejecuta la transformación Silver para todos los datasets configurados."""
        processed_at = processed_at or datetime.now()
        results: list[DatasetSilverResult] = []

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
        """Transforma un dataset específico y guarda el Parquet Silver."""
        bronze_path = self.bronze_base_path / dataset / f"{dataset}_raw.parquet"
        silver_dir = self.silver_base_path / dataset
        silver_path = silver_dir / f"{dataset}_silver.parquet"
        silver_dir.mkdir(parents=True, exist_ok=True)

        if not bronze_path.exists():
            raise FileNotFoundError(f"No existe el Parquet Bronze esperado: {bronze_path}")

        bronze_df = pd.read_parquet(bronze_path)
        bronze_rows = int(len(bronze_df))
        bronze_columns = int(len(bronze_df.columns))
        nulls_before = int(bronze_df.isna().sum().sum())
        original_dtypes = {column: str(dtype) for column, dtype in bronze_df.dtypes.items()}

        df = bronze_df.copy()
        notes: list[str] = []
        detail_rows: list[dict[str, Any]] = []

        # 1. Normalización general de tokens vacíos a null.
        if self.null_settings.get("convert_missing_tokens_to_null", True):
            df = _normalize_missing_tokens(df, self.null_settings.get("missing_tokens", []))

        # 2. Valores especiales a null antes de convertir tipos, por ejemplo fecha="0".
        df, special_to_null_counts = _apply_special_values_to_null(df, dataset_rules)

        # 3. Eliminar columnas aprobadas en reglas Silver.
        dropped_columns: list[str] = []
        for column in _as_list(dataset_rules.get("drop_columns")):
            if column in df.columns:
                dropped_columns.append(column)
                detail_rows.append(
                    {
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
                    }
                )
                df = df.drop(columns=[column])

        # 4. Eliminar filas completamente vacías si la política global lo indica.
        if self.row_policy.get("drop_empty_rows", True):
            before_drop = len(df)
            df = df.dropna(how="all")
            dropped_empty = before_drop - len(df)
            if dropped_empty:
                notes.append(f"Filas completamente vacías eliminadas: {dropped_empty}")

        # 5. Conversión de tipos por columna.
        conversion_errors_total = 0
        all_configured_columns = set(
            _as_list(dataset_rules.get("integer_columns"))
            + _as_list(dataset_rules.get("decimal_columns"))
            + _as_list(dataset_rules.get("datetime_columns"))
            + _as_list(dataset_rules.get("string_columns"))
        )

        for column in df.columns:
            if column in {"source_system", "source_dataset", "silver_processed_at", "record_hash", "anio_periodo_key"}:
                continue

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
                # Columnas no declaradas: se conservan sin conversión, pero se reportan.
                transformation = "kept_without_explicit_rule"

            conversion_errors_total += errors
            special_handled = int(special_to_null_counts.get(column, 0))
            nulls_col_after = int(df[column].isna().sum())
            status = "OK" if errors == 0 else "CONVERSION_ERRORS"

            reason_parts = []
            for section_name in ["documented_null_columns", "preserve_as_string", "preserve_zero_values", "special_values", "special_values_to_null"]:
                reason = _reason_from_mapping(dataset_rules.get(section_name), column)
                if reason:
                    reason_parts.append(f"{section_name}: {reason}")

            detail_rows.append(
                {
                    "dataset": dataset,
                    "column_name": column,
                    "bronze_type": bronze_type,
                    "silver_type": str(df[column].dtype),
                    "transformation": transformation,
                    "nulls_before": nulls_col_before,
                    "nulls_after": nulls_col_after,
                    "conversion_errors": errors,
                    "special_values_handled": special_handled,
                    "status": status,
                    "reason": " | ".join(reason_parts),
                }
            )

        # 6. Advertir columnas configuradas que no existen en el Parquet.
        missing_configured_columns = sorted([column for column in all_configured_columns if column not in df.columns and column not in dropped_columns])
        for column in missing_configured_columns:
            notes.append(f"Columna configurada no encontrada en Bronze: {column}")
            detail_rows.append(
                {
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
                }
            )

        # 7. Columnas derivadas y técnicas.
        df = _add_anio_periodo_key(df, dataset_rules)
        if self.defaults.get("add_technical_columns", True):
            df = _add_technical_columns(df, dataset, self.source_system, processed_at)

        nulls_after = int(df.isna().sum().sum())
        silver_rows = int(len(df))
        silver_columns = int(len(df.columns))
        status = "OK" if conversion_errors_total == 0 else "WITH_CONVERSION_ERRORS"

        df.to_parquet(silver_path, index=False)

        return DatasetSilverResult(
            dataset=dataset,
            bronze_path=bronze_path,
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
            status=status,
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
            rows.append(
                {
                    "dataset": result.dataset,
                    "bronze_rows": result.bronze_rows,
                    "silver_rows": result.silver_rows,
                    "bronze_columns": result.bronze_columns,
                    "silver_columns": result.silver_columns,
                    "columns_converted": converted,
                    "date_columns_converted": date_conversions,
                    "numeric_columns_converted": numeric_conversions,
                    "technical_columns_added": 5,
                    "columns_dropped": len(result.dropped_columns),
                    "dropped_columns": ", ".join(result.dropped_columns),
                    "conversion_errors": result.conversion_errors,
                    "nulls_before": result.nulls_before,
                    "nulls_after": result.nulls_after,
                    "status": result.status,
                    "notes": " | ".join(result.notes),
                    "silver_path": str(result.silver_path),
                }
            )
        return pd.DataFrame(rows)

    def build_detail(self, results: list[DatasetSilverResult]) -> pd.DataFrame:
        """Construye detalle una fila por columna."""
        rows = []
        for result in results:
            rows.extend(result.detail_rows)
        return pd.DataFrame(rows)

    def build_data_dictionary(self, results: list[DatasetSilverResult]) -> pd.DataFrame:
        """Construye un diccionario de datos Silver basado en reglas y resultados."""
        rows: list[dict[str, Any]] = []
        detail_df = self.build_detail(results)

        for _, row in detail_df.iterrows():
            dataset = row["dataset"]
            column = row["column_name"]
            dataset_rules = self.datasets.get(dataset, {})
            is_dropped = row["status"] == "DROPPED"
            is_technical = False
            nullable = None if is_dropped else row.get("nulls_after", 0) not in [0, "0", None]

            rows.append(
                {
                    "dataset": dataset,
                    "column_name": column,
                    "business_name": column,
                    "description": row.get("reason", ""),
                    "bronze_type": row.get("bronze_type", ""),
                    "silver_type": row.get("silver_type", ""),
                    "transformation_applied": row.get("transformation", ""),
                    "nullable": nullable,
                    "is_technical_column": is_technical,
                    "source": "Bronze SISMEPRE + silver_rules_sismepre.yaml",
                    "notes": row.get("reason", ""),
                }
            )

            # Evita warning por variable no usada; se deja por claridad para futuras descripciones oficiales.
            _ = dataset_rules

        technical_descriptions = self.defaults.get("technical_columns", {}) or {}
        for result in results:
            for tech_col in ["source_system", "source_dataset", "silver_processed_at", "anio_periodo_key", "record_hash"]:
                desc = technical_descriptions.get(tech_col, {}).get("description", "Columna técnica de trazabilidad Silver.")
                rows.append(
                    {
                        "dataset": result.dataset,
                        "column_name": tech_col,
                        "business_name": tech_col,
                        "description": desc,
                        "bronze_type": "not_in_bronze",
                        "silver_type": "technical",
                        "transformation_applied": "added_technical_column",
                        "nullable": False if tech_col != "anio_periodo_key" else True,
                        "is_technical_column": True,
                        "source": "silver_pipeline",
                        "notes": desc,
                    }
                )

        return pd.DataFrame(rows)
