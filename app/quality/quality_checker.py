from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


@dataclass
class RuleResult:
    """
    Resultado estándar de una regla de calidad.

    Cada regla devuelve:
    - filas evaluadas
    - filas que cumplen
    - filas observadas/fallidas
    - porcentaje de cumplimiento
    - severidad
    - fuente que justifica la regla
    - muestra de filas observadas
    """
    dataset: str
    dimension: str
    rule_id: str
    rule_name: str
    column_name: str
    rule_type: str
    expected_value: str
    evaluated_rows: int
    passed_rows: int
    failed_rows: int
    score: float | None
    severity: str
    source: str
    status: str
    observation: str
    sample_failed_rows: str


def safe_strip(value: Any) -> str:
    """Convierte valores nulos o mixtos a texto seguro para evaluación."""
    if pd.isna(value):
        return ""
    return str(value).strip()


class QualityChecker:
    """
    Motor de evaluación de calidad.

    Importante:
    - No modifica datos Bronze.
    - No limpia ni corrige datos.
    - Solo evalúa reglas y produce resultados trazables.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.settings = config["settings"]
        self.max_samples = int(self.settings.get("max_failed_samples_per_rule", 10))
        self.missing_tokens = set(config.get("missing_tokens", []))
        self.bronze_root = Path(self.settings.get("bronze_root", "data/bronze"))
        self.datasets_config = config.get("datasets", {})
        self.global_rules = config.get("global_rules", {})
        self.referential_rules = config.get("referential_integrity", [])
        self.current_year = datetime.now().year
        self.current_timestamp = pd.Timestamp.now()
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.failed_samples: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # Carga y normalización básica
    # ------------------------------------------------------------
    def load_dataset(self, dataset: str) -> pd.DataFrame:
        """Carga el Parquet Bronze de un dataset SISMEPRE."""
        if dataset in self.dataframes:
            return self.dataframes[dataset]

        path = self.bronze_root / "sismepre" / dataset / f"{dataset}_raw.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No existe Parquet Bronze para {dataset}: {path}")

        df = pd.read_parquet(path)
        self.dataframes[dataset] = df
        return df

    def clean_series(self, series: pd.Series) -> pd.Series:
        """Convierte una serie a texto limpio, conservando índice."""
        return series.map(safe_strip)

    def missing_mask(self, series: pd.Series) -> pd.Series:
        """Detecta nulos, cadenas vacías y tokens configurados como missing."""
        clean = self.clean_series(series)
        return series.isna() | clean.isin(self.missing_tokens)

    # ------------------------------------------------------------
    # Evidencia de filas observadas
    # ------------------------------------------------------------
    def _sample_indices(self, failed_mask: pd.Series | None) -> list[int | str]:
        """Obtiene una muestra limitada de índices de filas observadas."""
        if failed_mask is None or len(failed_mask) == 0:
            return []
        idx = failed_mask[failed_mask].index[: self.max_samples]
        return [int(i) if isinstance(i, (int, float)) and not pd.isna(i) else str(i) for i in idx]

    def _register_failed_samples(
        self,
        dataset: str,
        rule_id: str,
        dimension: str,
        column_name: str,
        expected_value: Any,
        df: pd.DataFrame | None,
        failed_mask: pd.Series | None,
        value_columns: list[str] | None = None,
    ) -> str:
        """
        Guarda ejemplos de filas observadas para trazabilidad.

        No guarda todo el dataset afectado para evitar reportes demasiado pesados.
        """
        sample_indices = self._sample_indices(failed_mask)
        if not sample_indices or df is None:
            return "[]"

        cols = value_columns or ([column_name] if column_name in df.columns else [])
        for idx in sample_indices:
            sample = {
                "dataset": dataset,
                "rule_id": rule_id,
                "dimension": dimension,
                "column_name": column_name,
                "row_index": idx,
                "expected_value": str(expected_value),
            }
            if idx in df.index:
                for col in cols:
                    if col in df.columns:
                        sample[col] = safe_strip(df.at[idx, col])
            self.failed_samples.append(sample)

        return json.dumps(sample_indices, ensure_ascii=False)

    def result(
        self,
        dataset: str,
        dimension: str,
        rule_id: str,
        rule_name: str,
        column_name: str,
        rule_type: str,
        expected_value: Any,
        evaluated_rows: int,
        failed_rows: int,
        severity: str,
        source: str,
        observation: str,
        df: pd.DataFrame | None = None,
        failed_mask: pd.Series | None = None,
        value_columns: list[str] | None = None,
    ) -> RuleResult:
        """Construye un resultado estándar a partir del conteo de fallos."""
        evaluated_rows, failed_rows = int(evaluated_rows), int(failed_rows)
        passed_rows = max(evaluated_rows - failed_rows, 0)

        if evaluated_rows == 0:
            score = None
            status = "No evaluable"
        else:
            score = round((passed_rows / evaluated_rows) * 100, 4)
            status = "Sin observaciones" if failed_rows == 0 else "Con observaciones"

        sample_failed_rows = self._register_failed_samples(
            dataset=dataset,
            rule_id=rule_id,
            dimension=dimension,
            column_name=column_name,
            expected_value=expected_value,
            df=df,
            failed_mask=failed_mask,
            value_columns=value_columns,
        )

        return RuleResult(
            dataset=dataset,
            dimension=dimension,
            rule_id=rule_id,
            rule_name=rule_name,
            column_name=column_name,
            rule_type=rule_type,
            expected_value=str(expected_value),
            evaluated_rows=evaluated_rows,
            passed_rows=passed_rows,
            failed_rows=failed_rows,
            score=score,
            severity=severity,
            source=source,
            status=status,
            observation=observation,
            sample_failed_rows=sample_failed_rows,
        )

    # ------------------------------------------------------------
    # Reglas base: completitud, validez, exactitud y razonabilidad
    # ------------------------------------------------------------
    def required_column(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Completitud: columna obligatoria no vacía."""
        dimension = meta.get("dimension", "Completitud")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.required"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Columna requerida presente y no vacía: {column}", column, "required_not_empty", "no nulo/no vacío", 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        failed_mask = self.missing_mask(df[column])
        return self.result(dataset, dimension, rule_id, f"Columna requerida no vacía: {column}", column, "required_not_empty", "no nulo/no vacío", len(df), int(failed_mask.sum()), severity, source, f"Registros vacíos o nulos en {column}.", df, failed_mask, [column])

    def observed_empty(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Completitud exploratoria: columnas detectadas como vacías por profiling."""
        dimension = meta.get("dimension", "Completitud")
        severity = meta.get("severity", "Baja")
        source = meta.get("source", "profiling_based")
        rule_id = f"{dataset}.{column}.observed_empty"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Columna observada como vacía: {column}", column, "observed_empty", "revisar ausencia", 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        failed_mask = self.missing_mask(df[column])
        return self.result(dataset, dimension, rule_id, f"Columna con ausencia observada: {column}", column, "observed_empty", "revisar ausencia", len(df), int(failed_mask.sum()), severity, source, "Columna reportada por profiling como potencialmente vacía o incompleta.", df, failed_mask, [column])

    def allowed_values(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Validez: valores dentro de un catálogo permitido."""
        values = [str(v) for v in meta.get("values", [])]
        dimension = meta.get("dimension", "Validez")
        severity = meta.get("severity", "Media")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.allowed_values"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Valores permitidos en {column}", column, "allowed_values", values, 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        failed_mask = evaluable & ~clean.isin(values)

        return self.result(dataset, dimension, rule_id, f"Valores permitidos en {column}", column, "allowed_values", values, int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Valores fuera del catálogo permitido {values}. Los vacíos se evalúan en Completitud.", df, failed_mask, [column])

    def numeric_column(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any], integer: bool = False) -> RuleResult:
        """Validez: columna convertible a número."""
        dimension = meta.get("dimension", "Validez")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.numeric"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Columna numérica válida: {column}", column, "numeric", "numérico", 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_sub = numeric.isna()

        if integer and len(numeric) > 0:
            failed_sub = failed_sub | ((numeric.dropna() % 1 != 0).reindex(numeric.index, fill_value=False))

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Columna convertible a número: {column}", column, "numeric", "numérico", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores no convertibles a número. Los vacíos se evalúan en Completitud.", df, failed_mask, [column])

    def regex_rule(self, dataset: str, df: pd.DataFrame, column: str, regex: str, meta: dict[str, Any], suffix: str) -> RuleResult:
        """Exactitud/Validez: formato esperado por expresión regular."""
        dimension = meta.get("dimension", "Validez")
        severity = meta.get("severity", "Media")
        source = meta.get("source", "naming_pattern")
        rule_id = f"{dataset}.{column}.{suffix}"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Formato esperado en {column}", column, "regex", regex, 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        failed_mask = evaluable & ~clean.str.match(regex, na=False)

        return self.result(dataset, dimension, rule_id, f"Formato esperado en {column}", column, "regex", regex, int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Valores que no cumplen el patrón {regex}. Los vacíos se evalúan en Completitud.", df, failed_mask, [column])

    def year_rule(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """
        Exactitud: año con formato de cuatro dígitos.

        Nota metodológica:
        Esta regla no determina si el año pertenece al periodo SISMEPRE.
        Esa evaluación se hace en Oportunidad.
        """
        min_year = int(meta.get("min_year", 1000))
        max_year = int(meta.get("max_year", 9999))
        dimension = meta.get("dimension", "Exactitud")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "naming_pattern")
        rule_id = f"{dataset}.{column}.year_4_digits"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Año válido: {column}", column, "year_4_digits", f"{min_year}-{max_year}", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        year_text = clean[evaluable].str.replace(r"\.0$", "", regex=True)
        numeric = pd.to_numeric(year_text, errors="coerce")
        failed_sub = numeric.isna() | (~year_text.str.match(r"^\d{4}$", na=False)) | (numeric < min_year) | (numeric > max_year)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Año con formato de 4 dígitos: {column}", column, "year_4_digits", f"{min_year}-{max_year}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores que no tienen 4 dígitos. La vigencia temporal se evalúa en Oportunidad.", df, failed_mask, [column])

    def date_rule(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Validez: fecha parseable."""
        dimension = meta.get("dimension", "Validez")
        severity = meta.get("severity", "Media")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.date_parseable"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Fecha válida: {column}", column, "date_parseable", "fecha parseable", 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        parsed = pd.to_datetime(clean[evaluable], dayfirst=True, errors="coerce")
        failed_sub = parsed.isna()

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Fecha parseable: {column}", column, "date_parseable", "fecha válida", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores no convertibles a fecha. Los vacíos se evalúan en Completitud.", df, failed_mask, [column])

    def non_negative(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any], integer: bool = False) -> RuleResult:
        """Razonabilidad: montos/cantidades no negativos."""
        dimension = meta.get("dimension", "Razonabilidad")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "naming_pattern")
        rule_id = f"{dataset}.{column}.non_negative"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Valor no negativo: {column}", column, "non_negative", ">=0", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_sub = numeric.isna() | (numeric < 0)

        if integer and len(numeric) > 0:
            failed_sub = failed_sub | ((numeric.dropna() % 1 != 0).reindex(numeric.index, fill_value=False))

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Valor no negativo: {column}", column, "non_negative", ">=0", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores negativos o no numéricos en campo cuantitativo.", df, failed_mask, [column])

    # ------------------------------------------------------------
    # Reglas de consistencia
    # ------------------------------------------------------------
    def conditional_date_rule(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """Consistencia: si una condición se cumple, una fecha debe ser válida."""
        condition_col = rule["condition_column"]
        condition_value = str(rule["condition_value"])
        date_col = rule["date_column"]
        dimension = rule.get("dimension", "Consistencia")
        severity = rule.get("severity", "Media")
        source = rule.get("source", "embedded_dictionary")
        rule_id = f"{dataset}.{date_col}.conditional_date"

        if condition_col not in df.columns or date_col not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Fecha válida en {date_col} cuando {condition_col} = {condition_value}", f"{condition_col},{date_col}", "conditional_date", f"{date_col} fecha válida si {condition_col}={condition_value}", 1, 1, severity, source, "No existen columnas requeridas para evaluar la regla condicional.")

        condition_series = self.clean_series(df[condition_col])
        date_series = self.clean_series(df[date_col])
        mask = condition_series == condition_value
        parsed = pd.to_datetime(date_series[mask], dayfirst=True, errors="coerce")
        failed_sub = parsed.isna()

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Fecha válida en {date_col} cuando {condition_col} = {condition_value}", f"{condition_col},{date_col}", "conditional_date", f"{date_col} fecha válida si {condition_col}={condition_value}", int(mask.sum()), int(failed_mask.sum()), severity, source, f"Registros donde {condition_col}={condition_value} pero {date_col} no es fecha válida.", df, failed_mask, [condition_col, date_col])

    def less_or_equal(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """Consistencia: columna izquierda debe ser menor o igual a columna derecha."""
        left = rule["left_column"]
        right = rule["right_column"]
        dimension = rule.get("dimension", "Consistencia")
        severity = rule.get("severity", "Alta")
        source = rule.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{left}_le_{right}"

        if left not in df.columns or right not in df.columns:
            return self.result(dataset, dimension, rule_id, f"{left} <= {right}", f"{left},{right}", "less_or_equal", f"{left}<={right}", 1, 1, severity, source, "No existen columnas requeridas.")

        left_clean = self.clean_series(df[left])
        right_clean = self.clean_series(df[right])
        evaluable = ~(self.missing_mask(df[left]) | self.missing_mask(df[right]))
        left_num = pd.to_numeric(left_clean[evaluable], errors="coerce")
        right_num = pd.to_numeric(right_clean[evaluable], errors="coerce")
        failed_sub = left_num.isna() | right_num.isna() | (left_num > right_num)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Consistencia: {left} <= {right}", f"{left},{right}", "less_or_equal", f"{left}<={right}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Registros donde {left} no es menor o igual que {right}.", df, failed_mask, [left, right])

    # ------------------------------------------------------------
    # Reglas de oportunidad temporal
    # ------------------------------------------------------------
    def year_not_future(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """
        Oportunidad: el año no debe ser futuro respecto al año de ejecución.

        Esta regla detectaría casos como 2035 si el pipeline se ejecuta en 2026.
        """
        dimension = meta.get("dimension", "Oportunidad")
        severity = meta.get("severity", "Media")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.year_not_future"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Año no futuro: {column}", column, "year_not_future", f"<= {self.current_year}", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column]).str.replace(r"\.0$", "", regex=True)
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_sub = numeric.isna() | (numeric > self.current_year)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Año no futuro: {column}", column, "year_not_future", f"<= {self.current_year}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Años posteriores al año de ejecución del pipeline.", df, failed_mask, [column])

    def month_period_between(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """
        Oportunidad: periodo estadístico mensual válido.

        Para SISMEPRE se acepta 1-13. El valor 13 se documenta como
        periodo especial observado en la fuente, no como mes calendario.
        """
        dimension = meta.get("dimension", "Oportunidad")
        severity = meta.get("severity", "Media")
        source = meta.get("source", "dictionary_csv_plus_profiling")
        min_value = int(meta.get("min_value", 1))
        max_value = int(meta.get("max_value", 13))
        rule_id = f"{dataset}.{column}.month_period_between"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Periodo estadístico válido: {column}", column, "month_period_between", f"{min_value}-{max_value}", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_sub = numeric.isna() | (numeric < min_value) | (numeric > max_value) | (numeric % 1 != 0)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Periodo estadístico válido: {column}", column, "month_period_between", f"{min_value}-{max_value}; 13=periodo especial observado", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores fuera del rango de periodo estadístico permitido. Se acepta 13 como periodo especial observado.", df, failed_mask, [column])

    def date_not_future(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Oportunidad: fechas de registro/respuesta no deben ser futuras."""
        dimension = meta.get("dimension", "Oportunidad")
        severity = meta.get("severity", "Baja")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.date_not_future"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Fecha no futura: {column}", column, "date_not_future", f"<= {self.current_timestamp.date()}", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        parsed = pd.to_datetime(clean[evaluable], dayfirst=True, errors="coerce")
        failed_sub = parsed.isna() | (parsed > self.current_timestamp)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub

        return self.result(dataset, dimension, rule_id, f"Fecha no futura: {column}", column, "date_not_future", f"<= {self.current_timestamp.date()}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Fechas posteriores a la ejecución del pipeline o no convertibles a fecha.", df, failed_mask, [column])

    def temporal_catalog_exists(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """
        Oportunidad: año/periodo debe existir en catálogo temporal interno.

        Para SISMEPRE, rentas_ano_aplicacion funciona como catálogo temporal
        de años y periodos de aplicación.
        """
        columns = rule["columns"]
        ref_dataset = rule["reference_dataset"]
        ref_columns = rule["reference_columns"]
        dimension = rule.get("dimension", "Oportunidad")
        severity = rule.get("severity", "Media")
        source = rule.get("source", "internal_temporal_catalog")
        rule_id = f"{dataset}.{'+'.join(columns)}.temporal_catalog_exists"

        missing_left = [c for c in columns if c not in df.columns]
        ref_df = self.load_dataset(ref_dataset)
        missing_right = [c for c in ref_columns if c not in ref_df.columns]

        if missing_left or missing_right:
            return self.result(dataset, dimension, rule_id, f"Periodo reconocido en catálogo temporal: {columns}", ",".join(columns), "temporal_catalog_exists", f"{ref_dataset}.{ref_columns}", 1, 1, severity, source, f"No existen columnas requeridas. Origen: {missing_left}; referencia: {missing_right}")

        left_missing = pd.Series(False, index=df.index)
        left_key_parts = []
        for col in columns:
            left_missing = left_missing | self.missing_mask(df[col])
            left_key_parts.append(self.clean_series(df[col]).str.replace(r"\.0$", "", regex=True))

        left_key = left_key_parts[0]
        for part in left_key_parts[1:]:
            left_key = left_key + "|" + part

        ref_missing = pd.Series(False, index=ref_df.index)
        ref_key_parts = []
        for col in ref_columns:
            ref_missing = ref_missing | self.missing_mask(ref_df[col])
            ref_key_parts.append(self.clean_series(ref_df[col]).str.replace(r"\.0$", "", regex=True))

        ref_key = ref_key_parts[0]
        for part in ref_key_parts[1:]:
            ref_key = ref_key + "|" + part

        ref_values = set(ref_key[~ref_missing].unique())
        evaluable = ~left_missing
        failed_mask = evaluable & ~left_key.isin(ref_values)

        return self.result(dataset, dimension, rule_id, f"Año/periodo reconocido en catálogo temporal SISMEPRE", ",".join(columns), "temporal_catalog_exists", f"{ref_dataset}.{','.join(ref_columns)}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Combinaciones {columns} que no existen en {ref_dataset}.{ref_columns}.", df, failed_mask, columns)

    # ------------------------------------------------------------
    # Reglas de unicidad e integridad
    # ------------------------------------------------------------
    def duplicate_rows(self, dataset: str, df: pd.DataFrame, meta: dict[str, Any]) -> RuleResult:
        """Unicidad: filas completamente duplicadas."""
        failed_mask = df.duplicated()
        return self.result(dataset, meta.get("dimension", "Unicidad"), f"{dataset}.full_row_duplicates", "Filas completamente duplicadas", "*", "duplicate_full_rows", "sin duplicados completos", len(df), int(failed_mask.sum()), meta.get("severity", "Media"), meta.get("source", "profiling_based"), "Filas idénticas repetidas en el dataset.", df, failed_mask, [])

    def unique_key(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """Unicidad: duplicados según clave candidata inferida."""
        columns = rule.get("columns", [])
        missing = [c for c in columns if c not in df.columns]
        rule_id = f"{dataset}.unique_key.{'+'.join(columns)}"

        if missing:
            return self.result(dataset, rule.get("dimension", "Unicidad"), rule_id, f"Clave candidata única: {columns}", ",".join(columns), "unique_key", columns, 1, 1, rule.get("severity", "Media"), rule.get("source", "inferred_business_key"), f"No existen columnas de clave: {missing}.")

        failed_mask = df.duplicated(subset=columns)
        return self.result(dataset, rule.get("dimension", "Unicidad"), rule_id, f"Clave candidata única: {columns}", ",".join(columns), "unique_key", columns, len(df), int(failed_mask.sum()), rule.get("severity", "Media"), rule.get("source", "inferred_business_key"), "Registros duplicados según clave candidata inferida.", df, failed_mask, columns)

    def referential_integrity(self) -> list[RuleResult]:
        """Integridad: valores de un dataset deben existir en otro dataset interno."""
        results: list[RuleResult] = []

        for rule in self.referential_rules:
            dataset = rule["dataset"]
            col = rule["column"]
            ref_dataset = rule["reference_dataset"]
            ref_col = rule["reference_column"]
            dimension = rule.get("dimension", "Integridad")
            severity = rule.get("severity", "Media")
            source = rule.get("source", "referential_rule")
            rule_id = f"{dataset}.{col}.ref_integrity"
            df = self.load_dataset(dataset)
            ref_df = self.load_dataset(ref_dataset)

            if col not in df.columns or ref_col not in ref_df.columns:
                results.append(self.result(dataset, dimension, rule_id, f"Integridad referencial {dataset}.{col} → {ref_dataset}.{ref_col}", col, "referential_integrity", f"{ref_dataset}.{ref_col}", 1, 1, severity, source, "No existen columnas requeridas."))
                continue

            left_values = self.clean_series(df[col])
            left_missing = self.missing_mask(df[col])
            ref_values = set(self.clean_series(ref_df[ref_col])[~self.missing_mask(ref_df[ref_col])].unique())
            evaluable = ~left_missing
            failed_mask = evaluable & ~left_values.isin(ref_values)

            results.append(self.result(dataset, dimension, rule_id, f"Integridad referencial {dataset}.{col} → {ref_dataset}.{ref_col}", col, "referential_integrity", f"{ref_dataset}.{ref_col}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Valores de {dataset}.{col} que no existen en {ref_dataset}.{ref_col}.", df, failed_mask, [col]))

        return results

    # ------------------------------------------------------------
    # Orquestación de reglas por dataset
    # ------------------------------------------------------------
    def pattern_rules(self, dataset: str, df: pd.DataFrame) -> list[RuleResult]:
        """Aplica reglas globales derivadas de patrones de nombres de columnas."""
        results: list[RuleResult] = []
        patterns = self.global_rules.get("pattern_rules", {})

        for rule_name, meta in patterns.items():
            if not meta.get("enabled", False):
                continue

            cols = set()
            for prefix in meta.get("startswith", []):
                cols.update([c for c in df.columns if c.startswith(prefix)])
            for token in meta.get("contains", []):
                cols.update([c for c in df.columns if token in c])
            for c in meta.get("exact_columns", []):
                if c in df.columns:
                    cols.add(c)

            skip_contains = meta.get("skip_contains", [])
            cols = {c for c in cols if not any(token in c for token in skip_contains)}

            for c in sorted(cols):
                if rule_name == "year_4_digits":
                    results.append(self.year_rule(dataset, df, c, meta))
                elif rule_name == "date_parseable":
                    results.append(self.date_rule(dataset, df, c, meta))
                elif rule_name == "monetary_non_negative":
                    results.append(self.non_negative(dataset, df, c, meta, integer=False))
                elif rule_name == "count_non_negative_integer":
                    results.append(self.non_negative(dataset, df, c, meta, integer=True))
                elif "regex" in meta:
                    results.append(self.regex_rule(dataset, df, c, meta["regex"], meta, rule_name))

        return results

    def timeliness_rules(self, dataset: str, df: pd.DataFrame, rules: list[dict[str, Any]]) -> list[RuleResult]:
        """Aplica reglas de oportunidad temporal declaradas por dataset."""
        results: list[RuleResult] = []
        for rule in rules:
            rule_type = rule.get("rule_type")

            if rule_type == "year_not_future":
                for c in rule.get("columns", []):
                    results.append(self.year_not_future(dataset, df, c, rule))
            elif rule_type == "month_period_between":
                for c in rule.get("columns", []):
                    results.append(self.month_period_between(dataset, df, c, rule))
            elif rule_type == "date_not_future":
                for c in rule.get("columns", []):
                    results.append(self.date_not_future(dataset, df, c, rule))
            elif rule_type == "temporal_catalog_exists":
                results.append(self.temporal_catalog_exists(dataset, df, rule))

        return results

    def evaluate_dataset(self, dataset: str) -> list[RuleResult]:
        """Ejecuta todas las reglas internas configuradas para un dataset."""
        df = self.load_dataset(dataset)
        rules = self.datasets_config.get(dataset, {})
        results: list[RuleResult] = []

        logger.info("Evaluando calidad del dataset: %s", dataset)

        dup_meta = self.global_rules.get("duplicate_full_rows", {})
        if dup_meta.get("enabled", True):
            results.append(self.duplicate_rows(dataset, df, dup_meta))

        req_meta = rules.get("required_columns", {})
        for c in req_meta.get("columns", []):
            results.append(self.required_column(dataset, df, c, req_meta))

        obs_meta = rules.get("observed_empty_columns", {})
        for c in obs_meta.get("columns", []):
            results.append(self.observed_empty(dataset, df, c, obs_meta))

        for c, meta in rules.get("allowed_values", {}).items():
            results.append(self.allowed_values(dataset, df, c, meta))

        num_meta = rules.get("numeric_columns", {})
        for c in num_meta.get("columns", []):
            results.append(self.numeric_column(dataset, df, c, num_meta))

        date_meta = rules.get("date_columns", {})
        for c in date_meta.get("columns", []):
            results.append(self.date_rule(dataset, df, c, date_meta))

        for key_rule in rules.get("unique_keys", []):
            results.append(self.unique_key(dataset, df, key_rule))

        for cons_rule in rules.get("consistency_rules", []):
            if cons_rule.get("rule_type") == "less_or_equal":
                results.append(self.less_or_equal(dataset, df, cons_rule))

        for conditional_rule in rules.get("conditional_date_rules", []):
            results.append(self.conditional_date_rule(dataset, df, conditional_rule))

        results.extend(self.timeliness_rules(dataset, df, rules.get("timeliness_rules", [])))
        results.extend(self.pattern_rules(dataset, df))

        logger.info("Calidad evaluada para %s | reglas=%s", dataset, len(results))
        return results

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta la calidad de todos los datasets y devuelve:
        - detalle de reglas
        - muestras de filas observadas
        """
        all_results: list[RuleResult] = []
        self.failed_samples = []

        for dataset in self.datasets_config.keys():
            all_results.extend(self.evaluate_dataset(dataset))

        all_results.extend(self.referential_integrity())

        detail_df = pd.DataFrame([r.__dict__ for r in all_results])
        failed_samples_df = pd.DataFrame(self.failed_samples)
        return detail_df, failed_samples_df
