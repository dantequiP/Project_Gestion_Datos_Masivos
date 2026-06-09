"""
siaf_quality_checker.py

Motor de evaluación de calidad para SIAF Ingresos.

Evalúa los 8 criterios de calidad definidos en quality_rules_siaf.yaml:
1. Completitud   — campos obligatorios no deben ser nulos
2. Validez       — valores dentro del dominio permitido
3. Exactitud     — datos corresponden a la realidad conocida
4. Consistencia  — coherencia interna entre columnas
5. Unicidad      — no debe haber registros duplicados
6. Integridad    — relaciones jerárquicas del clasificador de ingreso
7. Oportunidad   — temporalidad coherente
8. Conformidad   — formatos y longitudes según diccionario

Este módulo NO modifica datos Bronze.
Solo evalúa, reporta y genera auditoría.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def safe_strip(value: Any) -> str:
    """Convierte valores nulos o mixtos a texto seguro para evaluación."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_quality_rules(config_path: Path = Path("app/config/quality_rules_siaf.yaml")) -> dict[str, Any]:
    """
    Carga el archivo YAML con reglas de calidad de SIAF.
    El YAML es la única fuente de verdad: ninguna regla se hardcodea en el código.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de reglas: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("El archivo YAML de reglas no tiene una estructura válida.")
    for section in ["settings", "datasets", "global_rules"]:
        if section not in config:
            raise ValueError(f"Falta la sección obligatoria: {section}")
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Motor de evaluación
# ──────────────────────────────────────────────────────────────────────────────

class SiafQualityChecker:
    """
    Motor de evaluación de calidad para SIAF.

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
        self.bronze_root = Path(self.settings.get("bronze_root", "data/bronze/siaf"))
        self.datasets_config = config.get("datasets", {})
        self.global_rules = config.get("global_rules", {})
        self.current_year = datetime.now().year
        self.current_timestamp = pd.Timestamp.now()
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.failed_samples: list[dict[str, Any]] = []

    # ── Carga de datos ────────────────────────────────────────────────────────

    def load_dataset(self, dataset: str) -> pd.DataFrame:
        """
        Carga el Parquet Bronze de un dataset SIAF.

        SIAF puede tener múltiples subcarpetas (2024_ingreso, 2025_ingreso_mensual, etc.).
        Si el dataset pedido es 'ingresos', se combinan todos los Parquet disponibles.
        """
        if dataset in self.dataframes:
            return self.dataframes[dataset]

        siaf_root = self.bronze_root

        if dataset == "ingresos":
            # Combina todos los Parquets disponibles en data/bronze/siaf/
            frames = []
            if siaf_root.exists():
                for subdir in sorted(siaf_root.iterdir()):
                    if not subdir.is_dir():
                        continue
                    candidates = [
                        subdir / f"{subdir.name}.parquet",
                        subdir / f"{subdir.name}_raw.parquet",
                    ] + sorted(subdir.glob("*.parquet"))
                    for candidate in candidates:
                        if candidate.exists() and candidate.is_file():
                            try:
                                df_part = pd.read_parquet(candidate)
                                frames.append(df_part)
                                logger.info("  Cargado: %s (%d filas)", candidate, len(df_part))
                            except Exception as exc:
                                logger.warning("  No se pudo leer %s: %s", candidate, exc)
                            break

            if not frames:
                raise FileNotFoundError(
                    f"No se encontraron Parquet Bronze bajo {siaf_root}"
                )

            df = pd.concat(frames, ignore_index=True)
            logger.info("Dataset 'ingresos' combinado: %d filas totales de %d archivos.", len(df), len(frames))
        else:
            # Dataset específico por nombre
            candidates = [
                siaf_root / dataset / f"{dataset}.parquet",
                siaf_root / dataset / f"{dataset}_raw.parquet",
            ] + sorted((siaf_root / dataset).glob("*.parquet") if (siaf_root / dataset).exists() else [])
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                raise FileNotFoundError(f"No existe Parquet Bronze para dataset '{dataset}' bajo {siaf_root}")
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

    # ── Evidencia ─────────────────────────────────────────────────────────────

    def _sample_indices(self, failed_mask: pd.Series | None) -> list[int | str]:
        if failed_mask is None or len(failed_mask) == 0:
            return []
        idx = failed_mask[failed_mask].index[:self.max_samples]
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

    # ── Reglas base ───────────────────────────────────────────────────────────

    def required_column(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Completitud: columna obligatoria no vacía."""
        dimension = meta.get("dimension", "Completitud")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.required"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Columna requerida presente: {column}", column, "required_not_empty", "no nulo/no vacío", 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        failed_mask = self.missing_mask(df[column])
        return self.result(dataset, dimension, rule_id, f"Columna requerida no vacía: {column}", column, "required_not_empty", "no nulo/no vacío", len(df), int(failed_mask.sum()), severity, source, f"Registros vacíos o nulos en {column}.", df, failed_mask, [column])

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
        return self.result(dataset, dimension, rule_id, f"Valores permitidos en {column}", column, "allowed_values", values, int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Valores fuera del catálogo {values}. Vacíos evaluados en Completitud.", df, failed_mask, [column])

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
        return self.result(dataset, dimension, rule_id, f"Columna convertible a número: {column}", column, "numeric", "numérico", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores no convertibles a número.", df, failed_mask, [column])

    def non_negative(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Validez/Conformidad: columna no negativa."""
        dimension = meta.get("dimension", "Validez")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{column}.non_negative"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Valor no negativo: {column}", column, "non_negative", ">=0", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_sub = numeric.isna() | (numeric < 0)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub
        return self.result(dataset, dimension, rule_id, f"Valor no negativo: {column}", column, "non_negative", ">=0", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores negativos o no numéricos en campo cuantitativo.", df, failed_mask, [column])

    def regex_rule(self, dataset: str, df: pd.DataFrame, column: str, regex: str, meta: dict[str, Any], suffix: str) -> RuleResult:
        """Conformidad: formato esperado por expresión regular."""
        dimension = meta.get("dimension", "Conformidad")
        severity = meta.get("severity", "Media")
        source = meta.get("source", "profiling_based")
        rule_id = f"{dataset}.{column}.{suffix}"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Formato esperado en {column}", column, "regex", regex, 1, 1, severity, source, "La columna no existe en el Parquet Bronze.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        failed_mask = evaluable & ~clean.str.match(regex, na=False)
        return self.result(dataset, dimension, rule_id, f"Formato esperado en {column}", column, "regex", regex, int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Valores que no cumplen el patrón {regex}.", df, failed_mask, [column])

    def year_rule(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Exactitud: año con formato de cuatro dígitos dentro de rango válido."""
        min_year = int(meta.get("min_year", 2000))
        max_year = int(meta.get("max_year", 2100))
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
        return self.result(dataset, dimension, rule_id, f"Año con formato de 4 dígitos: {column}", column, "year_4_digits", f"{min_year}-{max_year}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, "Valores que no tienen 4 dígitos o están fuera de rango.", df, failed_mask, [column])

    # ── Reglas de consistencia específicas de SIAF ────────────────────────────

    def conditional_value_rule(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """
        Consistencia: si columna_condicion == valor, entonces columna_esperada == valor_esperado.

        Se usa para validar la coherencia entre NIVEL_GOBIERNO (código) y
        NIVEL_GOBIERNO_NOMBRE (descripción).
        """
        condition_col = rule.get("condition_column", "")
        condition_value = str(rule.get("condition_value", ""))
        expected_col = rule.get("expected_column", "")
        expected_value = str(rule.get("expected_value", ""))
        dimension = rule.get("dimension", "Consistencia")
        severity = rule.get("severity", "Alta")
        source = rule.get("source", "dictionary_csv")
        rule_id = f"{dataset}.{condition_col}_{condition_value}.{expected_col}.conditional_value"

        if condition_col not in df.columns or expected_col not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Coherencia {condition_col}={condition_value} → {expected_col}", f"{condition_col},{expected_col}", "conditional_value", f"{expected_col}=={expected_value}", 1, 1, severity, source, "No existen columnas requeridas para evaluar la regla.")

        cond_series = self.clean_series(df[condition_col])
        exp_series = self.clean_series(df[expected_col])
        mask = cond_series == condition_value
        failed_sub = mask & (exp_series != expected_value)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub.loc[failed_sub.index]

        return self.result(dataset, dimension, rule_id, f"Coherencia: {condition_col}={condition_value} → {expected_col}={expected_value}", f"{condition_col},{expected_col}", "conditional_value", f"{expected_col}=={expected_value}", int(mask.sum()), int(failed_mask.sum()), severity, source, f"Registros donde {condition_col}={condition_value} pero {expected_col} != '{expected_value}'.", df, failed_mask, [condition_col, expected_col])

    def column_comparison_rule(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """
        Exactitud: comparación aritmética entre dos columnas (ej: PIM >= PIA).
        """
        left = rule.get("left_column", "")
        operator = rule.get("operator", ">=")
        right = rule.get("right_column", "")
        dimension = rule.get("dimension", "Exactitud")
        severity = rule.get("severity", "Media")
        source = rule.get("source", "business_rule")
        description = rule.get("description", "")
        rule_id = f"{dataset}.{left}_{operator}_{right}"

        if left not in df.columns or right not in df.columns:
            return self.result(dataset, dimension, rule_id, f"{left} {operator} {right}", f"{left},{right}", "column_comparison", f"{left}{operator}{right}", 1, 1, severity, source, "No existen columnas requeridas.")

        left_clean = self.clean_series(df[left])
        right_clean = self.clean_series(df[right])
        evaluable = ~(self.missing_mask(df[left]) | self.missing_mask(df[right]))
        left_num = pd.to_numeric(left_clean[evaluable], errors="coerce")
        right_num = pd.to_numeric(right_clean[evaluable], errors="coerce")

        if operator == ">=":
            failed_sub = left_num.isna() | right_num.isna() | (left_num < right_num)
        elif operator == ">":
            failed_sub = left_num.isna() | right_num.isna() | (left_num <= right_num)
        elif operator == "<=":
            failed_sub = left_num.isna() | right_num.isna() | (left_num > right_num)
        else:
            failed_sub = pd.Series(False, index=left_num.index)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub
        return self.result(dataset, dimension, rule_id, f"Consistencia: {left} {operator} {right}", f"{left},{right}", "column_comparison", f"{left}{operator}{right}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, description or f"Registros donde {left} no cumple {operator} {right}.", df, failed_mask, [left, right])

    def parent_not_zero_if_child_nonzero(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """
        Integridad: si el nivel hijo del clasificador es != 0, el padre también debe ser != 0.

        Aplica a la jerarquía GENERICA → SUBGENERICA → SUBGENERICA_DET → ESPECIFICA → ESPECIFICA_DET.
        """
        parent_col = rule.get("parent_column", "")
        child_col = rule.get("child_column", "")
        dimension = rule.get("dimension", "Integridad")
        severity = rule.get("severity", "Media")
        source = rule.get("source", "business_rule")
        description = rule.get("description", "")
        rule_id = f"{dataset}.{parent_col}_parent_of_{child_col}"

        if parent_col not in df.columns or child_col not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Integridad jerárquica {parent_col} → {child_col}", f"{parent_col},{child_col}", "parent_not_zero_if_child_nonzero", "padre != 0 cuando hijo != 0", 1, 1, severity, source, "No existen columnas requeridas.")

        parent_clean = self.clean_series(df[parent_col])
        child_clean = self.clean_series(df[child_col])
        evaluable = ~(self.missing_mask(df[parent_col]) | self.missing_mask(df[child_col]))

        parent_num = pd.to_numeric(parent_clean[evaluable], errors="coerce").fillna(0)
        child_num = pd.to_numeric(child_clean[evaluable], errors="coerce").fillna(0)

        # Fallo: hijo != 0 pero padre == 0
        failed_sub = (child_num != 0) & (parent_num == 0)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub
        return self.result(dataset, dimension, rule_id, f"Integridad jerárquica: {parent_col} padre de {child_col}", f"{parent_col},{child_col}", "parent_not_zero_if_child_nonzero", "padre!=0 cuando hijo!=0", int(evaluable.sum()), int(failed_mask.sum()), severity, source, description or f"Registros donde {child_col} != 0 pero {parent_col} == 0.", df, failed_mask, [parent_col, child_col])

    # ── Reglas de oportunidad ─────────────────────────────────────────────────

    def year_not_future(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Oportunidad: el año no debe ser posterior al año de ejecución."""
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

    def month_between(self, dataset: str, df: pd.DataFrame, column: str, meta: dict[str, Any]) -> RuleResult:
        """Oportunidad: mes calendario válido (1-12 para SIAF)."""
        dimension = meta.get("dimension", "Oportunidad")
        severity = meta.get("severity", "Alta")
        source = meta.get("source", "dictionary_csv")
        min_value = int(meta.get("min_value", 1))
        max_value = int(meta.get("max_value", 12))
        rule_id = f"{dataset}.{column}.month_between"

        if column not in df.columns:
            return self.result(dataset, dimension, rule_id, f"Mes válido: {column}", column, "month_between", f"{min_value}-{max_value}", 1, 1, severity, source, "La columna no existe.")

        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_sub = numeric.isna() | (numeric < min_value) | (numeric > max_value) | (numeric % 1 != 0)

        failed_mask = pd.Series(False, index=df.index)
        failed_mask.loc[failed_sub.index] = failed_sub
        return self.result(dataset, dimension, rule_id, f"Mes calendario válido: {column}", column, "month_between", f"{min_value}-{max_value}", int(evaluable.sum()), int(failed_mask.sum()), severity, source, f"Valores fuera del rango {min_value}-{max_value}. SIAF no usa periodos especiales fuera de este rango.", df, failed_mask, [column])

    # ── Reglas de unicidad ────────────────────────────────────────────────────

    def duplicate_rows(self, dataset: str, df: pd.DataFrame, meta: dict[str, Any]) -> RuleResult:
        """Unicidad: filas completamente duplicadas."""
        failed_mask = df.duplicated()
        return self.result(dataset, meta.get("dimension", "Unicidad"), f"{dataset}.full_row_duplicates", "Filas completamente duplicadas", "*", "duplicate_full_rows", "sin duplicados completos", len(df), int(failed_mask.sum()), meta.get("severity", "Media"), meta.get("source", "profiling_based"), "Filas idénticas repetidas en el dataset.", df, failed_mask, [])

    def unique_key(self, dataset: str, df: pd.DataFrame, rule: dict[str, Any]) -> RuleResult:
        """Unicidad: duplicados según clave candidata inferida."""
        columns = rule.get("columns", [])
        missing = [c for c in columns if c not in df.columns]
        rule_id = f"{dataset}.unique_key.{'+'.join(columns[:4])}"

        if missing:
            return self.result(dataset, rule.get("dimension", "Unicidad"), rule_id, f"Clave candidata única: {columns}", ",".join(columns), "unique_key", columns, 1, 1, rule.get("severity", "Media"), rule.get("source", "inferred_business_key"), f"No existen columnas de clave: {missing}.")

        failed_mask = df.duplicated(subset=columns)
        return self.result(dataset, rule.get("dimension", "Unicidad"), rule_id, f"Clave candidata única (clasificador completo + período + ejecutora)", ",".join(columns), "unique_key", columns, len(df), int(failed_mask.sum()), rule.get("severity", "Media"), rule.get("source", "inferred_business_key"), "Registros duplicados según clave candidata inferida del clasificador SIAF.", df, failed_mask, columns[:4])

    # ── Reglas globales por patrón ────────────────────────────────────────────

    def pattern_rules(self, dataset: str, df: pd.DataFrame) -> list[RuleResult]:
        """Aplica reglas globales derivadas de patrones de nombres de columnas."""
        results: list[RuleResult] = []
        patterns = self.global_rules.get("pattern_rules", {})

        for rule_name, meta in patterns.items():
            if not meta.get("enabled", False):
                continue

            cols: set[str] = set()
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
                elif rule_name == "monetary_non_negative":
                    results.append(self.non_negative(dataset, df, c, {
                        **meta,
                        "dimension": "Conformidad",
                    }))
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
            elif rule_type == "month_between":
                for c in rule.get("columns", []):
                    results.append(self.month_between(dataset, df, c, rule))
        return results

    def consistency_rules_dispatch(self, dataset: str, df: pd.DataFrame, rules: dict[str, Any]) -> list[RuleResult]:
        """Despacha todas las reglas de consistencia/conformidad/exactitud del dataset."""
        results: list[RuleResult] = []
        for rule_name, rule in rules.items():
            if not rule.get("enabled", True):
                continue
            rule_type = rule.get("rule_type", "")

            if rule_type == "conditional_value":
                results.append(self.conditional_value_rule(dataset, df, rule))
            elif rule_type == "column_comparison":
                results.append(self.column_comparison_rule(dataset, df, rule))
            elif rule_type == "regex":
                column = rule.get("column", "")
                regex = rule.get("regex", "")
                if column and regex:
                    results.append(self.regex_rule(dataset, df, column, regex, rule, rule_name))
            elif rule_type == "parent_not_zero_if_child_nonzero":
                results.append(self.parent_not_zero_if_child_nonzero(dataset, df, rule))

        return results

    def referential_rules_dispatch(self, dataset: str, df: pd.DataFrame, rules: dict[str, Any]) -> list[RuleResult]:
        """Despacha reglas de integridad referencial interna del clasificador SIAF."""
        results: list[RuleResult] = []
        for rule_name, rule in rules.items():
            if not rule.get("enabled", True):
                continue
            rule_type = rule.get("rule_type", "")
            if rule_type == "parent_not_zero_if_child_nonzero":
                results.append(self.parent_not_zero_if_child_nonzero(dataset, df, rule))
        return results

    # ── Orquestación ──────────────────────────────────────────────────────────

    def evaluate_dataset(self, dataset: str) -> list[RuleResult]:
        """Ejecuta todas las reglas configuradas para el dataset."""
        df = self.load_dataset(dataset)
        rules = self.datasets_config.get(dataset, {})
        results: list[RuleResult] = []

        logger.info("Evaluando calidad del dataset SIAF: %s (%d filas)", dataset, len(df))

        # Unicidad: filas duplicadas completas
        dup_meta = self.global_rules.get("duplicate_full_rows", {})
        if dup_meta.get("enabled", True):
            results.append(self.duplicate_rows(dataset, df, dup_meta))

        # Completitud: columnas requeridas
        req_meta = rules.get("required_columns", {})
        for c in req_meta.get("columns", []):
            results.append(self.required_column(dataset, df, c, req_meta))

        # Validez: valores permitidos
        for c, meta in rules.get("allowed_values", {}).items():
            results.append(self.allowed_values(dataset, df, c, meta))

        # Validez: columnas numéricas
        num_meta = rules.get("numeric_columns", {})
        for c in num_meta.get("columns", []):
            results.append(self.numeric_column(dataset, df, c, num_meta))

        # Validez: montos no negativos (explícito en non_negative_columns)
        neg_meta = rules.get("non_negative_columns", {})
        for c in neg_meta.get("columns", []):
            results.append(self.non_negative(dataset, df, c, neg_meta))

        # Unicidad: clave de negocio
        for key_rule in rules.get("unique_keys", []):
            results.append(self.unique_key(dataset, df, key_rule))

        # Consistencia + Conformidad + Exactitud (consistency_rules en YAML)
        cons_rules = rules.get("consistency_rules", {})
        if isinstance(cons_rules, dict):
            results.extend(self.consistency_rules_dispatch(dataset, df, cons_rules))

        # Integridad referencial jerárquica
        ref_rules = rules.get("referential_rules", {})
        if isinstance(ref_rules, dict):
            results.extend(self.referential_rules_dispatch(dataset, df, ref_rules))

        # Oportunidad temporal
        results.extend(self.timeliness_rules(dataset, df, rules.get("timeliness_rules", [])))

        # Patrones globales (year_4_digits, monetary_non_negative)
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

        detail_df = pd.DataFrame([r.__dict__ for r in all_results])
        failed_samples_df = pd.DataFrame(self.failed_samples)
        return detail_df, failed_samples_df