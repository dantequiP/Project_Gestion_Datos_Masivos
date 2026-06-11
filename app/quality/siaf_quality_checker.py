"""
siaf_quality_checker.py

Motor de evaluación de calidad de datos para SIAF Ingresos.

Diferencias clave con el QualityChecker de SISMEPRE:
- Trabaja con PySpark (10,777,068 filas en Bronze histórico).
- Las reglas se evalúan como expresiones Spark (DataFrame API).
- Los failed_samples se obtienen con .limit() para no traer millones de filas
  al driver.
- El checker NO usa pandas para evaluaciones masivas — solo para los resultados
  y muestras (tablas pequeñas).

Principio heredado: este checker NO modifica datos Bronze. Solo observa y reporta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


# ── Resultado estándar (idéntico a SISMEPRE para interoperabilidad) ──────────

@dataclass
class RuleResult:
    """
    Resultado estándar de una regla de calidad SIAF.

    Idéntico al de SISMEPRE para que los reportes HTML y el quality_report
    sean compartidos sin modificaciones.
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


# ── Checker principal ─────────────────────────────────────────────────────────

class SiafQualityChecker:
    """
    Motor de calidad para SIAF Ingresos sobre PySpark.

    Parameters
    ----------
    spark : SparkSession
        Sesión Spark activa.
    df : DataFrame
        DataFrame PySpark Bronze (sin _diario, mergeSchema=True).
    config : dict
        Contenido de quality_rules_siaf.yaml.
    """

    MISSING_LIKE = {"", " ", "nan", "NaN", "None", "none", "NULL", "null", "<NA>", "NaT"}

    def __init__(self, spark: SparkSession, df: DataFrame, config: dict[str, Any]) -> None:
        self.spark = spark
        self.df = df
        self.config = config
        self.settings = config["settings"]
        self.max_samples: int = int(self.settings.get("max_failed_samples_per_rule", 10))
        self.missing_tokens: set[str] = set(config.get("missing_tokens", [])) | self.MISSING_LIKE
        self.current_year: int = datetime.now().year
        self.failed_samples: list[dict] = []
        self._total_rows: int | None = None

    @property
    def total_rows(self) -> int:
        if self._total_rows is None:
            self._total_rows = int(self.df.count())
        return self._total_rows

    # ── Utilidades internas ──────────────────────────────────────────────────

    def _col_exists(self, col: str) -> bool:
        return col in self.df.columns

    def _is_missing_expr(self, col: str) -> Any:
        """Expresión Spark que detecta nulo o token de ausencia."""
        return F.col(col).isNull() | F.trim(F.col(col).cast("string")).isin(list(self.missing_tokens))

    def _count_failed(self, condition_expr) -> int:
        """Cuenta filas que NO cumplen la condición dada."""
        return int(self.df.filter(condition_expr).count())

    def _count_evaluable(self, col: str) -> int:
        """Filas no-nulas/no-vacías para una columna."""
        return int(self.df.filter(~self._is_missing_expr(col)).count())

    def _get_samples(self, filter_expr, cols: list[str]) -> str:
        """
        Obtiene muestras de filas que fallan una regla.
        Usa .limit() para no materializar millones de filas.
        """
        available = [c for c in cols if c in self.df.columns]
        if not available:
            return "[]"
        try:
            rows = (
                self.df
                .filter(filter_expr)
                .select(*available)
                .limit(self.max_samples)
                .toPandas()
            )
            if rows.empty:
                return "[]"
            samples = rows.fillna("").astype(str).to_dict(orient="records")
            return json.dumps(samples, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo obtener muestra de fallos: %s", exc)
            return "[]"

    def _result(
        self,
        *,
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
        sample_expr=None,
        sample_cols: list[str] | None = None,
    ) -> RuleResult:
        evaluated_rows = int(evaluated_rows)
        failed_rows = int(failed_rows)
        passed_rows = max(evaluated_rows - failed_rows, 0)

        if evaluated_rows == 0:
            score = None
            status = "No evaluable"
        else:
            score = round((passed_rows / evaluated_rows) * 100, 4)
            status = "Sin observaciones" if failed_rows == 0 else "Con observaciones"

        sample_failed_rows = "[]"
        if sample_expr is not None and failed_rows > 0 and sample_cols:
            sample_failed_rows = self._get_samples(sample_expr, sample_cols)

        return RuleResult(
            dataset="siaf_ingresos",
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

    # ── DIMENSIÓN 1: COMPLETITUD ─────────────────────────────────────────────

    def check_not_null(self, col: str, severity: str, source: str) -> RuleResult:
        """Columna completamente no nula (completitud 100% esperada)."""
        if not self._col_exists(col):
            return self._result(
                dimension="Completitud", rule_id=f"siaf.{col}.not_null",
                rule_name=f"No nulo: {col}", column_name=col,
                rule_type="not_null", expected_value="no nulo",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe en el Bronze.",
            )
        failed_expr = self._is_missing_expr(col)
        failed_rows = self._count_failed(failed_expr)
        return self._result(
            dimension="Completitud", rule_id=f"siaf.{col}.not_null",
            rule_name=f"No nulo: {col}", column_name=col,
            rule_type="not_null", expected_value="no nulo",
            evaluated_rows=self.total_rows, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Registros nulos o vacíos en {col}.",
            sample_expr=failed_expr, sample_cols=[col],
        )

    def check_null_threshold(self, col: str, threshold_pct: float, severity: str, source: str) -> RuleResult:
        """Columna casi completa — alerta si supera threshold_pct% de nulos."""
        if not self._col_exists(col):
            return self._result(
                dimension="Completitud", rule_id=f"siaf.{col}.null_threshold",
                rule_name=f"Nulos bajo umbral: {col}", column_name=col,
                rule_type="null_threshold", expected_value=f"< {threshold_pct}% nulos",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe en el Bronze.",
            )
        failed_expr = self._is_missing_expr(col)
        failed_rows = self._count_failed(failed_expr)
        actual_pct = (failed_rows / self.total_rows * 100) if self.total_rows > 0 else 0
        # Tratamos como "observación" si supera el umbral
        obs_rows = failed_rows if actual_pct > threshold_pct else 0
        return self._result(
            dimension="Completitud", rule_id=f"siaf.{col}.null_threshold",
            rule_name=f"Nulos bajo umbral ({threshold_pct}%): {col}", column_name=col,
            rule_type="null_threshold", expected_value=f"< {threshold_pct}% nulos",
            evaluated_rows=self.total_rows, failed_rows=obs_rows,
            severity=severity, source=source,
            observation=(
                f"Nulos en {col}: {actual_pct:.4f}% (umbral: {threshold_pct}%). "
                f"{'Superado — revisar.' if obs_rows > 0 else 'Dentro del umbral esperado.'}"
            ),
            sample_expr=failed_expr if obs_rows > 0 else None,
            sample_cols=[col],
        )

    def check_conditional_null_allowed(
        self, col: str, condition_col: str, condition_val: str,
        expected_null_pct_when_match: float, expected_null_pct_when_not_match: float,
        severity: str, source: str,
    ) -> RuleResult:
        """
        Vacíos estructurales: una columna puede ser nula SOLO cuando
        otra columna tiene un valor específico (ej: SECTOR vacío solo cuando M).
        """
        if not self._col_exists(col) or not self._col_exists(condition_col):
            return self._result(
                dimension="Completitud", rule_id=f"siaf.{col}.conditional_null",
                rule_name=f"Nulo condicional {col} | {condition_col}={condition_val}",
                column_name=col, rule_type="conditional_null_allowed",
                expected_value=f"vacío solo cuando {condition_col}={condition_val}",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="No existen columnas requeridas.",
            )

        # Fallos: vacío cuando NO debería (condition_col != condition_val)
        not_match = F.col(condition_col) != F.lit(condition_val)
        is_missing = self._is_missing_expr(col)
        fail_expr = not_match & is_missing
        n_not_match = int(self.df.filter(not_match).count())
        failed_rows = int(self.df.filter(fail_expr).count())

        return self._result(
            dimension="Completitud",
            rule_id=f"siaf.{col}.conditional_null",
            rule_name=f"Vacío estructural válido: {col} (solo cuando {condition_col}={condition_val})",
            column_name=col, rule_type="conditional_null_allowed",
            expected_value=f"vacío solo cuando {condition_col}={condition_val}",
            evaluated_rows=n_not_match, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"{col} vacío cuando {condition_col} ≠ {condition_val}: "
                f"{failed_rows:,} casos. Cuando {condition_col}={condition_val} el vacío es esperado (diseño SIAF)."
            ),
            sample_expr=fail_expr if failed_rows > 0 else None,
            sample_cols=[col, condition_col],
        )

    # ── DIMENSIÓN 2: VALIDEZ ─────────────────────────────────────────────────

    def check_accepted_values(
        self, col: str, accepted: list[str], severity: str, source: str,
    ) -> RuleResult:
        """Valores dentro de catálogo cerrado."""
        if not self._col_exists(col):
            return self._result(
                dimension="Validez", rule_id=f"siaf.{col}.accepted_values",
                rule_name=f"Catálogo válido: {col}", column_name=col,
                rule_type="accepted_values", expected_value=accepted,
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        not_missing = ~self._is_missing_expr(col)
        evaluable = int(self.df.filter(not_missing).count())
        accepted_str = [str(v) for v in accepted]
        fail_expr = not_missing & (~F.col(col).cast("string").isin(accepted_str))
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Validez", rule_id=f"siaf.{col}.accepted_values",
            rule_name=f"Catálogo válido: {col}", column_name=col,
            rule_type="accepted_values", expected_value=accepted_str,
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Valores fuera del catálogo {accepted_str} en {col}.",
            sample_expr=fail_expr, sample_cols=[col],
        )

    def check_numeric_type(self, col: str, severity: str, source: str) -> RuleResult:
        """Columna convertible a número."""
        if not self._col_exists(col):
            return self._result(
                dimension="Validez", rule_id=f"siaf.{col}.numeric",
                rule_name=f"Numérico: {col}", column_name=col,
                rule_type="numeric_type", expected_value="numérico",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        not_missing = ~self._is_missing_expr(col)
        evaluable = int(self.df.filter(not_missing).count())
        fail_expr = not_missing & F.col(col).cast(DoubleType()).isNull()
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Validez", rule_id=f"siaf.{col}.numeric",
            rule_name=f"Tipo numérico: {col}", column_name=col,
            rule_type="numeric_type", expected_value="numérico",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Valores no convertibles a número en {col}.",
            sample_expr=fail_expr, sample_cols=[col],
        )

    # ── DIMENSIÓN 3: EXACTITUD ───────────────────────────────────────────────

    def check_range(
        self, col: str, min_val: float | None, max_val: float | None,
        severity: str, source: str, allow_zero: bool = True,
    ) -> RuleResult:
        """Valores dentro de rango numérico."""
        if not self._col_exists(col):
            return self._result(
                dimension="Exactitud", rule_id=f"siaf.{col}.range",
                rule_name=f"Rango válido: {col}", column_name=col,
                rule_type="accepted_range", expected_value=f"[{min_val}, {max_val}]",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        not_missing = ~self._is_missing_expr(col)
        evaluable = int(self.df.filter(not_missing).count())
        num_col = F.col(col).cast(DoubleType())
        fail_conditions = [not_missing, num_col.isNull()]

        if min_val is not None:
            fail_conditions.append(num_col < F.lit(min_val))
        if max_val is not None:
            fail_conditions.append(num_col > F.lit(max_val))

        # Armamos: evaluable Y (nulo_al_castear OR fuera_de_rango)
        fail_expr = not_missing & (
            num_col.isNull()
            | (F.lit(min_val is not None) & (num_col < F.lit(min_val or 0)))
            | (F.lit(max_val is not None) & (num_col > F.lit(max_val or 0)))
        )

        # Reconstruimos de forma más limpia
        range_fail = num_col.isNull()
        if min_val is not None:
            range_fail = range_fail | (num_col < F.lit(min_val))
        if max_val is not None:
            range_fail = range_fail | (num_col > F.lit(max_val))
        fail_expr = not_missing & range_fail

        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Exactitud", rule_id=f"siaf.{col}.range",
            rule_name=f"Rango numérico: {col}", column_name=col,
            rule_type="accepted_range", expected_value=f"[{min_val}, {max_val}]",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Valores fuera del rango [{min_val}, {max_val}] en {col}.",
            sample_expr=fail_expr, sample_cols=[col],
        )

    def check_regex(self, col: str, pattern: str, severity: str, source: str,
                    skip_when: tuple[str, str] | None = None) -> RuleResult:
        """Formato por expresión regular, con opción de excluir condición."""
        if not self._col_exists(col):
            return self._result(
                dimension="Conformidad", rule_id=f"siaf.{col}.regex",
                rule_name=f"Formato regex: {col}", column_name=col,
                rule_type="regex", expected_value=pattern,
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        not_missing = ~self._is_missing_expr(col)
        if skip_when:
            skip_col, skip_val = skip_when
            if self._col_exists(skip_col):
                not_missing = not_missing & (F.col(skip_col) != F.lit(skip_val))

        evaluable = int(self.df.filter(not_missing).count())
        fail_expr = not_missing & ~F.col(col).cast("string").rlike(pattern)
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Conformidad", rule_id=f"siaf.{col}.regex",
            rule_name=f"Formato regex: {col}", column_name=col,
            rule_type="regex", expected_value=pattern,
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Valores que no cumplen el patrón {pattern} en {col}.",
            sample_expr=fail_expr, sample_cols=[col],
        )

    def check_ano_format(self, col: str, severity: str, source: str) -> RuleResult:
        """ANO_DOC: 4 dígitos exactos."""
        return self.check_regex(col, r"^\d{4}$", severity, source)

    # ── DIMENSIÓN 4: CONSISTENCIA ────────────────────────────────────────────

    def check_conditional_value(
        self, condition_col: str, condition_val: str,
        expected_col: str, expected_val: str,
        severity: str, source: str, rule_key: str,
    ) -> RuleResult:
        """Si condition_col = condition_val entonces expected_col debe = expected_val."""
        if not self._col_exists(condition_col) or not self._col_exists(expected_col):
            return self._result(
                dimension="Consistencia", rule_id=f"siaf.{rule_key}.conditional_value",
                rule_name=f"Consistencia: {condition_col}={condition_val} → {expected_col}={expected_val}",
                column_name=f"{condition_col},{expected_col}",
                rule_type="conditional_value", expected_value=expected_val,
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="Columnas requeridas no existen.",
            )
        match = F.col(condition_col) == F.lit(condition_val)
        n_match = int(self.df.filter(match).count())
        fail_expr = match & (F.col(expected_col) != F.lit(expected_val))
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Consistencia",
            rule_id=f"siaf.{rule_key}.conditional_value",
            rule_name=f"{condition_col}={condition_val} → {expected_col}={expected_val}",
            column_name=f"{condition_col},{expected_col}",
            rule_type="conditional_value", expected_value=expected_val,
            evaluated_rows=n_match, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Inconsistencias entre {condition_col} y {expected_col}.",
            sample_expr=fail_expr, sample_cols=[condition_col, expected_col],
        )

    def check_code_name_consistency(
        self, code_col: str, name_col: str, severity: str, source: str,
    ) -> RuleResult:
        """Un código debe tener exactamente un nombre (relación N:1)."""
        if not self._col_exists(code_col) or not self._col_exists(name_col):
            return self._result(
                dimension="Consistencia",
                rule_id=f"siaf.{code_col}_{name_col}.code_name",
                rule_name=f"Consistencia código-nombre: {code_col} → {name_col}",
                column_name=f"{code_col},{name_col}",
                rule_type="code_name_consistency", expected_value="1 nombre por código",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="Columnas requeridas no existen.",
            )
        not_missing = ~self._is_missing_expr(code_col) & ~self._is_missing_expr(name_col)
        evaluable = int(self.df.filter(not_missing).count())

        # Alias fijo para evitar que Spark renombre la columna a "CAST(X AS STRING)"
        code_alias = "__code__"
        multi = (
            self.df
            .filter(not_missing)
            .groupBy(F.col(code_col).cast("string").alias(code_alias))
            .agg(F.countDistinct(F.col(name_col).cast("string")).alias("n_names"))
            .filter(F.col("n_names") > 1)
        )
        codes_with_multi = multi.count()

        # Filas afectadas = filas cuyo código tiene >1 nombre
        failed_rows = 0
        if codes_with_multi > 0:
            bad_codes = [r[0] for r in multi.select(code_alias).limit(500).collect()]
            fail_expr = not_missing & F.col(code_col).cast("string").isin(bad_codes)
            failed_rows = int(self.df.filter(fail_expr).count())

        return self._result(
            dimension="Consistencia",
            rule_id=f"siaf.{code_col}_{name_col}.code_name",
            rule_name=f"Consistencia código-nombre: {code_col} → {name_col}",
            column_name=f"{code_col},{name_col}",
            rule_type="code_name_consistency", expected_value="1 nombre por código",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"{codes_with_multi} código(s) de {code_col} con múltiples nombres en {name_col}. "
                "Puede indicar renombramientos institucionales sin homologar."
            ),
            sample_expr=None,  # ya tenemos el conteo suficiente
            sample_cols=[code_col, name_col],
        )

    def check_column_comparison(
        self, left_col: str, operator: str, right_col: str,
        filter_condition: str | None, severity: str, source: str, rule_key: str,
    ) -> RuleResult:
        """Comparación aritmética entre dos columnas (ej: PIM >= PIA)."""
        if not self._col_exists(left_col) or not self._col_exists(right_col):
            return self._result(
                dimension="Consistencia", rule_id=f"siaf.{rule_key}.column_comparison",
                rule_name=f"Comparación: {left_col} {operator} {right_col}",
                column_name=f"{left_col},{right_col}",
                rule_type="column_comparison", expected_value=f"{left_col} {operator} {right_col}",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="Columnas requeridas no existen.",
            )
        base_df = self.df
        if filter_condition:
            base_df = base_df.filter(filter_condition)

        not_missing = ~self._is_missing_expr(left_col) & ~self._is_missing_expr(right_col)
        evaluable = int(base_df.filter(not_missing).count())

        left = F.col(left_col).cast(DoubleType())
        right = F.col(right_col).cast(DoubleType())

        if operator == ">=":
            fail_expr = not_missing & (left < right)
        elif operator == "<=":
            fail_expr = not_missing & (left > right)
        elif operator == ">":
            fail_expr = not_missing & (left <= right)
        else:
            fail_expr = not_missing & (left != right)

        failed_rows = int(base_df.filter(fail_expr).count())
        return self._result(
            dimension="Consistencia",
            rule_id=f"siaf.{rule_key}.column_comparison",
            rule_name=f"Comparación: {left_col} {operator} {right_col}",
            column_name=f"{left_col},{right_col}",
            rule_type="column_comparison",
            expected_value=f"{left_col} {operator} {right_col}",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"Registros donde {left_col} no cumple {operator} {right_col}. "
                + (f"Filtro aplicado: {filter_condition}." if filter_condition else "")
            ),
            sample_expr=None,
            sample_cols=[left_col, right_col],
        )

    def check_null_only_when(
        self, col: str, cond_col: str, cond_val: str,
        severity: str, source: str,
    ) -> RuleResult:
        """La columna debe ser nula ÚNICAMENTE cuando cond_col = cond_val."""
        if not self._col_exists(col) or not self._col_exists(cond_col):
            return self._result(
                dimension="Consistencia",
                rule_id=f"siaf.{col}.null_only_when",
                rule_name=f"Nulo solo cuando {cond_col}={cond_val}: {col}",
                column_name=col, rule_type="null_only_when",
                expected_value=f"nulo solo cuando {cond_col}={cond_val}",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="Columnas requeridas no existen.",
            )
        is_missing = self._is_missing_expr(col)
        not_expected = F.col(cond_col) != F.lit(cond_val)
        fail_expr = is_missing & not_expected
        evaluable = int(self.df.filter(not_expected).count())
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Consistencia",
            rule_id=f"siaf.{col}.null_only_when",
            rule_name=f"Nulo solo cuando {cond_col}={cond_val}: {col}",
            column_name=col, rule_type="null_only_when",
            expected_value=f"nulo solo cuando {cond_col}={cond_val}",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"{col} vacío cuando {cond_col} ≠ {cond_val}: {failed_rows:,} casos inesperados."
            ),
            sample_expr=fail_expr, sample_cols=[col, cond_col],
        )

    # ── DIMENSIÓN 5: UNICIDAD ────────────────────────────────────────────────

    def check_duplicate_full_rows(self, severity: str, source: str) -> RuleResult:
        """Filas completamente duplicadas."""
        total = self.total_rows
        distinct = int(self.df.distinct().count())
        failed_rows = total - distinct
        return self._result(
            dimension="Unicidad",
            rule_id="siaf.full_row_duplicates",
            rule_name="Filas completamente duplicadas",
            column_name="*", rule_type="duplicate_full_rows",
            expected_value="0 duplicados exactos",
            evaluated_rows=total, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"{failed_rows:,} filas exactamente duplicadas detectadas.",
            sample_expr=None, sample_cols=[],
        )

    def check_unique_key(self, columns: list[str], severity: str, source: str) -> RuleResult:
        """Clave de negocio compuesta: duplicados según las 13 columnas del clasificador."""
        missing = [c for c in columns if not self._col_exists(c)]
        if missing:
            return self._result(
                dimension="Unicidad",
                rule_id=f"siaf.unique_key.{'_'.join(columns[:3])}...",
                rule_name=f"Clave única ({len(columns)} columnas)",
                column_name=",".join(columns[:5]) + ",...",
                rule_type="unique_key", expected_value="sin duplicados",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columnas faltantes: {missing}.",
            )
        total = self.total_rows
        distinct_keys = int(self.df.select(*columns).distinct().count())
        failed_rows = total - distinct_keys
        return self._result(
            dimension="Unicidad",
            rule_id="siaf.unique_key.13cols",
            rule_name=f"Clave de negocio única ({len(columns)} columnas clasificador completo)",
            column_name=",".join(columns),
            rule_type="unique_key", expected_value="sin duplicados por clave",
            evaluated_rows=total, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"{failed_rows:,} filas duplicadas según clave de {len(columns)} columnas. "
                "Puede indicar solapamiento entre archivos anuales al consolidar el histórico."
            ),
            sample_expr=None, sample_cols=columns[:4],
        )

    # ── DIMENSIÓN 6: INTEGRIDAD ──────────────────────────────────────────────

    def check_parent_not_zero_if_child_nonzero(
        self, parent_col: str, child_col: str, severity: str, source: str,
    ) -> RuleResult:
        """Jerarquía: si el hijo ≠ 0, el padre no puede ser 0."""
        if not self._col_exists(parent_col) or not self._col_exists(child_col):
            return self._result(
                dimension="Integridad",
                rule_id=f"siaf.{parent_col}_{child_col}.hierarchy",
                rule_name=f"Jerarquía: {parent_col} padre de {child_col}",
                column_name=f"{parent_col},{child_col}",
                rule_type="parent_not_zero_if_child_nonzero",
                expected_value=f"{parent_col} ≠ 0 si {child_col} ≠ 0",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="Columnas requeridas no existen.",
            )
        child_nonzero = F.col(child_col).cast(DoubleType()) != F.lit(0.0)
        evaluable = int(self.df.filter(~self._is_missing_expr(child_col) & child_nonzero).count())
        fail_expr = (
            ~self._is_missing_expr(parent_col)
            & ~self._is_missing_expr(child_col)
            & child_nonzero
            & (F.col(parent_col).cast(DoubleType()) == F.lit(0.0))
        )
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Integridad",
            rule_id=f"siaf.{parent_col}_{child_col}.hierarchy",
            rule_name=f"Jerarquía clasificador: {parent_col} padre de {child_col}",
            column_name=f"{parent_col},{child_col}",
            rule_type="parent_not_zero_if_child_nonzero",
            expected_value=f"{parent_col} ≠ 0 cuando {child_col} ≠ 0",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Filas donde {child_col} ≠ 0 pero {parent_col} = 0 (rompería la jerarquía).",
            sample_expr=fail_expr, sample_cols=[parent_col, child_col],
        )

    def check_cardinality(
        self, col: str, expected_cardinality: int, severity: str, source: str,
    ) -> RuleResult:
        """Cardinalidad del catálogo no debe exceder el valor observado en profiling."""
        if not self._col_exists(col):
            return self._result(
                dimension="Integridad", rule_id=f"siaf.{col}.cardinality",
                rule_name=f"Cardinalidad: {col}", column_name=col,
                rule_type="cardinality_check", expected_value=str(expected_cardinality),
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        actual_card = int(
            self.df.filter(~self._is_missing_expr(col))
            .select(F.col(col).cast("string")).distinct().count()
        )
        failed_rows = max(0, actual_card - expected_cardinality)
        return self._result(
            dimension="Integridad", rule_id=f"siaf.{col}.cardinality",
            rule_name=f"Cardinalidad catálogo: {col}",
            column_name=col, rule_type="cardinality_check",
            expected_value=str(expected_cardinality),
            evaluated_rows=actual_card, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"Cardinalidad observada: {actual_card} | esperada: {expected_cardinality}. "
                + ("⚠️ Supera el catálogo del profiling — revisar versión MEF." if failed_rows > 0
                   else "Dentro del rango esperado.")
            ),
        )

    # ── DIMENSIÓN 7: OPORTUNIDAD ─────────────────────────────────────────────

    def check_year_not_future(self, col: str, severity: str, source: str) -> RuleResult:
        """ANO_DOC no puede ser mayor al año en curso."""
        if not self._col_exists(col):
            return self._result(
                dimension="Oportunidad", rule_id=f"siaf.{col}.year_not_future",
                rule_name=f"Año no futuro: {col}", column_name=col,
                rule_type="year_not_future", expected_value=f"<= {self.current_year}",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        not_missing = ~self._is_missing_expr(col)
        evaluable = int(self.df.filter(not_missing).count())
        fail_expr = not_missing & (F.col(col).cast(LongType()) > F.lit(self.current_year))
        failed_rows = int(self.df.filter(fail_expr).count())
        return self._result(
            dimension="Oportunidad", rule_id=f"siaf.{col}.year_not_future",
            rule_name=f"Año no futuro: {col}", column_name=col,
            rule_type="year_not_future", expected_value=f"<= {self.current_year}",
            evaluated_rows=evaluable, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=f"Años posteriores al año de ejecución ({self.current_year}).",
            sample_expr=fail_expr, sample_cols=[col],
        )

    def check_temporal_coverage(
        self, year_col: str, month_col: str,
        expected_months: int, exception_years: list[int],
        severity: str, source: str,
    ) -> RuleResult:
        """Cada año debe tener 12 meses, excepto los años parciales declarados."""
        if not self._col_exists(year_col) or not self._col_exists(month_col):
            return self._result(
                dimension="Oportunidad", rule_id="siaf.temporal_coverage",
                rule_name="Cobertura temporal por año",
                column_name=f"{year_col},{month_col}",
                rule_type="temporal_coverage_check", expected_value=f"{expected_months} meses/año",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation="Columnas requeridas no existen.",
            )
        coverage = (
            self.df
            .filter(~self._is_missing_expr(year_col) & ~self._is_missing_expr(month_col))
            .groupBy(F.col(year_col).cast(LongType()).alias("ano"))
            .agg(F.countDistinct(F.col(month_col).cast(LongType())).alias("n_meses"))
            .orderBy("ano")
        )
        rows = coverage.toPandas()
        incomplete = rows[
            (~rows["ano"].isin(exception_years)) & (rows["n_meses"] < expected_months)
        ]
        failed_rows = int(len(incomplete))
        obs = (
            f"Años con cobertura incompleta (< {expected_months} meses): "
            + (incomplete[["ano", "n_meses"]].to_dict(orient="records").__str__()
               if not incomplete.empty else "ninguno")
            + f". Excepciones permitidas: {exception_years}."
        )
        return self._result(
            dimension="Oportunidad", rule_id="siaf.temporal_coverage",
            rule_name="Cobertura temporal: 12 meses por año",
            column_name=f"{year_col},{month_col}",
            rule_type="temporal_coverage_check", expected_value=f"{expected_months} meses/año",
            evaluated_rows=int(len(rows)), failed_rows=failed_rows,
            severity=severity, source=source,
            observation=obs,
        )

    # ── DIMENSIÓN 8: CONFORMIDAD ─────────────────────────────────────────────

    def check_cardinality_upper_bound(
        self, col: str, max_cardinality: int, severity: str, source: str,
    ) -> RuleResult:
        """Cardinalidad no debe superar el límite superior (catálogo abierto)."""
        if not self._col_exists(col):
            return self._result(
                dimension="Conformidad", rule_id=f"siaf.{col}.cardinality_upper",
                rule_name=f"Cardinalidad máxima: {col}", column_name=col,
                rule_type="cardinality_upper_bound", expected_value=f"<= {max_cardinality}",
                evaluated_rows=1, failed_rows=1,
                severity=severity, source=source,
                observation=f"Columna {col} no existe.",
            )
        actual = int(
            self.df.filter(~self._is_missing_expr(col))
            .select(F.col(col).cast("string")).distinct().count()
        )
        failed_rows = max(0, actual - max_cardinality)
        return self._result(
            dimension="Conformidad", rule_id=f"siaf.{col}.cardinality_upper",
            rule_name=f"Cardinalidad máxima: {col}", column_name=col,
            rule_type="cardinality_upper_bound", expected_value=f"<= {max_cardinality}",
            evaluated_rows=actual, failed_rows=failed_rows,
            severity=severity, source=source,
            observation=(
                f"Cardinalidad actual: {actual} | límite: {max_cardinality}. "
                + ("⚠️ Supera el máximo — posible versión nueva del catálogo MEF." if failed_rows > 0
                   else "OK.")
            ),
        )

    # ── Orquestación completa ────────────────────────────────────────────────

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta todas las reglas del YAML y devuelve:
        - detail_df : pd.DataFrame con una fila por regla.
        - failed_samples_df : pd.DataFrame con muestras de fallos (puede estar vacío).
        """
        logger.info("Iniciando evaluación de calidad SIAF | total_rows=%d", self.total_rows)
        results: list[RuleResult] = []

        cfg = self.config

        # ── Completitud ──────────────────────────────────────
        comp = cfg.get("completitud", {})

        for col in comp.get("columnas_criticas_no_nulas", {}).get("columns", []):
            meta = comp["columnas_criticas_no_nulas"]
            results.append(self.check_not_null(col, meta["severity"], meta["source"]))

        ubigeo_meta = comp.get("columnas_ubigeo_casi_completas", {})
        for col in ubigeo_meta.get("columns", []):
            results.append(self.check_null_threshold(
                col, ubigeo_meta["threshold_pct"],
                ubigeo_meta["severity"], ubigeo_meta["source"],
            ))

        cond_meta = comp.get("columnas_vacias_estructurales", {})
        for col in cond_meta.get("columns", []):
            results.append(self.check_conditional_null_allowed(
                col,
                condition_col=cond_meta["null_allowed_when"]["column"],
                condition_val=cond_meta["null_allowed_when"]["value"],
                expected_null_pct_when_match=cond_meta.get("expected_null_pct_when_m", 100.0),
                expected_null_pct_when_not_match=cond_meta.get("expected_null_pct_when_not_m", 0.0),
                severity=cond_meta["severity"], source=cond_meta["source"],
            ))

        # ── Validez ──────────────────────────────────────────
        val = cfg.get("validez", {})

        for rule_key, rule in val.items():
            rt = rule.get("rule_type")
            sev = rule.get("severity", "Media")
            src = rule.get("source", "dictionary_mef")
            if rt == "accepted_values":
                col = rule.get("column") or rule.get("columns")
                if isinstance(col, str):
                    results.append(self.check_accepted_values(col, rule["accepted"], sev, src))
            elif rt == "numeric_type":
                for col in rule.get("columns", []):
                    results.append(self.check_numeric_type(col, sev, src))

        # ── Exactitud ────────────────────────────────────────
        exac = cfg.get("exactitud", {})

        for rule_key, rule in exac.items():
            rt = rule.get("rule_type")
            sev = rule.get("severity", "Media")
            src = rule.get("source", "profiling_bronze")
            col = rule.get("column", "")
            if rt == "accepted_range":
                results.append(self.check_range(
                    col,
                    min_val=rule.get("min_value"),
                    max_val=rule.get("max_value"),
                    severity=sev, source=src,
                ))
            elif rt == "regex":
                results.append(self.check_regex(col, rule["pattern"], sev, src))

        # ── Consistencia ─────────────────────────────────────
        cons = cfg.get("consistencia", {})

        for rule_key, rule in cons.items():
            rt = rule.get("rule_type")
            sev = rule.get("severity", "Media")
            src = rule.get("source", "profiling_bronze")

            if rt == "conditional_value":
                results.append(self.check_conditional_value(
                    rule["condition_column"], rule["condition_value"],
                    rule["expected_column"], rule["expected_value"],
                    sev, src, rule_key,
                ))
            elif rt == "code_name_consistency":
                for pair in rule.get("pairs", []):
                    results.append(self.check_code_name_consistency(
                        pair["code"], pair["name"], sev, src,
                    ))
            elif rt == "column_comparison":
                results.append(self.check_column_comparison(
                    rule["left_column"], rule["operator"], rule["right_column"],
                    rule.get("filter_condition"), sev, src, rule_key,
                ))
            elif rt == "null_only_when":
                results.append(self.check_null_only_when(
                    rule["column"], rule["null_only_when_column"],
                    rule["null_only_when_value"], sev, src,
                ))

        # ── Unicidad ─────────────────────────────────────────
        uni = cfg.get("unicidad", {})
        dup_meta = uni.get("filas_exactas_duplicadas", {})
        results.append(self.check_duplicate_full_rows(
            dup_meta.get("severity", "Media"),
            dup_meta.get("source", "profiling_bronze"),
        ))
        key_meta = uni.get("clave_negocio_completa", {})
        results.append(self.check_unique_key(
            key_meta.get("columns", []),
            key_meta.get("severity", "Alta"),
            key_meta.get("source", "inferred_business_key"),
        ))

        # ── Integridad ───────────────────────────────────────
        integ = cfg.get("integridad", {})

        for rule_key, rule in integ.items():
            rt = rule.get("rule_type")
            sev = rule.get("severity", "Media")
            src = rule.get("source", "business_rule")
            if rt == "parent_not_zero_if_child_nonzero":
                results.append(self.check_parent_not_zero_if_child_nonzero(
                    rule["parent_column"], rule["child_column"], sev, src,
                ))
            elif rt == "cardinality_check":
                for col, exp_card in rule.get("expected_cardinality", {}).items():
                    results.append(self.check_cardinality(col, int(exp_card), sev, src))

        # ── Oportunidad ──────────────────────────────────────
        opor = cfg.get("oportunidad", {})

        ano_futuro = opor.get("ano_no_futuro", {})
        results.append(self.check_year_not_future(
            ano_futuro.get("column", "ANO_DOC"),
            ano_futuro.get("severity", "Media"),
            ano_futuro.get("source", "business_rule"),
        ))

        mes_meta = opor.get("mes_valido", {})
        results.append(self.check_range(
            mes_meta.get("column", "MES_DOC"),
            min_val=mes_meta.get("min_value", 1),
            max_val=mes_meta.get("max_value", 12),
            severity=mes_meta.get("severity", "Alta"),
            source=mes_meta.get("source", "profiling_bronze"),
        ))

        cov_meta = opor.get("cobertura_temporal_esperada", {})
        results.append(self.check_temporal_coverage(
            year_col="ANO_DOC", month_col="MES_DOC",
            expected_months=cov_meta.get("expected_months_per_year", 12),
            exception_years=cov_meta.get("exception_years", [2026]),
            severity=cov_meta.get("severity", "Baja"),
            source=cov_meta.get("source", "profiling_bronze"),
        ))

        # ── Conformidad ──────────────────────────────────────
        conf = cfg.get("conformidad", {})

        for rule_key, rule in conf.items():
            rt = rule.get("rule_type")
            sev = rule.get("severity", "Media")
            src = rule.get("source", "profiling_bronze")
            col = rule.get("column", "")
            if rt == "regex":
                skip = None
                if "null_when_nivel_gobierno" in rule:
                    skip = ("NIVEL_GOBIERNO", rule["null_when_nivel_gobierno"])
                results.append(self.check_regex(col, rule["pattern"], sev, src, skip_when=skip))
            elif rt == "cardinality_upper_bound":
                results.append(self.check_cardinality_upper_bound(
                    col, rule["max_cardinality"], sev, src,
                ))

        logger.info("Evaluación completada | reglas=%d", len(results))

        detail_df = pd.DataFrame([r.__dict__ for r in results])
        # failed_samples se capturan dentro de sample_failed_rows como JSON string
        # compatibilidad con SISMEPRE: devolvemos DF vacío de samples (ya incrustados)
        failed_samples_df = pd.DataFrame(columns=[
            "dataset", "rule_id", "dimension", "column_name",
            "row_index", "expected_value",
        ])
        return detail_df, failed_samples_df