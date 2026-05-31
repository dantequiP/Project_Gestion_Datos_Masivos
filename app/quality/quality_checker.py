from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__, log_dir=Path("logs"))


@dataclass
class RuleResult:
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


def safe_strip(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


class QualityChecker:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.settings = config["settings"]
        self.missing_tokens = set(config.get("missing_tokens", []))
        self.bronze_root = Path(self.settings.get("bronze_root", "data/bronze"))
        self.datasets_config = config.get("datasets", {})
        self.global_rules = config.get("global_rules", {})
        self.referential_rules = config.get("referential_integrity", [])
        self.dataframes: dict[str, pd.DataFrame] = {}

    def load_dataset(self, dataset: str) -> pd.DataFrame:
        if dataset in self.dataframes:
            return self.dataframes[dataset]
        path = self.bronze_root / "sismepre" / dataset / f"{dataset}_raw.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No existe Parquet Bronze para {dataset}: {path}")
        df = pd.read_parquet(path)
        self.dataframes[dataset] = df
        return df

    def clean_series(self, series: pd.Series) -> pd.Series:
        return series.map(safe_strip)

    def missing_mask(self, series: pd.Series) -> pd.Series:
        clean = self.clean_series(series)
        return series.isna() | clean.isin(self.missing_tokens)

    def result(self, dataset, dimension, rule_id, rule_name, column_name, rule_type, expected_value, evaluated_rows, failed_rows, severity, source, observation) -> RuleResult:
        evaluated_rows, failed_rows = int(evaluated_rows), int(failed_rows)
        passed_rows = max(evaluated_rows - failed_rows, 0)
        if evaluated_rows == 0:
            score = None
            status = "No evaluable"
        else:
            score = round((passed_rows / evaluated_rows) * 100, 4)
            status = "Sin observaciones" if failed_rows == 0 else "Con observaciones"
        return RuleResult(dataset, dimension, rule_id, rule_name, column_name, rule_type, str(expected_value), evaluated_rows, passed_rows, failed_rows, score, severity, source, status, observation)

    def required_column(self, dataset, df, column, meta):
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Completitud"), f"{dataset}.{column}.required", f"Columna requerida presente y no vacía: {column}", column, "required_not_empty", "no nulo/no vacío", 1, 1, meta.get("severity","Alta"), meta.get("source","dictionary_csv"), "La columna no existe en el Parquet Bronze.")
        failed = self.missing_mask(df[column]).sum()
        return self.result(dataset, meta.get("dimension","Completitud"), f"{dataset}.{column}.required", f"Columna requerida no vacía: {column}", column, "required_not_empty", "no nulo/no vacío", len(df), failed, meta.get("severity","Alta"), meta.get("source","dictionary_csv"), f"Registros vacíos o nulos en {column}.")

    def observed_empty(self, dataset, df, column, meta):
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Completitud"), f"{dataset}.{column}.observed_empty", f"Columna observada como vacía: {column}", column, "observed_empty", "revisar ausencia", 1, 1, meta.get("severity","Baja"), meta.get("source","profiling_based"), "La columna no existe en el Parquet Bronze.")
        failed = self.missing_mask(df[column]).sum()
        return self.result(dataset, meta.get("dimension","Completitud"), f"{dataset}.{column}.observed_empty", f"Columna con ausencia observada: {column}", column, "observed_empty", "revisar ausencia", len(df), failed, meta.get("severity","Baja"), meta.get("source","profiling_based"), "Columna reportada por profiling como potencialmente vacía o incompleta.")

    def allowed_values(self, dataset, df, column, meta):
        values = [str(v) for v in meta.get("values", [])]
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.allowed_values", f"Valores permitidos en {column}", column, "allowed_values", values, 1, 1, meta.get("severity","Media"), meta.get("source","dictionary_csv"), "La columna no existe en el Parquet Bronze.")
        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        failed = (evaluable & ~clean.isin(values)).sum()
        return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.allowed_values", f"Valores permitidos en {column}", column, "allowed_values", values, evaluable.sum(), failed, meta.get("severity","Media"), meta.get("source","dictionary_csv"), f"Valores fuera del catálogo permitido {values}. Los vacíos se evalúan en Completitud.")

    def numeric_column(self, dataset, df, column, meta, integer=False):
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.numeric", f"Columna numérica válida: {column}", column, "numeric", "numérico", 1, 1, meta.get("severity","Alta"), meta.get("source","dictionary_csv"), "La columna no existe en el Parquet Bronze.")
        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed_mask = numeric.isna()
        if integer and len(numeric) > 0:
            failed_mask = failed_mask | ((numeric.dropna() % 1 != 0).reindex(numeric.index, fill_value=False))
        return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.numeric", f"Columna convertible a número: {column}", column, "numeric", "numérico", evaluable.sum(), failed_mask.sum(), meta.get("severity","Alta"), meta.get("source","dictionary_csv"), "Valores no convertibles a número. Los vacíos se evalúan en Completitud.")

    def regex_rule(self, dataset, df, column, regex, meta, suffix):
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.{suffix}", f"Formato esperado en {column}", column, "regex", regex, 1, 1, meta.get("severity","Media"), meta.get("source","naming_pattern"), "La columna no existe en el Parquet Bronze.")
        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        failed = (evaluable & ~clean.str.match(regex, na=False)).sum()
        return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.{suffix}", f"Formato esperado en {column}", column, "regex", regex, evaluable.sum(), failed, meta.get("severity","Media"), meta.get("source","naming_pattern"), f"Valores que no cumplen el patrón {regex}. Los vacíos se evalúan en Completitud.")

    def year_rule(self, dataset, df, column, meta):
        min_year, max_year = int(meta.get("min_year", 2000)), int(meta.get("max_year", 2030))
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Exactitud"), f"{dataset}.{column}.year_4_digits", f"Año válido: {column}", column, "year_4_digits", f"{min_year}-{max_year}", 1, 1, meta.get("severity","Alta"), meta.get("source","naming_pattern"), "La columna no existe.")
        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed = numeric.isna() | (~clean[evaluable].str.match(r"^\\d{4}$", na=False)) | (numeric < min_year) | (numeric > max_year)
        return self.result(dataset, meta.get("dimension","Exactitud"), f"{dataset}.{column}.year_4_digits", f"Año de 4 dígitos en rango: {column}", column, "year_4_digits", f"{min_year}-{max_year}", evaluable.sum(), failed.sum(), meta.get("severity","Alta"), meta.get("source","naming_pattern"), f"Valores que no tienen 4 dígitos o no están entre {min_year} y {max_year}.")

    def date_rule(self, dataset, df, column, meta):
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.date_parseable", f"Fecha válida: {column}", column, "date_parseable", "fecha parseable", 1, 1, meta.get("severity","Media"), meta.get("source","dictionary_csv"), "La columna no existe en el Parquet Bronze.")
        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        parsed = pd.to_datetime(clean[evaluable], dayfirst=True, errors="coerce")
        failed = parsed.isna()
        return self.result(dataset, meta.get("dimension","Validez"), f"{dataset}.{column}.date_parseable", f"Fecha parseable: {column}", column, "date_parseable", "fecha válida", evaluable.sum(), failed.sum(), meta.get("severity","Media"), meta.get("source","dictionary_csv"), "Valores no convertibles a fecha. Los vacíos se evalúan en Completitud.")

    def non_negative(self, dataset, df, column, meta, integer=False):
        if column not in df.columns:
            return self.result(dataset, meta.get("dimension","Razonabilidad"), f"{dataset}.{column}.non_negative", f"Valor no negativo: {column}", column, "non_negative", ">=0", 1, 1, meta.get("severity","Alta"), meta.get("source","naming_pattern"), "La columna no existe.")
        clean = self.clean_series(df[column])
        evaluable = ~self.missing_mask(df[column])
        numeric = pd.to_numeric(clean[evaluable], errors="coerce")
        failed = numeric.isna() | (numeric < 0)
        if integer and len(numeric) > 0:
            failed = failed | ((numeric.dropna() % 1 != 0).reindex(numeric.index, fill_value=False))
        return self.result(dataset, meta.get("dimension","Razonabilidad"), f"{dataset}.{column}.non_negative", f"Valor no negativo: {column}", column, "non_negative", ">=0", evaluable.sum(), failed.sum(), meta.get("severity","Alta"), meta.get("source","naming_pattern"), "Valores negativos o no numéricos en campo cuantitativo.")

    def duplicate_rows(self, dataset, df, meta):
        failed = df.duplicated().sum()
        return self.result(dataset, meta.get("dimension","Unicidad"), f"{dataset}.full_row_duplicates", "Filas completamente duplicadas", "*", "duplicate_full_rows", "sin duplicados completos", len(df), failed, meta.get("severity","Media"), meta.get("source","profiling_based"), "Filas idénticas repetidas en el dataset.")

    def unique_key(self, dataset, df, rule):
        columns = rule.get("columns", [])
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return self.result(dataset, rule.get("dimension","Unicidad"), f"{dataset}.unique_key.{'+'.join(columns)}", f"Clave candidata única: {columns}", ",".join(columns), "unique_key", columns, 1, 1, rule.get("severity","Media"), rule.get("source","inferred_business_key"), f"No existen columnas de clave: {missing}.")
        failed = df.duplicated(subset=columns).sum()
        return self.result(dataset, rule.get("dimension","Unicidad"), f"{dataset}.unique_key.{'+'.join(columns)}", f"Clave candidata única: {columns}", ",".join(columns), "unique_key", columns, len(df), failed, rule.get("severity","Media"), rule.get("source","inferred_business_key"), "Registros duplicados según clave candidata.")

    def less_or_equal(self, dataset, df, rule):
        left, right = rule["left_column"], rule["right_column"]
        if left not in df.columns or right not in df.columns:
            return self.result(dataset, rule.get("dimension","Consistencia"), f"{dataset}.{left}_le_{right}", f"{left} <= {right}", f"{left},{right}", "less_or_equal", f"{left}<={right}", 1, 1, rule.get("severity","Alta"), rule.get("source","dictionary_csv"), "No existen columnas requeridas.")
        left_clean, right_clean = self.clean_series(df[left]), self.clean_series(df[right])
        evaluable = ~(self.missing_mask(df[left]) | self.missing_mask(df[right]))
        left_num, right_num = pd.to_numeric(left_clean[evaluable], errors="coerce"), pd.to_numeric(right_clean[evaluable], errors="coerce")
        failed = left_num.isna() | right_num.isna() | (left_num > right_num)
        return self.result(dataset, rule.get("dimension","Consistencia"), f"{dataset}.{left}_le_{right}", f"Consistencia: {left} <= {right}", f"{left},{right}", "less_or_equal", f"{left}<={right}", evaluable.sum(), failed.sum(), rule.get("severity","Alta"), rule.get("source","dictionary_csv"), f"Registros donde {left} no es menor o igual que {right}.")

    def pattern_rules(self, dataset, df):
        results = []
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

    def evaluate_dataset(self, dataset):
        df = self.load_dataset(dataset)
        rules = self.datasets_config.get(dataset, {})
        results = []
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

        results.extend(self.pattern_rules(dataset, df))
        logger.info("Calidad evaluada para %s | reglas=%s", dataset, len(results))
        return results

    def referential_integrity(self):
        results = []
        for rule in self.referential_rules:
            dataset, col = rule["dataset"], rule["column"]
            ref_dataset, ref_col = rule["reference_dataset"], rule["reference_column"]
            df, ref_df = self.load_dataset(dataset), self.load_dataset(ref_dataset)

            if col not in df.columns or ref_col not in ref_df.columns:
                results.append(self.result(dataset, rule.get("dimension","Integridad"), f"{dataset}.{col}.ref_integrity", f"Integridad referencial {dataset}.{col} → {ref_dataset}.{ref_col}", col, "referential_integrity", f"{ref_dataset}.{ref_col}", 1, 1, rule.get("severity","Media"), rule.get("source","referential_rule"), "No existen columnas requeridas."))
                continue

            left_values = self.clean_series(df[col])
            left_missing = self.missing_mask(df[col])
            ref_values = set(self.clean_series(ref_df[ref_col])[~self.missing_mask(ref_df[ref_col])].unique())
            evaluable = ~left_missing
            failed = (evaluable & ~left_values.isin(ref_values)).sum()
            results.append(self.result(dataset, rule.get("dimension","Integridad"), f"{dataset}.{col}.ref_integrity", f"Integridad referencial {dataset}.{col} → {ref_dataset}.{ref_col}", col, "referential_integrity", f"{ref_dataset}.{ref_col}", evaluable.sum(), failed, rule.get("severity","Media"), rule.get("source","referential_rule"), f"Valores de {dataset}.{col} que no existen en {ref_dataset}.{ref_col}."))
        return results

    def run(self) -> pd.DataFrame:
        all_results = []
        for dataset in self.datasets_config.keys():
            all_results.extend(self.evaluate_dataset(dataset))
        all_results.extend(self.referential_integrity())
        return pd.DataFrame([r.__dict__ for r in all_results])