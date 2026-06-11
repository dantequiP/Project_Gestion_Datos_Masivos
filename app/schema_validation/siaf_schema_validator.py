"""
siaf_schema_validator.py

Clase SiafSchemaValidator: valida el schema de los Parquet Bronze de SIAF
contra el contrato declarado en schema_rules_siaf.yaml.

Estrategia:
    - Usa df_spark.columns para extraer columnas reales sin collect().
    - Compara con expected_columns y critical_columns usando operaciones de set.
    - Si faltan columnas críticas → lanza RuntimeError (detiene el pipeline).
    - Devuelve un Pandas DataFrame con el detalle columna por columna.
    - PySpark hace todo el trabajo pesado; Pandas solo arma el reporte final.

Esta clase NO modifica datos Bronze.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import DataFrame


class SiafSchemaValidator:
    """
    Validador de schema para SIAF Ingresos.

    Parameters
    ----------
    config : dict
        Contenido completo de schema_rules_siaf.yaml ya cargado con yaml.safe_load.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.settings = config["settings"]

        # Columnas esperadas según el YAML (normalizadas a upper sin espacios)
        self.expected_columns: list[str] = [
            str(c).strip().upper() for c in config["expected_columns"]
        ]

        # Tipos lógicos por columna
        self.column_types: dict[str, dict[str, Any]] = {
            str(k).strip().upper(): v
            for k, v in config.get("column_types", {}).items()
        }

        # Columnas críticas: su ausencia detiene el pipeline
        self.critical_columns: set[str] = {
            str(c).strip().upper() for c in config.get("critical_columns", [])
        }

        # Política para columnas extra
        self.extra_policy: str = (
            config.get("extra_columns_policy", {}).get("action", "warn")
        )

    # ── Validación principal ──────────────────────────────────────────────────

    def validate(self, df_spark: DataFrame) -> pd.DataFrame:
        """
        Compara las columnas del DataFrame Spark contra el contrato YAML.

        Usa df_spark.columns (metadatos, sin collect) para eficiencia máxima
        en datasets de millones de filas.

        Parameters
        ----------
        df_spark : DataFrame
            DataFrame PySpark con todos los Parquets Bronze cargados.

        Returns
        -------
        pd.DataFrame
            Detalle columna por columna con estado de validación.

        Raises
        ------
        RuntimeError
            Si hay columnas críticas faltantes (detiene el pipeline completo).
        """
        print("[INFO] Iniciando validación de schema SIAF con PySpark...")

        # Columnas reales del Parquet (normalizadas)
        # df_spark.columns es una operación de metadatos: NO hace collect().
        columnas_parquet: set[str] = {
            str(c).strip().upper() for c in df_spark.columns
        }
        columnas_esperadas: set[str] = set(self.expected_columns)

        # Operaciones de conjuntos
        presentes: set[str] = columnas_esperadas & columnas_parquet
        faltantes: set[str] = columnas_esperadas - columnas_parquet
        extras: set[str] = columnas_parquet - columnas_esperadas

        # Columnas críticas faltantes → error duro
        criticas_faltantes: set[str] = faltantes & self.critical_columns
        if criticas_faltantes:
            msg = (
                f"[CRITICAL] Columnas CRÍTICAS faltantes en Bronze SIAF: "
                f"{sorted(criticas_faltantes)}. El pipeline no puede continuar."
            )
            print(msg)
            raise RuntimeError(msg)

        print(f"[INFO] Columnas esperadas (YAML):  {len(columnas_esperadas)}")
        print(f"[INFO] Columnas en Parquet:        {len(columnas_parquet)}")
        print(f"[INFO] Columnas OK:                {len(presentes)}")
        print(f"[INFO] Faltantes en Parquet:       {len(faltantes)}")
        print(f"[INFO] Extras en Parquet:          {len(extras)}")

        if faltantes:
            print(f"[WARN] Columnas faltantes: {sorted(faltantes)}")
        if extras:
            print(f"[WARN] Columnas extras ({self.extra_policy}): {sorted(extras)}")

        rows: list[dict[str, Any]] = []

        # Columnas esperadas: presentes o faltantes
        for col in self.expected_columns:
            col_upper = col.upper()
            meta = self.column_types.get(col_upper, {})
            presente = col_upper in columnas_parquet
            es_critica = col_upper in self.critical_columns

            if presente:
                estado = "OK"
            else:
                estado = "FALTA_EN_PARQUET"

            rows.append({
                "columna": col_upper,
                "tipo_logico_esperado": meta.get("logical_type", ""),
                "nullable_esperado": meta.get("nullable", True),
                "es_critica": es_critica,
                "presente_en_parquet": presente,
                "estado": estado,
                "tipo_parquet": self._get_spark_type(df_spark, col_upper),
                "fuente_regla": meta.get("source", "schema_rules_siaf.yaml"),
            })

        # Columnas extra en Parquet (no declaradas en YAML)
        for col in sorted(extras):
            rows.append({
                "columna": col,
                "tipo_logico_esperado": "",
                "nullable_esperado": None,
                "es_critica": False,
                "presente_en_parquet": True,
                "estado": f"EXTRA_EN_PARQUET_{self.extra_policy.upper()}",
                "tipo_parquet": self._get_spark_type(df_spark, col),
                "fuente_regla": f"extra_columns_policy={self.extra_policy}",
            })

        detail_df = pd.DataFrame(rows)
        print(f"[INFO] Validación completada: {len(detail_df)} columnas evaluadas.")
        return detail_df

    # ── Construcción de resumen ───────────────────────────────────────────────

    def build_summary(
        self,
        detail_df: pd.DataFrame,
        total_rows: int,
        parquet_paths: list[str],
    ) -> pd.DataFrame:
        """
        Construye un resumen de una fila con las métricas principales.

        Este Pandas DataFrame es el que se escribe en el CSV de resumen
        y se embebe en el reporte HTML.
        """
        n_ok = int((detail_df["estado"] == "OK").sum())
        n_falta = int((detail_df["estado"] == "FALTA_EN_PARQUET").sum())
        n_extra = int(detail_df["estado"].str.startswith("EXTRA_EN_PARQUET").sum())
        n_criticas = int(
            (detail_df["es_critica"] & (detail_df["estado"] == "FALTA_EN_PARQUET")).sum()
        )
        estado = "FALLA_CRITICA" if n_criticas > 0 else ("REVISAR" if n_falta > 0 else "OK")

        resumen = {
            "fuente": "SIAF",
            "dataset": "ingresos",
            "contrato_yaml": "schema_rules_siaf.yaml",
            "estado_validacion": estado,
            "columnas_esperadas_yaml": len(
                [r for r in detail_df["estado"] if r != f"EXTRA_EN_PARQUET_{self.extra_policy.upper()}"]
            ),
            "columnas_en_parquet": int(detail_df["presente_en_parquet"].sum()),
            "columnas_ok": n_ok,
            "columnas_faltantes": n_falta,
            "columnas_extras": n_extra,
            "columnas_criticas_faltantes": n_criticas,
            "total_filas_bronze": total_rows,
            "archivos_parquet_procesados": len(parquet_paths),
            "politica_columnas_extra": self.extra_policy,
        }

        return pd.DataFrame([resumen])

    # ── Utilidades privadas ───────────────────────────────────────────────────

    @staticmethod
    def _get_spark_type(df_spark: DataFrame, column: str) -> str:
        """
        Extrae el tipo Spark de una columna desde el schema del DataFrame.
        Operación sobre metadatos: no hace collect().
        Retorna cadena vacía si la columna no existe.
        """
        try:
            field = next(
                (f for f in df_spark.schema.fields if f.name.upper() == column.upper()),
                None,
            )
            return str(field.dataType) if field else ""
        except Exception:
            return ""