"""
siaf_client.py — Cliente para descargar datos RAW del SIAF Ingresos.

Soporta dos modos de extracción:
  1. CSV históricos (2012–2021): descarga directa de archivos CSV.
  2. API CKAN (2022–2026): extracción paginada completa.

Filosofía RAW:
  - NO se modifica ningún valor.
  - NO se castean tipos.
  - NO se eliminan columnas ni filas.
  - Los DataFrames se entregan exactamente como los parsea pandas desde la fuente.
"""

from io import StringIO
from typing import Any, Iterator, Optional

import pandas as pd

from app.clients.base_client import BaseHTTPClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Encoding más común en los CSV del MEF; si falla, se reintenta con latin-1
_CSV_ENCODINGS = ["utf-8", "latin-1", "cp1252"]


class SIAFClient(BaseHTTPClient):
    """
    Cliente de extracción RAW para el Sistema Integrado de Administración
    Financiera (SIAF) — módulo de Ingresos.
    """

    def __init__(
        self,
        api_base: str,
        ckan_page_limit: int = 10000,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        api_base : str
            URL base del endpoint CKAN del MEF.
        ckan_page_limit : int
            Número máximo de registros por página CKAN.
        **kwargs
            Parámetros adicionales pasados a BaseHTTPClient.
        """
        super().__init__(**kwargs)
        self.api_base = api_base
        self.ckan_page_limit = ckan_page_limit

    # ------------------------------------------------------------------
    # Modo 1: CSV históricos directos
    # ------------------------------------------------------------------

    def fetch_csv_raw(self, dataset_key: str, url: str) -> pd.DataFrame:
        """
        Descarga un CSV histórico y lo retorna como DataFrame RAW.

        - Todos los campos se leen como string (dtype=str) para no perder ceros
          iniciales ni alterar formatos originales.
        - low_memory=False evita inferencias de tipo mixtas.
        - keep_default_na=False preserva los strings vacíos sin convertirlos a NaN.
          (se puede cambiar si se prefiere NaN para análisis posteriores; aquí
           mantenemos la semántica original del CSV).

        Parameters
        ----------
        dataset_key : str
            Identificador del dataset (ej.: '2015_ingreso').
        url : str
            URL directa del archivo CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame con todos los registros y columnas originales.
        """
        logger.info("Descargando CSV | dataset=%s | url=%s", dataset_key, url)

        content_bytes = self.download_binary(url)

        # Intenta decodificar con los encodings conocidos del MEF
        content_str: Optional[str] = None
        for enc in _CSV_ENCODINGS:
            try:
                content_str = content_bytes.decode(enc)
                logger.debug("CSV decodificado con encoding=%s", enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content_str is None:
            raise ValueError(
                f"No se pudo decodificar el CSV '{dataset_key}' con ninguno de: {_CSV_ENCODINGS}"
            )

        df = pd.read_csv(
            StringIO(content_str),
            dtype=str,           # RAW: todo como string, sin inferencia de tipos
            keep_default_na=False,
            low_memory=False,
            on_bad_lines="warn", # Registra filas problemáticas sin abortar
        )

        logger.info(
            "CSV RAW cargado | dataset=%s | rows=%d | cols=%d",
            dataset_key,
            len(df),
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Modo 2: API CKAN — paginación completa
    # ------------------------------------------------------------------

    def fetch_ckan_all_pages(
        self, dataset_key: str, resource_id: str
    ) -> Iterator[tuple[pd.DataFrame, dict]]:
        """
        Genera pares (DataFrame_página, json_raw_página) para cada página
        del recurso CKAN hasta agotar todos los registros.

        Usa el parámetro offset para paginar hasta obtener el total declarado
        por la API en result.total.

        Parameters
        ----------
        dataset_key : str
            Identificador del dataset (ej.: '2022_ingreso').
        resource_id : str
            UUID del recurso en el datastore CKAN del MEF.

        Yields
        ------
        tuple[pd.DataFrame, dict]
            DataFrame con los registros de la página y el JSON RAW completo.
        """
        offset = 0
        total: Optional[int] = None
        page_num = 0

        logger.info(
            "Iniciando extracción CKAN | dataset=%s | resource_id=%s | limit=%d",
            dataset_key,
            resource_id,
            self.ckan_page_limit,
        )

        while True:
            params = {
                "resource_id": resource_id,
                "limit": self.ckan_page_limit,
                "offset": offset,
            }

            response = self.get(self.api_base, params=params)
            raw_json = response.json()

            # Valida respuesta CKAN
            # Nota: la API del MEF tiene typo "sucess" (una sola c)
            success = (
                raw_json.get("success", False)
                or str(raw_json.get("sucess", "false")).lower() == "true"
            )
            if not success:
                error_msg = raw_json.get("error", {})
                raise RuntimeError(
                    f"CKAN error | dataset={dataset_key} | error={error_msg}"
                )

            # La API del MEF devuelve "records" en la RAÍZ del JSON,
            # no dentro de "result". El dict "result" solo trae metadata.
            records = raw_json.get("records") or []
            result  = raw_json.get("result", {})

            if total is None:
                total = int(
                    result.get("include_total")
                    or result.get("total")
                    or 0
                )
                logger.info(
                    "Total registros declarado por CKAN | dataset=%s | total=%d",
                    dataset_key,
                    total,
                )

            # Sin registros en primera página → dataset vacío
            if not records and page_num == 0:
                logger.info(
                    "No hay registros | dataset=%s",
                    dataset_key,
                )
                break

            if not records:
                logger.info(
                    "No hay más registros | dataset=%s | offset=%d",
                    dataset_key,
                    offset,
                )
                break

            page_num += 1
            df_page = pd.DataFrame(records)

            logger.info(
                "Página %d extraída | dataset=%s | registros_pagina=%d | "
                "acumulado=%d/%d",
                page_num,
                dataset_key,
                len(records),
                offset + len(records),
                total,
            )

            yield df_page, raw_json

            offset += len(records)

            # Condición de parada: página incompleta o API no devuelve "next"
            # (misma lógica que el cliente del otro grupo, más confiable
            #  que comparar offset >= total declarado)
            if len(records) < self.ckan_page_limit:
                logger.info(
                    "Extracción completa | dataset=%s | total_extraído=%d",
                    dataset_key,
                    offset,
                )
                break

    def fetch_ckan_page_raw(
        self, resource_id: str, offset: int = 0
    ) -> dict:
        """
        Retorna el JSON RAW de una sola página CKAN (útil para debugging).

        Parameters
        ----------
        resource_id : str
            UUID del recurso.
        offset : int
            Offset de inicio.

        Returns
        -------
        dict
            Respuesta JSON completa de la API.
        """
        params = {
            "resource_id": resource_id,
            "limit": self.ckan_page_limit,
            "offset": offset,
        }
        response = self.get(self.api_base, params=params)
        return response.json()