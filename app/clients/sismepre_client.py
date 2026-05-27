"""
sismepre_client.py — Cliente para extraer datos RAW del SISMEPRE.

SISMEPRE (Sistema de Medición del Desempeño de la Recaudación Predial)
expone sus datos a través del API CKAN del MEF, igual que los recursos
SIAF modernos.

Extracción completamente paginada: todos los registros de cada recurso,
sin filtros, sin modificaciones.
"""

import csv
import io
from typing import Any, Iterator, Optional

import pandas as pd

from app.clients.base_client import BaseHTTPClient
from app.utils.logger import get_logger

# URL del endpoint dump CSV del MEF (fallback cuando datastore_search devuelve records=[])
_DUMP_URL_TEMPLATE = "https://datosabiertos.mef.gob.pe/datastore/dump/{resource_id}?bom=True"


logger = get_logger(__name__)


class SISMEPREClient(BaseHTTPClient):
    """
    Cliente de extracción RAW para el Sistema de Medición del Desempeño
    de la Recaudación Predial (SISMEPRE) del MEF.
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
            Registros por página CKAN.
        **kwargs
            Parámetros adicionales para BaseHTTPClient.
        """
        super().__init__(**kwargs)
        self.api_base = api_base
        self.ckan_page_limit = ckan_page_limit

    # ------------------------------------------------------------------
    # Extracción paginada completa
    # ------------------------------------------------------------------

    def fetch_all_pages(
        self, dataset_key: str, resource_id: str
    ) -> Iterator[tuple[pd.DataFrame, dict]]:
        """
        Genera pares (DataFrame_página, json_raw_página) hasta agotar
        todos los registros del recurso CKAN.

        Parameters
        ----------
        dataset_key : str
            Identificador del dataset (ej.: 'rentas_estadistica').
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
            "Iniciando extracción SISMEPRE | dataset=%s | resource_id=%s",
            dataset_key,
            resource_id,
        )

        while True:
            params = {
                "resource_id": resource_id,
                "limit": self.ckan_page_limit,
                "offset": offset,
            }

            response = self.get(self.api_base, params=params)
            raw_json = response.json()

            # Nota: la API del MEF tiene typo "sucess" (una sola c)
            success = (
                raw_json.get("success", False)
                or str(raw_json.get("sucess", "false")).lower() == "true"
            )
            if not success:
                error_info = raw_json.get("error", {})
                raise RuntimeError(
                    f"CKAN error | dataset={dataset_key} | error={error_info}"
                )

            # La API MEF devuelve "records" en la RAÍZ del JSON,
            # no dentro de "result". El dict "result" solo trae metadata.
            records = raw_json.get("records") or []
            result  = raw_json.get("result", {})

            if total is None:
                # El total viene en result.include_total (no en result.total)
                total = (
                    result.get("include_total")
                    or result.get("total")
                    or 0
                )
                logger.info(
                    "Total registros SISMEPRE | dataset=%s | total=%d",
                    dataset_key,
                    total,
                )

            # La API devuelve records=[] incluso cuando total>0 en algunos casos.
            # Activamos fallback CSV via /datastore/dump/ si offset==0 y no hay datos.
            if not records and offset == 0 and total and total > 0:
                logger.info(
                    "API devolvió records=[] con total=%d — activando CSV fallback | dataset=%s",
                    total,
                    dataset_key,
                )
                yield from self._fetch_csv_fallback(dataset_key, resource_id, raw_json)
                return  # CSV fallback agota todo, salimos del while

            if not records:
                logger.info(
                    "Extracción finalizada (sin más registros) | dataset=%s | offset=%d",
                    dataset_key,
                    offset,
                )
                break

            page_num += 1
            df_page = pd.DataFrame(records)

            logger.info(
                "Página %d | dataset=%s | registros=%d | acumulado=%d/%d",
                page_num,
                dataset_key,
                len(records),
                offset + len(records),
                total,
            )

            yield df_page, raw_json

            offset += len(records)

            if offset >= total:
                logger.info(
                    "Extracción completa | dataset=%s | total_extraído=%d",
                    dataset_key,
                    offset,
                )
                break

    def _fetch_csv_fallback(
        self, dataset_key: str, resource_id: str, original_raw_json: dict
    ) -> Iterator[tuple[pd.DataFrame, dict]]:
        """
        Fallback: descarga el CSV completo via /datastore/dump/<resource_id>.
        Se activa cuando datastore_search devuelve records=[] pese a total>0.
        El portal MEF expone este endpoint para todos los recursos CKAN.
        """
        dump_url = _DUMP_URL_TEMPLATE.format(resource_id=resource_id)
        logger.info(
            "CSV fallback | dataset=%s | url=%s",
            dataset_key,
            dump_url,
        )
        content_bytes = self.download_binary(dump_url)
        # El BOM (byte order mark) se elimina con utf-8-sig
        content_str = content_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content_str))
        records = list(reader)

        if not records:
            logger.warning(
                "CSV fallback también devolvió 0 registros | dataset=%s",
                dataset_key,
            )
            return

        df = pd.DataFrame(records)
        logger.info(
            "CSV fallback OK | dataset=%s | rows=%d | cols=%d",
            dataset_key,
            len(df),
            len(df.columns),
        )
        # Empaquetamos igual que el path normal para que ingestion_service no diferencie
        fallback_json = dict(original_raw_json)
        fallback_json["_source"] = "csv_fallback"
        fallback_json["records"] = records
        yield df, fallback_json

    def get_resource_info(self, resource_id: str) -> dict:
        """
        Retorna metadata del recurso CKAN (total de registros, campos, etc.)
        sin descargar los datos.

        Parameters
        ----------
        resource_id : str
            UUID del recurso.

        Returns
        -------
        dict
            Resultado parcial de la API (limit=0 trae solo metadata).
        """
        params = {
            "resource_id": resource_id,
            "limit": 0,
        }
        response = self.get(self.api_base, params=params)
        raw_json = response.json()
        result = raw_json.get("result", {})
        logger.debug(
            "Metadata recurso | resource_id=%s | total=%s | fields=%s",
            resource_id,
            result.get("total"),
            [f.get("id") for f in result.get("fields", [])],
        )
        return result