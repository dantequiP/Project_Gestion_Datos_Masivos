"""
renamu_client.py — Cliente para descargar y extraer datos RAW del RENAMU.

El RENAMU (Registro Nacional de Municipalidades) del INEI distribuye sus
datos como archivos ZIP que contienen múltiples CSV o XLSX.

Este cliente:
  1. Descarga el ZIP binario sin modificarlo.
  2. Lo extrae en disco preservando la estructura original.
  3. Lee cada archivo CSV/XLSX del ZIP como DataFrame RAW.
  4. NO modifica ningún valor, tipo ni estructura.
"""

import io
import zipfile
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from app.clients.base_client import BaseHTTPClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CSV_ENCODINGS = ["utf-8", "latin-1", "cp1252"]


class RENAMUClient(BaseHTTPClient):
    """
    Cliente de extracción RAW para el Registro Nacional de Municipalidades
    (RENAMU) publicado por el INEI.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Parámetros para BaseHTTPClient (timeout, retries, etc.).
        """
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Descarga del ZIP
    # ------------------------------------------------------------------

    def download_zip_raw(self, url: str) -> bytes:
        """
        Descarga el archivo ZIP del RENAMU como bytes sin modificar.

        Parameters
        ----------
        url : str
            URL del archivo ZIP del INEI.

        Returns
        -------
        bytes
            Contenido binario del ZIP.
        """
        logger.info("Descargando ZIP RENAMU | url=%s", url)
        content = self.download_binary(url)
        logger.info("ZIP descargado | size=%d bytes", len(content))
        return content

    # ------------------------------------------------------------------
    # Extracción del ZIP en disco
    # ------------------------------------------------------------------

    def extract_zip_to_disk(
        self,
        zip_bytes: bytes,
        extract_path: Path,
    ) -> list[Path]:
        """
        Extrae todos los archivos del ZIP al directorio indicado,
        preservando la estructura de carpetas original.

        Parameters
        ----------
        zip_bytes : bytes
            Contenido binario del ZIP descargado.
        extract_path : Path
            Directorio destino de la extracción.

        Returns
        -------
        list[Path]
            Lista de rutas de los archivos extraídos.
        """
        extract_path = Path(extract_path)
        extract_path.mkdir(parents=True, exist_ok=True)

        extracted_files: list[Path] = []

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            members = zf.namelist()
            logger.info(
                "ZIP contiene %d archivos | extrayendo en: %s",
                len(members),
                extract_path,
            )
            for member in members:
                zf.extract(member, extract_path)
                full_path = extract_path / member
                extracted_files.append(full_path)
                logger.debug("Extraído: %s", full_path)

        logger.info(
            "Extracción completa | %d archivos extraídos en %s",
            len(extracted_files),
            extract_path,
        )
        return extracted_files

    # ------------------------------------------------------------------
    # Lectura RAW de archivos CSV extraídos
    # ------------------------------------------------------------------

    def read_csv_raw(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Lee un archivo CSV extraído del ZIP como DataFrame RAW.

        - dtype=str: preserva todos los valores exactamente.
        - keep_default_na=False: no convierte vacíos a NaN (opcional,
          comentar si se prefiere NaN para análisis Silver).
        - on_bad_lines='warn': registra filas problemáticas sin abortar.

        Parameters
        ----------
        csv_path : Path
            Ruta al archivo CSV.

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame RAW, o None si el archivo no es un CSV legible.
        """
        if not csv_path.is_file():
            logger.warning("Archivo no existe: %s", csv_path)
            return None

        if csv_path.suffix.lower() not in {".csv", ".txt"}:
            logger.debug("Omitiendo (no es CSV): %s", csv_path)
            return None

        logger.info("Leyendo CSV RAW | path=%s", csv_path)

        # El RENAMU del INEI usa ';' como separador. Intentamos ';' primero,
        # luego ',' como fallback para otros CSV que pudieran incluirse.
        separators = [";", ",", "\t"]

        for enc in _CSV_ENCODINGS:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        csv_path,
                        sep=sep,
                        dtype=str,
                        keep_default_na=False,
                        low_memory=False,
                        encoding=enc,
                        on_bad_lines="warn",
                    )
                    # Si solo tiene 1 columna con este separador, no es el correcto
                    if len(df.columns) == 1 and len(separators) > 1:
                        logger.debug(
                            "sep='%s' enc='%s' produjo 1 columna — probando siguiente separador",
                            sep, enc,
                        )
                        continue
                    logger.info(
                        "CSV leído | path=%s | encoding=%s | sep='%s' | rows=%d | cols=%d",
                        csv_path.name,
                        enc,
                        sep,
                        len(df),
                        len(df.columns),
                    )
                    return df
                except UnicodeDecodeError:
                    logger.debug("Encoding %s falló para %s", enc, csv_path.name)
                    break  # Probar siguiente encoding con todos los seps
                except Exception as exc:
                    logger.error("Error leyendo CSV %s: %s", csv_path, exc)
                    return None

        logger.error(
            "No se pudo leer %s con ningún encoding de: %s", csv_path.name, _CSV_ENCODINGS
        )
        return None

    # ------------------------------------------------------------------
    # Lectura RAW de archivos XLSX extraídos
    # ------------------------------------------------------------------

    def read_xlsx_raw(self, xlsx_path: Path) -> dict[str, pd.DataFrame]:
        """
        Lee todas las hojas de un XLSX como DataFrames RAW.

        Parameters
        ----------
        xlsx_path : Path
            Ruta al archivo XLSX.

        Returns
        -------
        dict[str, pd.DataFrame]
            Diccionario {nombre_hoja: DataFrame_RAW}.
        """
        if not xlsx_path.is_file():
            logger.warning("Archivo no existe: %s", xlsx_path)
            return {}

        if xlsx_path.suffix.lower() not in {".xlsx", ".xls"}:
            logger.debug("Omitiendo (no es XLSX): %s", xlsx_path)
            return {}

        logger.info("Leyendo XLSX RAW | path=%s", xlsx_path)
        result: dict[str, pd.DataFrame] = {}

        try:
            excel_file = pd.ExcelFile(xlsx_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    dtype=str,
                    keep_default_na=False,
                )
                logger.info(
                    "Hoja leída | file=%s | sheet=%s | rows=%d | cols=%d",
                    xlsx_path.name,
                    sheet_name,
                    len(df),
                    len(df.columns),
                )
                result[sheet_name] = df
        except Exception as exc:
            logger.error("Error leyendo XLSX %s: %s", xlsx_path, exc)

        return result

    # ------------------------------------------------------------------
    # Generador: itera todos los archivos CSV/XLSX del ZIP
    # ------------------------------------------------------------------

    def iter_extracted_dataframes(
        self, extracted_files: list[Path]
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """
        Itera sobre todos los archivos extraídos y genera pares
        (identificador, DataFrame_RAW) para cada CSV u hoja de XLSX.

        Parameters
        ----------
        extracted_files : list[Path]
            Lista de archivos extraídos del ZIP.

        Yields
        ------
        tuple[str, pd.DataFrame]
            (identificador_único, DataFrame_RAW)
        """
        for file_path in extracted_files:
            suffix = file_path.suffix.lower()

            if suffix in {".csv", ".txt"}:
                df = self.read_csv_raw(file_path)
                if df is not None:
                    yield file_path.stem, df

            elif suffix in {".xlsx", ".xls"}:
                sheets = self.read_xlsx_raw(file_path)
                for sheet_name, df in sheets.items():
                    identifier = f"{file_path.stem}__{sheet_name}"
                    yield identifier, df

            else:
                logger.debug("Archivo omitido (tipo no soportado): %s", file_path.name)
