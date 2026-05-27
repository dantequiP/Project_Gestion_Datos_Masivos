"""
data_lake.py — Gestor de almacenamiento para la capa Bronze (RAW).

Responsabilidades:
  - Construir rutas de particionado por fecha (year/month/day).
  - Guardar DataFrames como Parquet RAW sin modificar ningún valor.
  - Guardar JSON RAW originales tal como llegan de la API.
  - Guardar archivos binarios (ZIP, PDF) sin modificar.
  - Crear directorios automáticamente.
  - Registrar metadatos mínimos de cada escritura.

Filosofía: NADA se modifica. Si el dato viene sucio, se guarda sucio.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataLakeStorage:
    """
    Gestiona la escritura en la capa Bronze del data lake.

    Toda escritura es append-friendly y particionada por fecha de ejecución,
    lo que permite reprocesar sin sobrescribir datos históricos.
    """

    def __init__(
        self,
        bronze_root: Path,
        compression: str = "snappy",
    ) -> None:
        """
        Parameters
        ----------
        bronze_root : Path
            Raíz de la capa bronze, ej.: data/bronze
        compression : str
            Algoritmo de compresión Parquet (snappy, gzip, zstd, none).
        """
        self.bronze_root = Path(bronze_root)
        self.compression = compression
        self.bronze_root.mkdir(parents=True, exist_ok=True)
        logger.info("DataLakeStorage inicializado. Root: %s", self.bronze_root)

    # ------------------------------------------------------------------
    # Construcción de rutas
    # ------------------------------------------------------------------

    def build_partition_path(
        self,
        source: str,
        dataset: str,
        execution_dt: Optional[datetime] = None,
    ) -> Path:
        """
        Construye la ruta de particionado: bronze/<source>/<dataset>/year=YYYY/month=MM/day=DD/

        Parameters
        ----------
        source : str
            Nombre de la fuente (siaf, sismepre, renamu).
        dataset : str
            Nombre del dataset (ej.: 2022_ingreso).
        execution_dt : Optional[datetime]
            Fecha de ejecución. Por defecto: now().

        Returns
        -------
        Path
            Ruta de particionado lista para escribir.
        """
        dt = execution_dt or datetime.now()
        partition = (
            self.bronze_root
            / source
            / dataset
            / f"year={dt.year}"
            / f"month={dt.month:02d}"
            / f"day={dt.day:02d}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        return partition

    # ------------------------------------------------------------------
    # Escritura Parquet RAW
    # ------------------------------------------------------------------

    def save_parquet_raw(
        self,
        df: pd.DataFrame,
        source: str,
        dataset: str,
        execution_dt: Optional[datetime] = None,
        filename_suffix: str = "",
    ) -> Path:
        """
        Guarda un DataFrame como Parquet RAW preservando toda la data original.

        - NO castea tipos.
        - NO elimina columnas.
        - NO elimina nulos ni duplicados.
        - NO normaliza valores.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos crudos tal como se descargaron.
        source : str
            Nombre de la fuente (siaf, sismepre, renamu).
        dataset : str
            Nombre del dataset.
        execution_dt : Optional[datetime]
            Fecha de ejecución para el particionado.
        filename_suffix : str
            Sufijo opcional para el nombre del archivo (ej.: "_page_1").

        Returns
        -------
        Path
            Ruta completa del archivo Parquet generado.
        """
        dt = execution_dt or datetime.now()
        partition_path = self.build_partition_path(source, dataset, dt)

        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset}{filename_suffix}_{timestamp_str}.parquet"
        file_path = partition_path / filename

        # Conversión directa: pandas → pyarrow → parquet
        # preserve_index=False evita columna extra de índice
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            file_path,
            compression=self.compression,
        )

        logger.info(
            "Parquet RAW guardado | source=%s | dataset=%s | rows=%d | path=%s",
            source,
            dataset,
            len(df),
            file_path,
        )
        return file_path

    def save_parquet_flat(
        self,
        df: pd.DataFrame,
        source: str,
        dataset: str,
        filename: str,
    ) -> Path:
        """
        Guarda un DataFrame como Parquet RAW directamente en la raíz del dataset,
        sin subcarpetas de particionado por fecha.

        Estructura resultante:
            bronze/<source>/<dataset>/<filename>.parquet

        Usado por SISMEPRE para producir un único archivo por dataset:
            bronze/sismepre/rentas_preguntas/rentas_preguntas_raw.parquet

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos crudos.
        source : str
            Nombre de la fuente (sismepre, etc.).
        dataset : str
            Nombre del dataset.
        filename : str
            Nombre del archivo sin extensión (ej.: "rentas_preguntas_raw").

        Returns
        -------
        Path
            Ruta completa del archivo Parquet generado.
        """
        dataset_root = self.bronze_root / source / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)

        file_path = dataset_root / f"{filename}.parquet"

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, file_path, compression=self.compression)

        logger.info(
            "Parquet RAW guardado | source=%s | dataset=%s | rows=%d | path=%s",
            source,
            dataset,
            len(df),
            file_path,
        )
        return file_path

    # ------------------------------------------------------------------
    # Escritura JSON RAW
    # ------------------------------------------------------------------

    def save_json_raw(
        self,
        data: Any,
        source: str,
        dataset: str,
        execution_dt: Optional[datetime] = None,
        filename_suffix: str = "",
    ) -> Path:
        """
        Guarda la respuesta JSON cruda de una API exactamente como llegó.

        Parameters
        ----------
        data : Any
            Objeto Python (dict/list) resultado de json.loads().
        source : str
            Nombre de la fuente.
        dataset : str
            Nombre del dataset.
        execution_dt : Optional[datetime]
            Fecha de ejecución para el particionado.
        filename_suffix : str
            Sufijo opcional para el nombre del archivo.

        Returns
        -------
        Path
            Ruta completa del archivo JSON generado.
        """
        dt = execution_dt or datetime.now()
        partition_path = self.build_partition_path(source, dataset, dt)

        # Sub-carpeta raw_json dentro de la partición
        json_path = partition_path / "raw_json"
        json_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset}{filename_suffix}_{timestamp_str}.json"
        file_path = json_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug("JSON RAW guardado | path=%s", file_path)
        return file_path

    # ------------------------------------------------------------------
    # Escritura de archivos binarios (ZIP, PDF, etc.)
    # ------------------------------------------------------------------

    def save_binary_raw(
        self,
        content: bytes,
        source: str,
        dataset: str,
        original_filename: str,
        execution_dt: Optional[datetime] = None,
    ) -> Path:
        """
        Guarda un archivo binario (ZIP, PDF) sin ninguna modificación.

        Parameters
        ----------
        content : bytes
            Contenido binario descargado.
        source : str
            Nombre de la fuente.
        dataset : str
            Nombre del dataset.
        original_filename : str
            Nombre del archivo con extensión (ej.: 2022.zip).
        execution_dt : Optional[datetime]
            Fecha de ejecución para el particionado.

        Returns
        -------
        Path
            Ruta completa del archivo guardado.
        """
        dt = execution_dt or datetime.now()
        partition_path = self.build_partition_path(source, dataset, dt)

        file_path = partition_path / original_filename
        file_path.write_bytes(content)

        logger.info(
            "Binario RAW guardado | source=%s | dataset=%s | size=%d bytes | path=%s",
            source,
            dataset,
            len(content),
            file_path,
        )
        return file_path

    # ------------------------------------------------------------------
    # Auditoría básica de escritura
    # ------------------------------------------------------------------

    def save_audit_record(
        self,
        audit_dir: Path,
        record: dict,
    ) -> None:
        """
        Guarda un registro de auditoría JSON en el directorio indicado.

        Parameters
        ----------
        audit_dir : Path
            Directorio de auditoría (ej.: data/audit/executions).
        record : dict
            Diccionario con metadatos de la ejecución.
        """
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"audit_{record.get('source', 'unknown')}_{record.get('dataset', 'unknown')}_{timestamp_str}.json"
        file_path = audit_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.debug("Auditoría guardada | path=%s", file_path)