"""
ingestion_service.py — Servicio de orquestación de ingestión RAW.

Coordina la extracción completa de todas las fuentes (SIAF, SISMEPRE, RENAMU)
usando los clientes y el DataLakeStorage.

Responsabilidades:
  - Inicializar clientes con la configuración del YAML.
  - Ejecutar la ingestión de cada fuente y dataset.
  - Guardar Parquet RAW y JSON RAW por partición.
  - Registrar auditoría básica de cada ejecución.
  - Manejar errores de forma aislada (un dataset fallido no aborta los demás).
"""

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.clients.renamu_client import RENAMUClient
from app.clients.siaf_client import SIAFClient
from app.clients.sismepre_client import SISMEPREClient
from app.storage.data_lake import DataLakeStorage
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """
    Servicio principal de ingestión RAW para la capa Bronze.

    Orquesta la extracción de SIAF, SISMEPRE y RENAMU y delega
    el almacenamiento al DataLakeStorage.
    """

    def __init__(self, config: dict, audit_dir: Path) -> None:
        """
        Parameters
        ----------
        config : dict
            Configuración cargada desde config.yaml.
        audit_dir : Path
            Directorio para registros de auditoría.
        """
        self.config = config
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        bronze_root = Path(config["paths"]["bronze_root"])
        parquet_compression = config["parquet"]["compression"]

        self.storage = DataLakeStorage(
            bronze_root=bronze_root,
            compression=parquet_compression,
        )

        http_cfg = config["http"]
        ckan_limit = config["ckan"]["page_limit"]

        # Inicializar clientes
        logger.info("DEBUG base_url: '%s'", config["ckan"]["base_url"])
        self.siaf_client = SIAFClient(
            api_base="https://api.datosabiertos.mef.gob.pe/DatosAbiertos/v1/datastore_search",
            ckan_page_limit=ckan_limit,
            timeout=http_cfg["timeout"],
            max_retries=http_cfg["max_retries"],
            backoff_factor=http_cfg["backoff_factor"],
        )

        self.sismepre_client = SISMEPREClient(
            api_base=config["sources"]["sismepre"]["api_base"],
            ckan_page_limit=ckan_limit,
            timeout=http_cfg["timeout"],
            max_retries=http_cfg["max_retries"],
            backoff_factor=http_cfg["backoff_factor"],
        )

        self.renamu_client = RENAMUClient(
            timeout=http_cfg["timeout"],
            max_retries=http_cfg["max_retries"],
            backoff_factor=http_cfg["backoff_factor"],
        )

        logger.info("IngestionService inicializado correctamente.")

    # ------------------------------------------------------------------
    # Auditoría
    # ------------------------------------------------------------------

    def _record_audit(
        self,
        source: str,
        dataset: str,
        status: str,
        rows: int = 0,
        files_written: Optional[list] = None,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Registra un evento de auditoría básico en disco."""
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "dataset": dataset,
            "status": status,
            "rows_extracted": rows,
            "files_written": [str(f) for f in (files_written or [])],
        }
        if error:
            record["error"] = error
        if extra:
            record.update(extra)

        self.storage.save_audit_record(
            audit_dir=self.audit_dir / "executions",
            record=record,
        )

    # ------------------------------------------------------------------
    # SIAF — CSV históricos (2012–2021)
    # ------------------------------------------------------------------

    def ingest_siaf_csv(self) -> None:
        """
        Descarga todos los CSV históricos del SIAF (2012–2021) y los
        guarda como Parquet RAW + JSON RAW.
        """
        source = "siaf"
        csv_resources = self.config["sources"]["siaf"]["csv_resources"]
        execution_dt = datetime.now()

        logger.info("=== Iniciando ingestión SIAF CSV (2012–2021) ===")

        for dataset_key, url in csv_resources.items():
            try:
                logger.info("Procesando: %s", dataset_key)
                df = self.siaf_client.fetch_csv_raw(dataset_key, url)

                # Guardar Parquet RAW
                parquet_path = self.storage.save_parquet_raw(
                    df=df,
                    source=source,
                    dataset=dataset_key,
                    execution_dt=execution_dt,
                )

                # Guardar copia del CSV original como JSON de metadatos mínimos
                meta = {
                    "source": source,
                    "dataset": dataset_key,
                    "url": url,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "extraction_ts": execution_dt.isoformat(),
                }
                self.storage.save_json_raw(
                    data=meta,
                    source=source,
                    dataset=dataset_key,
                    execution_dt=execution_dt,
                    filename_suffix="_meta",
                )

                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="success",
                    rows=len(df),
                    files_written=[parquet_path],
                )
                logger.info("✓ %s completado | rows=%d", dataset_key, len(df))

            except Exception as exc:
                error_msg = traceback.format_exc()
                logger.error("✗ Error en %s: %s", dataset_key, exc)
                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="error",
                    error=error_msg,
                )

    # ------------------------------------------------------------------
    # SIAF — API CKAN (2022–2026)
    # ------------------------------------------------------------------

    def ingest_siaf_api(self) -> None:
        import pandas as pd

        source = "siaf"
        api_resources = self.config["sources"]["siaf"]["api_resources"]
        execution_dt = datetime.now()

        logger.info("=== Iniciando ingestión SIAF API CKAN (2022–2026) ===")

        for dataset_key, resource_id in api_resources.items():
            try:
                logger.info("Procesando: %s | resource_id=%s", dataset_key, resource_id)

                pages = []
                last_raw_json = {}

                for page_num, (df_page, raw_json) in enumerate(
                    self.siaf_client.fetch_ckan_all_pages(dataset_key, resource_id),
                    start=1,
                ):
                    pages.append(df_page)
                    last_raw_json = raw_json

                if not pages:
                    logger.warning("✗ %s — sin datos extraídos", dataset_key)
                    self._record_audit(source=source, dataset=dataset_key, status="empty")
                    continue

                # Un solo Parquet con todas las páginas concatenadas
                df_full = pd.concat(pages, ignore_index=True)
                total_rows = len(df_full)

                parquet_path = self.storage.save_parquet_flat(
                    df=df_full,
                    source=source,
                    dataset=dataset_key,
                    filename=f"{dataset_key}_raw",
                )

                # JSON de metadata
                meta = {
                    "source": source,
                    "dataset": dataset_key,
                    "resource_id": resource_id,
                    "rows": total_rows,
                    "columns": list(df_full.columns),
                    "pages_fetched": len(pages),
                    "extraction_ts": execution_dt.isoformat(),
                }
                self.storage.save_json_raw(
                    data=meta,
                    source=source,
                    dataset=dataset_key,
                    execution_dt=execution_dt,
                )

                written_files = [parquet_path]

                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="success",
                    rows=total_rows,
                    files_written=written_files,
                    extra={"pages": len(written_files)},
                )
                logger.info(
                    "✓ %s completado | total_rows=%d | pages=%d",
                    dataset_key,
                    total_rows,
                    len(written_files),
                )

            except Exception as exc:
                error_msg = traceback.format_exc()
                logger.error("✗ Error en %s: %s", dataset_key, exc)
                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="error",
                    error=error_msg,
                )

    # ------------------------------------------------------------------
    # SISMEPRE — API CKAN
    # ------------------------------------------------------------------

    def ingest_sismepre(self) -> None:
        """
        Extrae todos los recursos SISMEPRE vía API CKAN con paginación
        completa. Acumula todas las páginas y guarda UN SOLO Parquet RAW
        por dataset + un JSON de metadata.
        Estructura resultante:
            data/bronze/sismepre/<dataset>/
            │   <dataset>_raw.parquet
            └── <year>/<month>/<day>/
                    <timestamp>.json
        """
        import pandas as pd

        source = "sismepre"
        api_resources = self.config["sources"]["sismepre"]["api_resources"]
        execution_dt = datetime.now()

        logger.info("=== Iniciando ingestión SISMEPRE ===")

        for dataset_key, resource_id in api_resources.items():
            try:
                logger.info("Procesando: %s | resource_id=%s", dataset_key, resource_id)

                pages = []
                last_raw_json = {}

                for page_num, (df_page, raw_json) in enumerate(
                    self.sismepre_client.fetch_all_pages(dataset_key, resource_id),
                    start=1,
                ):
                    pages.append(df_page)
                    last_raw_json = raw_json

                if not pages:
                    logger.warning("✗ %s — sin datos extraídos", dataset_key)
                    self._record_audit(source=source, dataset=dataset_key, status="empty")
                    continue

                # Concatenar todas las páginas en un único DataFrame
                df_full = pd.concat(pages, ignore_index=True)
                total_rows = len(df_full)

                # Guardar UN SOLO Parquet en la raíz del dataset (sin subcarpetas de fecha)
                # Resultado: data/bronze/sismepre/<dataset>/<dataset>_raw.parquet
                parquet_path = self.storage.save_parquet_flat(
                    df=df_full,
                    source=source,
                    dataset=dataset_key,
                    filename=f"{dataset_key}_raw",
                )

                # Guardar JSON de metadata particionado por fecha
                meta = {
                    "source": source,
                    "dataset": dataset_key,
                    "resource_id": resource_id,
                    "rows": total_rows,
                    "columns": list(df_full.columns),
                    "pages_fetched": len(pages),
                    "extraction_ts": execution_dt.isoformat(),
                }
                self.storage.save_json_raw(
                    data=meta,
                    source=source,
                    dataset=dataset_key,
                    execution_dt=execution_dt,
                )

                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="success",
                    rows=total_rows,
                    files_written=[parquet_path],
                    extra={"pages": len(pages)},
                )
                logger.info(
                    "✓ %s completado | total_rows=%d | pages=%d",
                    dataset_key,
                    total_rows,
                    len(pages),
                )

            except Exception as exc:
                error_msg = traceback.format_exc()
                logger.error("✗ Error en %s: %s", dataset_key, exc)
                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="error",
                    error=error_msg,
                )

    # ------------------------------------------------------------------
    # RENAMU — ZIP del INEI
    # ------------------------------------------------------------------

    def ingest_renamu(self) -> None:
        """
        Descarga el ZIP del RENAMU, lo extrae, lee cada CSV/XLSX como
        DataFrame RAW y guarda Parquet RAW particionado por fecha.
        """
        source = "renamu"
        renamu_cfg = self.config["sources"]["renamu"]
        execution_dt = datetime.now()

        logger.info("=== Iniciando ingestión RENAMU ===")

        for dataset_key, dataset_info in renamu_cfg["datasets"].items():
            url = dataset_info["url"]
            file_type = dataset_info["type"]

            try:
                logger.info("Procesando: %s | url=%s", dataset_key, url)

                # Descargar binario
                content_bytes = self.renamu_client.download_zip_raw(url) \
                    if file_type == "zip" \
                    else self.renamu_client.download_binary(url)

                # Guardar binario RAW (ZIP o PDF original)
                ext = ".zip" if file_type == "zip" else ".pdf"
                original_filename = f"{dataset_key}{ext}"
                self.storage.save_binary_raw(
                    content=content_bytes,
                    source=source,
                    dataset=dataset_key,
                    original_filename=original_filename,
                    execution_dt=execution_dt,
                )

                # Solo procesar ZIPs a Parquet
                if file_type != "zip":
                    logger.info(
                        "Archivo %s guardado como binario RAW (tipo=%s). No se genera Parquet.",
                        dataset_key,
                        file_type,
                    )
                    self._record_audit(
                        source=source,
                        dataset=dataset_key,
                        status="success_binary",
                        extra={"type": file_type, "size_bytes": len(content_bytes)},
                    )
                    continue

                # Extraer ZIP en disco
                extract_path = Path(renamu_cfg["raw_extracted_path"]) / dataset_key
                extracted_files = self.renamu_client.extract_zip_to_disk(
                    zip_bytes=content_bytes,
                    extract_path=extract_path,
                )

                # Leer cada CSV/XLSX y guardar Parquet RAW
                total_rows = 0
                written_files = []

                for identifier, df in self.renamu_client.iter_extracted_dataframes(extracted_files):
                    safe_id = identifier.replace(" ", "_").replace("/", "_")[:80]
                    parquet_path = self.storage.save_parquet_raw(
                        df=df,
                        source=source,
                        dataset=f"{dataset_key}__{safe_id}",
                        execution_dt=execution_dt,
                    )
                    total_rows += len(df)
                    written_files.append(parquet_path)

                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="success",
                    rows=total_rows,
                    files_written=written_files,
                    extra={
                        "zip_files_extracted": len(extracted_files),
                        "parquets_generated": len(written_files),
                    },
                )
                logger.info(
                    "✓ %s completado | total_rows=%d | parquets=%d",
                    dataset_key,
                    total_rows,
                    len(written_files),
                )

            except Exception as exc:
                error_msg = traceback.format_exc()
                logger.error("✗ Error en %s: %s", dataset_key, exc)
                self._record_audit(
                    source=source,
                    dataset=dataset_key,
                    status="error",
                    error=error_msg,
                )

    # ------------------------------------------------------------------
    # Ejecución completa
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        """
        Ejecuta la ingestión RAW completa de todas las fuentes en secuencia.
        Un error en una fuente NO aborta las demás.
        """
        start_time = datetime.now()
        logger.info("════════════════════════════════════════")
        logger.info("  PIPELINE RAW — INICIO COMPLETO")
        logger.info("  %s", start_time.isoformat())
        logger.info("════════════════════════════════════════")

        self.ingest_siaf_csv()
        self.ingest_siaf_api()
        self.ingest_sismepre()
        self.ingest_renamu()

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        logger.info("════════════════════════════════════════")
        logger.info("  PIPELINE RAW — COMPLETADO")
        logger.info("  Duración: %.1f segundos", elapsed)
        logger.info("════════════════════════════════════════")