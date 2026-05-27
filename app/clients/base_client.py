"""
base_client.py — Cliente HTTP base con reintentos, backoff y logging.

Todos los clientes de fuentes heredan de BaseHTTPClient para compartir:
  - Session de requests con headers comunes.
  - Lógica de reintentos con backoff exponencial.
  - Timeout configurable.
  - Logging centralizado de cada request.
"""

import time
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests import Response, Session
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseHTTPClient:
    """
    Cliente HTTP base con reintentos y backoff exponencial.

    Parámetros configurables desde config.yaml / variables de entorno.
    """

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        backoff_factor: int = 2,
        headers: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        timeout : int
            Segundos de espera por respuesta HTTP.
        max_retries : int
            Número máximo de reintentos ante fallos transitorios.
        backoff_factor : int
            Segundos base de espera entre reintentos (se multiplica por intento).
        headers : Optional[dict]
            Headers HTTP adicionales para todas las peticiones.
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        self.session: Session = requests.Session()
        default_headers = {
            "User-Agent": "DataEngineeringPipeline/1.0 (academic research)",
            "Accept": "application/json, text/csv, */*",
        }
        if headers:
            default_headers.update(headers)
        self.session.headers.update(default_headers)

        logger.debug(
            "BaseHTTPClient inicializado | timeout=%ds | max_retries=%d",
            self.timeout,
            self.max_retries,
        )

    # ------------------------------------------------------------------
    # GET genérico con reintentos
    # ------------------------------------------------------------------

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> Response:
        """
        Ejecuta un GET HTTP con reintentos y backoff exponencial.

        Parameters
        ----------
        url : str
            URL completa del recurso.
        params : Optional[dict]
            Query parameters del request.
        stream : bool
            Si True, descarga el contenido en modo streaming (para archivos grandes).

        Returns
        -------
        Response
            Objeto Response de requests con status 2xx.

        Raises
        ------
        RequestException
            Si se agotaron todos los reintentos sin respuesta exitosa.
        """
        attempt = 0
        last_exception: Optional[Exception] = None

        while attempt <= self.max_retries:
            try:
                logger.debug(
                    "GET | url=%s | params=%s | attempt=%d/%d",
                    url,
                    params,
                    attempt + 1,
                    self.max_retries + 1,
                )
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    stream=stream,
                )
                response.raise_for_status()
                logger.debug(
                    "GET OK | url=%s | status=%d | size=%s bytes",
                    url,
                    response.status_code,
                    response.headers.get("Content-Length", "unknown"),
                )
                return response

            except (Timeout, ConnectionError) as exc:
                last_exception = exc
                wait = self.backoff_factor ** attempt
                logger.warning(
                    "Error de red (intento %d/%d): %s — esperando %ds...",
                    attempt + 1,
                    self.max_retries + 1,
                    str(exc),
                    wait,
                )
                time.sleep(wait)
                attempt += 1

            except HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                # No reintentar errores 4xx (son errores del cliente, no transitorios)
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    logger.error(
                        "Error HTTP %s (no reintentable) | url=%s",
                        status,
                        url,
                    )
                    raise
                last_exception = exc
                wait = self.backoff_factor ** attempt
                logger.warning(
                    "Error HTTP %s (intento %d/%d) — esperando %ds...",
                    status,
                    attempt + 1,
                    self.max_retries + 1,
                    wait,
                )
                time.sleep(wait)
                attempt += 1

        raise RequestException(
            f"Se agotaron {self.max_retries + 1} intentos para {url}. "
            f"Último error: {last_exception}"
        ) from last_exception

    # ------------------------------------------------------------------
    # Descarga de contenido binario (streaming para archivos grandes)
    # ------------------------------------------------------------------

    def download_binary(self, url: str, chunk_size: int = 8192) -> bytes:
        """
        Descarga el contenido binario completo de una URL (ZIP, CSV grande, PDF).

        Parameters
        ----------
        url : str
            URL del archivo a descargar.
        chunk_size : int
            Tamaño de cada chunk en bytes durante la descarga streaming.

        Returns
        -------
        bytes
            Contenido completo del archivo.
        """
        response = self.get(url, stream=True)
        chunks = []
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                chunks.append(chunk)
        content = b"".join(chunks)
        logger.info(
            "Descarga completada | url=%s | total=%d bytes",
            url,
            len(content),
        )
        return content

    def close(self) -> None:
        """Cierra la sesión HTTP."""
        self.session.close()
        logger.debug("Sesión HTTP cerrada.")

    def __enter__(self) -> "BaseHTTPClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
