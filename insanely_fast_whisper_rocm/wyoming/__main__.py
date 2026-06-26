"""Entry point for the Wyoming STT server.

Usage:
    python -m insanely_fast_whisper_rocm.wyoming [--host HOST] [--port PORT]
    insanely-fast-whisper-wyoming [--host HOST] [--port PORT]
"""

from __future__ import annotations

import asyncio
import logging

import click

from insanely_fast_whisper_rocm.utils.constants import WYOMING_HOST, WYOMING_PORT
from insanely_fast_whisper_rocm.wyoming.server import run_wyoming_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default=WYOMING_HOST, show_default=True, help="Bind host")
@click.option("--port", default=WYOMING_PORT, show_default=True, type=int, help="TCP port")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging")
def main(host: str, port: int, verbose: bool) -> None:
    """Start the Wyoming STT server for Home Assistant integration."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(run_wyoming_server(host=host, port=port))


if __name__ == "__main__":
    main()
