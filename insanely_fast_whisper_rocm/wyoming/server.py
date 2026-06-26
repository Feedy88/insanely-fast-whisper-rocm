"""Wyoming TCP server setup and capability info builder."""

from __future__ import annotations

import asyncio
import logging
from asyncio import StreamReader, StreamWriter

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncTcpServer

from insanely_fast_whisper_rocm.utils.constants import DEFAULT_MODEL, WYOMING_HOST, WYOMING_PORT
from insanely_fast_whisper_rocm.wyoming.handler import WyomingEventHandler

logger = logging.getLogger(__name__)

_SUPPORTED_LANGUAGES = [
    "af", "ar", "az", "be", "bg", "bs", "ca", "cs", "cy", "da", "de",
    "el", "en", "es", "et", "fa", "fi", "fr", "gl", "he", "hi", "hr",
    "hu", "hy", "id", "is", "it", "ja", "kk", "kn", "ko", "lt", "lv",
    "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "pl", "pt",
    "ro", "ru", "sk", "sl", "sr", "sv", "sw", "ta", "te", "th", "tl",
    "tr", "uk", "ur", "uz", "vi", "zh",
]


def build_info() -> Info:
    """Build Wyoming Info describing this server's ASR capabilities.

    Returns:
        Info object with the active Whisper model and supported languages.
    """
    return Info(
        asr=[
            AsrProgram(
                name="insanely-fast-whisper-rocm",
                description="ROCm-accelerated Whisper STT",
                attribution=Attribution(
                    name="insanely-fast-whisper-rocm",
                    url="https://github.com/beecave-homelab/insanely-fast-whisper-rocm",
                ),
                installed=True,
                version=None,
                models=[
                    AsrModel(
                        name=DEFAULT_MODEL,
                        description=DEFAULT_MODEL,
                        attribution=Attribution(
                            name="HuggingFace Transformers",
                            url="https://huggingface.co/" + DEFAULT_MODEL,
                        ),
                        installed=True,
                        version=None,
                        languages=_SUPPORTED_LANGUAGES,
                    )
                ],
            )
        ]
    )


async def run_wyoming_server(host: str = WYOMING_HOST, port: int = WYOMING_PORT) -> None:
    """Start the Wyoming TCP server and handle connections until cancelled.

    Args:
        host: Bind host address.
        port: TCP port to listen on.
    """
    wyoming_info = build_info()

    def handler_factory(reader: StreamReader, writer: StreamWriter) -> WyomingEventHandler:
        return WyomingEventHandler(
            wyoming_info=wyoming_info,
            reader=reader,
            writer=writer,
        )

    server = AsyncTcpServer(host=host, port=port)
    logger.info("Wyoming STT server listening on %s:%d", host, port)
    await server.run(handler_factory)
