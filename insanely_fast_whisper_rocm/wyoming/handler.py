"""Wyoming STT event handler.

Receives AudioStart/AudioChunk/AudioStop events from a Wyoming client (e.g.
Home Assistant), buffers the PCM audio into a temporary WAV file, runs
ROCm-accelerated Whisper transcription via the existing orchestrator, and
returns a Transcript event.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import wave
from typing import Any

from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from insanely_fast_whisper_rocm.core.asr_backend import (
    GPU_LOST_EXIT_CODE,
    HuggingFaceBackendConfig,
    exit_due_to_gpu_context_loss,
)
from insanely_fast_whisper_rocm.core.errors import GpuContextLostError
from insanely_fast_whisper_rocm.core.orchestrator import TranscriptionOrchestrator
from insanely_fast_whisper_rocm.utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MODEL,
    DEFAULT_PROGRESS_GROUP_SIZE,
    WYOMING_EXIT_ON_GPU_ERROR,
    WYOMING_LANGUAGE,
)

logger = logging.getLogger(__name__)

_TARGET_RATE = 16000
_TARGET_WIDTH = 2  # 16-bit
_TARGET_CHANNELS = 1  # mono

# Non-zero exit code used when bailing out on an unrecoverable GPU error, so the
# service supervisor treats it as a failure and restarts the process. Shared
# with the FastAPI service via the asr_backend module so they stay in sync.
_GPU_LOST_EXIT_CODE = GPU_LOST_EXIT_CODE


class WyomingEventHandler(AsyncEventHandler):
    """Handles Wyoming protocol events for one client connection."""

    def __init__(self, wyoming_info: Info, *args: Any, **kwargs: Any) -> None:
        """Initialise handler with server info.

        Args:
            wyoming_info: Pre-built Info object describing this server.
            *args: Forwarded to AsyncEventHandler.
            **kwargs: Forwarded to AsyncEventHandler.
        """
        super().__init__(*args, **kwargs)
        self._wyoming_info = wyoming_info
        self._language: str = WYOMING_LANGUAGE
        self._converter: AudioChunkConverter | None = None
        self._audio_buffer: list[bytes] = []
        self._wav_path: str | None = None

    async def handle_event(self, event: Event) -> bool:
        """Dispatch incoming Wyoming events.

        Args:
            event: Incoming Wyoming event.

        Returns:
            True to keep connection open, False to close after Transcript.
        """
        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info.event())
            logger.debug("Sent Info response to Wyoming client")

        elif Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                logger.debug("Language set to %s by client", self._language)

        elif AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self._converter = AudioChunkConverter(
                rate=_TARGET_RATE,
                width=_TARGET_WIDTH,
                channels=_TARGET_CHANNELS,
            )
            self._audio_buffer = []
            self._wav_path = None
            logger.debug(
                "AudioStart: rate=%s width=%s channels=%s",
                audio_start.rate,
                audio_start.width,
                audio_start.channels,
            )

        elif AudioChunk.is_type(event.type):
            if self._converter is None:
                # AudioChunk arrived before AudioStart — initialise with defaults
                logger.warning("AudioChunk before AudioStart; using default 16kHz mono")
                self._converter = AudioChunkConverter(
                    rate=_TARGET_RATE,
                    width=_TARGET_WIDTH,
                    channels=_TARGET_CHANNELS,
                )
                self._audio_buffer = []
            raw_chunk = AudioChunk.from_event(event)
            normalised = self._converter.convert(raw_chunk)
            self._audio_buffer.append(normalised.audio)

        elif AudioStop.is_type(event.type):
            try:
                text = await asyncio.to_thread(self._transcribe_buffered_audio)
            except GpuContextLostError:
                await self._handle_gpu_context_lost()
                return False
            logger.info("Transcription result: %r", text)
            await self.write_event(Transcript(text=text).event())
            return False  # close this connection after sending transcript

        return True

    async def _handle_gpu_context_lost(self) -> None:
        """Respond to an unrecoverable GPU error and restart the process.

        The HIP/GPU context is poisoned for the whole process, so no further
        request can succeed. Send an empty transcript so the client does not
        hang, then exit (unless disabled) so the supervisor restarts us with a
        clean context.
        """
        logger.critical(
            "GPU context lost (unrecoverable HIP error). Sending empty transcript "
            "and exiting so the service restarts with a clean GPU context."
        )
        # Best-effort empty response; the connection/context may already be gone.
        with contextlib.suppress(Exception):
            await self.write_event(Transcript(text="").event())

        if not WYOMING_EXIT_ON_GPU_ERROR:
            logger.warning(
                "WYOMING_EXIT_ON_GPU_ERROR is disabled; staying up, but further "
                "transcriptions will keep failing until the process is restarted."
            )
            return

        # Shared recovery: free caches and force-exit so the supervisor restarts
        # us with a clean context. The temp WAV is already cleaned up in the
        # worker's finally block before this error propagated. This call does
        # not return.
        exit_due_to_gpu_context_loss(_GPU_LOST_EXIT_CODE)

    def _transcribe_buffered_audio(self) -> str:
        """Write buffered PCM to a temp WAV, run Whisper, return text.

        Returns:
            Transcribed text string (empty string on error).
        """
        if not self._audio_buffer:
            logger.warning("No audio received — returning empty transcript")
            return ""

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(tmp_fd, "wb") as raw_fd:
                wav_writer = wave.open(raw_fd, "wb")
                wav_writer.setnchannels(_TARGET_CHANNELS)
                wav_writer.setsampwidth(_TARGET_WIDTH)
                wav_writer.setframerate(_TARGET_RATE)
                for chunk in self._audio_buffer:
                    wav_writer.writeframes(chunk)
                wav_writer.close()

            config = HuggingFaceBackendConfig(
                model_name=DEFAULT_MODEL,
                device=DEFAULT_DEVICE,
                dtype=DEFAULT_DTYPE,
                batch_size=DEFAULT_BATCH_SIZE,
                chunk_length=DEFAULT_CHUNK_LENGTH,
                progress_group_size=DEFAULT_PROGRESS_GROUP_SIZE,
            )
            orchestrator = TranscriptionOrchestrator()
            result = orchestrator.run_transcription(
                audio_path=tmp_path,
                backend_config=config,
                task="transcribe",
                language=self._language if self._language != "None" else None,
                timestamp_type=False,
                save_transcriptions=False,
            )
            return result.get("text", "")

        except GpuContextLostError:
            # Unrecoverable: let it propagate so handle_event can restart the
            # process instead of silently returning an empty transcript forever.
            raise

        except Exception:
            logger.exception("Transcription failed in Wyoming handler")
            return ""

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
