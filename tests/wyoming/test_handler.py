"""Tests for the Wyoming event handler's GPU-context-lost handling.

When inference raises GpuContextLostError (an unrecoverable HIP failure), the
handler must send an empty transcript and exit the process so a supervisor
restarts it with a clean context — unless that behaviour is disabled.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.core.errors import GpuContextLostError
from insanely_fast_whisper_rocm.wyoming.handler import (
    _GPU_LOST_EXIT_CODE,
    WyomingEventHandler,
)


def _make_handler() -> WyomingEventHandler:
    """Build a handler with mock transport and a capturing write_event.

    Returns:
        A WyomingEventHandler whose ``write_event`` is an AsyncMock.
    """
    handler = WyomingEventHandler(
        wyoming_info=MagicMock(),
        reader=MagicMock(),
        writer=MagicMock(),
    )
    handler.write_event = AsyncMock()  # type: ignore[method-assign]
    return handler


def test_handle_gpu_context_lost_exits_when_enabled() -> None:
    """Empty transcript is sent and the process exits with the failure code."""
    handler = _make_handler()

    with patch(
        "insanely_fast_whisper_rocm.wyoming.handler.WYOMING_EXIT_ON_GPU_ERROR",
        True,
    ):
        with patch(
            "insanely_fast_whisper_rocm.wyoming.handler.os._exit"
        ) as mock_exit:
            asyncio.run(handler._handle_gpu_context_lost())

    mock_exit.assert_called_once_with(_GPU_LOST_EXIT_CODE)
    handler.write_event.assert_awaited_once()
    sent_event = handler.write_event.await_args.args[0]
    assert sent_event.type == "transcript"
    assert sent_event.data.get("text") == ""


def test_handle_gpu_context_lost_stays_up_when_disabled() -> None:
    """With the toggle off, the process keeps running (no exit)."""
    handler = _make_handler()

    with patch(
        "insanely_fast_whisper_rocm.wyoming.handler.WYOMING_EXIT_ON_GPU_ERROR",
        False,
    ):
        with patch(
            "insanely_fast_whisper_rocm.wyoming.handler.os._exit"
        ) as mock_exit:
            asyncio.run(handler._handle_gpu_context_lost())

    mock_exit.assert_not_called()
    handler.write_event.assert_awaited_once()


def test_audio_stop_routes_gpu_error_to_handler() -> None:
    """AudioStop dispatch catches GpuContextLostError and triggers the exit path."""
    handler = _make_handler()

    audio_stop_event = MagicMock()
    with patch(
        "insanely_fast_whisper_rocm.wyoming.handler.AudioStop"
    ) as mock_audio_stop:
        # Only AudioStop matches; all other is_type checks are False.
        mock_audio_stop.is_type.return_value = True
        for other in ("Describe", "Transcribe", "AudioStart", "AudioChunk"):
            patcher = patch(
                f"insanely_fast_whisper_rocm.wyoming.handler.{other}"
            )
            mock_other = patcher.start()
            mock_other.is_type.return_value = False

        with patch.object(
            handler,
            "_transcribe_buffered_audio",
            side_effect=GpuContextLostError("GPU context lost during inference"),
        ):
            with patch.object(
                handler, "_handle_gpu_context_lost", new=AsyncMock()
            ) as mock_handle:
                keep_open = asyncio.run(handler.handle_event(audio_stop_event))

    mock_handle.assert_awaited_once()
    assert keep_open is False
