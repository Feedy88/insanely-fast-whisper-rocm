"""Tests for the shared GPU-context-loss recovery helpers in asr_backend.

These primitives are the single source of truth for HIP/GPU context-loss
recovery, shared by the Wyoming server and the FastAPI service so the two
cannot drift apart. They cover:

* ``free_gpu_caches`` — the in-process ``empty_cache`` step.
* ``exit_due_to_gpu_context_loss`` — free caches then force-exit for a
  supervised restart.
* the HIP-error branch of ``process_audio`` freeing caches before raising
  ``GpuContextLostError``.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.core import asr_backend
from insanely_fast_whisper_rocm.core.asr_backend import (
    GPU_LOST_EXIT_CODE,
    HuggingFaceBackend,
    HuggingFaceBackendConfig,
    exit_due_to_gpu_context_loss,
    free_gpu_caches,
)
from insanely_fast_whisper_rocm.core.errors import GpuContextLostError


def test_free_gpu_caches_empties_cuda_when_available() -> None:
    """``empty_cache`` is invoked when CUDA/ROCm is available."""
    with patch.object(asr_backend.torch.cuda, "is_available", return_value=True):
        with patch.object(asr_backend.torch.cuda, "empty_cache") as mock_empty:
            free_gpu_caches()
    mock_empty.assert_called_once()


def test_free_gpu_caches_noop_when_cuda_unavailable() -> None:
    """No cache call is made (and no error raised) when CUDA is unavailable."""
    with patch.object(asr_backend.torch.cuda, "is_available", return_value=False):
        with patch.object(asr_backend.torch.cuda, "empty_cache") as mock_empty:
            free_gpu_caches()
    mock_empty.assert_not_called()


def test_free_gpu_caches_swallows_errors() -> None:
    """A failing ``empty_cache`` is suppressed (cleanup is best-effort)."""
    with patch.object(asr_backend.torch.cuda, "is_available", return_value=True):
        with patch.object(
            asr_backend.torch.cuda,
            "empty_cache",
            side_effect=RuntimeError("HIP error: unspecified launch failure"),
        ):
            # Must not raise.
            free_gpu_caches()


def test_exit_due_to_gpu_context_loss_frees_then_exits() -> None:
    """The helper frees caches and force-exits with the given code."""
    with patch.object(asr_backend, "free_gpu_caches") as mock_free:
        with patch.object(asr_backend.os, "_exit") as mock_exit:
            exit_due_to_gpu_context_loss()

    mock_free.assert_called_once()
    mock_exit.assert_called_once_with(GPU_LOST_EXIT_CODE)


def test_exit_due_to_gpu_context_loss_honours_custom_code() -> None:
    """A caller-provided exit code is forwarded to ``os._exit``."""
    with patch.object(asr_backend, "free_gpu_caches"):
        with patch.object(asr_backend.os, "_exit") as mock_exit:
            exit_due_to_gpu_context_loss(1)
    mock_exit.assert_called_once_with(1)


def test_gpu_lost_exit_code_is_nonzero() -> None:
    """A non-zero code is required for systemd ``Restart=on-failure``."""
    assert GPU_LOST_EXIT_CODE != 0


def test_hip_error_frees_caches_before_raising(tmp_path: pathlib.Path) -> None:
    """The HIP branch drops the pipeline and frees caches, then raises."""
    config = HuggingFaceBackendConfig(
        model_name="openai/whisper-tiny",
        device="cpu",
        dtype="float32",
        batch_size=1,
        chunk_length=30,
        progress_group_size=4,
    )
    backend = HuggingFaceBackend(config)

    class DummyModel:
        generation_config = types.SimpleNamespace(no_timestamps_token_id=50363)
        config = types.SimpleNamespace(lang_to_id=None, task_to_id=None)

    class DummyPipe:
        def __init__(self) -> None:
            self.model = DummyModel()

        def __call__(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("HIP error: unspecified launch failure")

    backend.asr_pipe = DummyPipe()

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"0")

    with patch.object(asr_backend, "free_gpu_caches") as mock_free:
        with pytest.raises(GpuContextLostError, match="GPU context lost"):
            backend.process_audio(
                str(audio_file),
                language=None,
                task="transcribe",
                return_timestamps_value=False,
            )

    # Pipeline dropped and caches freed as part of the in-process recovery.
    assert backend.asr_pipe is None
    mock_free.assert_called_once()
