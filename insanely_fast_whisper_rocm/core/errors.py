"""Custom exception classes for the ASR pipeline."""

from __future__ import annotations


class TranscriptionError(Exception):
    """Custom exception raised when transcription fails."""


class OutOfMemoryError(TranscriptionError):
    """Base class for Out of Memory errors."""

    def __init__(
        self,
        message: str,
        device: str | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize the OutOfMemoryError.

        Args:
            message: Error message.
            device: Optional device identifier.
            config: Optional configuration dictionary.
        """
        super().__init__(message)
        self.device = device
        self.config = config


class ModelLoadingOOMError(OutOfMemoryError):
    """Raised when model initialization fails due to OOM."""


class InferenceOOMError(OutOfMemoryError):
    """Raised when audio processing fails due to OOM."""


class GpuContextLostError(TranscriptionError):
    """Raised when the GPU/HIP context is irrecoverably lost during inference.

    On ROCm a "HIP error: unspecified launch failure" poisons the device context
    for the entire process; no in-process recovery is possible (even
    ``torch.cuda.synchronize()`` keeps failing). A supervised service should exit
    on this error so the process is restarted with a clean context.
    """


class TranscriptionCancelledError(TranscriptionError):
    """Raised when transcription is cancelled by the caller."""


class DeviceNotFoundError(TranscriptionError):
    """Custom exception raised when a requested compute device is not available."""
