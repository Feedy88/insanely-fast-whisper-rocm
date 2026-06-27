"""API route definitions for the Insanely Fast Whisper API.

This module contains clean, focused route definitions that use dependency
injection for ASR pipeline instances and file handling.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from insanely_fast_whisper_rocm.api.dependencies import (
    get_asr_pipeline,
    get_file_handler,
)
from insanely_fast_whisper_rocm.api.responses import ResponseFormatter
from insanely_fast_whisper_rocm.core.asr_backend import exit_due_to_gpu_context_loss
from insanely_fast_whisper_rocm.core.errors import GpuContextLostError, OutOfMemoryError
from insanely_fast_whisper_rocm.core.integrations.stable_ts import stabilize_timestamps
from insanely_fast_whisper_rocm.core.orchestrator import create_orchestrator
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_DEMUCS,
    DEFAULT_STABILIZE,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
    RESPONSE_FORMAT_JSON,
    SUPPORTED_RESPONSE_FORMATS,
    FileHandler,
)
from insanely_fast_whisper_rocm.utils.constants import EXIT_ON_GPU_ERROR

logger = logging.getLogger(__name__)

router = APIRouter()

# Message returned to the client when the GPU/HIP context is lost and the
# service is restarting (or staying up, if restarts are disabled).
_GPU_CONTEXT_LOST_DETAIL = (
    "GPU context lost during inference; the service is restarting with a clean "
    "GPU context. Please retry the request shortly."
)


def _gpu_context_lost_response() -> JSONResponse:
    """Build the HTTP 503 response for an unrecoverable GPU context loss.

    A "HIP error: unspecified launch failure" poisons the GPU context for the
    whole process, so every subsequent request would otherwise fail with a 500.
    The pipeline and allocator caches were already dropped in
    ``asr_backend.process_audio``; here we return a 503 to the in-flight client
    and, unless restarts are disabled, attach a background task that force-exits
    the process *after* the response is flushed so systemd
    (``Restart=on-failure``) brings the service back with a clean context.

    This mirrors the Wyoming server's recovery so both services behave the same.

    Returns:
        A 503 JSON response, with a process-restart background task when
        ``WHISPER_EXIT_ON_GPU_ERROR`` is enabled.
    """
    if not EXIT_ON_GPU_ERROR:
        logger.warning(
            "WHISPER_EXIT_ON_GPU_ERROR is disabled; staying up, but further "
            "requests will keep failing until the process is restarted."
        )
        return JSONResponse(
            status_code=503, content={"detail": _GPU_CONTEXT_LOST_DETAIL}
        )

    logger.critical(
        "GPU context lost; returning 503 and scheduling a process restart for a "
        "clean GPU context."
    )
    return JSONResponse(
        status_code=503,
        content={"detail": _GPU_CONTEXT_LOST_DETAIL},
        background=BackgroundTask(exit_due_to_gpu_context_loss),
    )


@router.post(
    "/v1/audio/transcriptions",
    tags=["Transcription"],
    summary="Transcribe Audio",
    description="Convert speech in an audio file to text using the Whisper model",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    task: Literal["transcribe"] = Form("transcribe", description="ASR task type"),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(DEFAULT_DEMUCS, description="Enable Demucs noise reduction"),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> str | dict:
    """Transcribe speech in an audio file to text.

    This endpoint processes an audio file and returns its transcription using the
    specified Whisper model. It supports various configuration options including
    timestamp generation.

    Args:
        file: The audio file to transcribe (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json", "verbose_json",
            "text", "srt", or "vtt").
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        task: ASR task type (must be "transcribe")
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable Demucs noise reduction if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        asr_pipeline: Injected ASR pipeline instance
        file_handler: Injected file handler instance

    Returns:
        Union[str, dict]: Transcription result as plain text or JSON with metadata

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    logger.info("-" * 50)
    logger.info("Received transcription request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Task: %s", task)

    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)

    try:
        logger.info("Starting transcription process...")

        # Use orchestrator for transcription with OOM recovery
        orchestrator = create_orchestrator()

        # We need to construct a backend config.
        # Since we use dependency injection for asr_pipeline,
        # we can get the config from it.
        # However, the orchestrator handles pipeline acquisition
        # via borrow_pipeline.
        # We'll use the config from the injected pipeline as
        # the starting point.
        base_config = asr_pipeline.asr_backend.config

        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task=task,
                timestamp_type=timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for transcription: {str(oom)}",
            ) from oom
        except GpuContextLostError as gpu_err:
            logger.critical(
                "GPU context lost during transcription: %s", gpu_err, exc_info=True
            )
            return _gpu_context_lost_response()
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Optional stabilization (post-process) applied here for API
        if stabilize:
            try:
                result = stabilize_timestamps(
                    result, demucs=demucs, vad=vad, vad_threshold=vad_threshold
                )
            except Exception as stab_exc:  # noqa: BLE001
                logger.error("Stabilization failed: %s", stab_exc, exc_info=True)
        logger.info("Transcription completed successfully")

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")
        logger.debug("Transcription result: %s", result)

        # Format response according to requested response_format
        return ResponseFormatter.format_transcription(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)


@router.post(
    "/v1/audio/translations",
    tags=["Translation"],
    summary="Translate Audio",
    description="Translate speech in an audio file to English using the Whisper model",
    responses={
        200: {
            "description": "Successful translation",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_translation(
    file: UploadFile = File(..., description="The audio file to translate"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(DEFAULT_DEMUCS, description="Enable Demucs noise reduction"),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> str | dict:
    """Translate speech in an audio file to English.

    This endpoint processes an audio file in any supported language and translates
    the speech to English using the specified Whisper model. It supports various
    configuration options.

    Args:
        file: The audio file to translate (supported formats: mp3, wav, etc.)
        response_format: Desired response format ("json" or "text")
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable Demucs noise reduction if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        asr_pipeline: Injected ASR pipeline instance
        file_handler: Injected file handler instance

    Returns:
        Union[str, dict]: Translation result as plain text or JSON with metadata

    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    logger.info("-" * 50)
    logger.info("Received translation request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Response format: %s", response_format)

    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)

    try:
        logger.info("Starting translation process...")

        # Use orchestrator for translation with OOM recovery
        orchestrator = create_orchestrator()
        base_config = asr_pipeline.asr_backend.config

        try:
            result = orchestrator.run_transcription(
                audio_path=temp_filepath,
                backend_config=base_config,
                language=language,
                task="translate",
                timestamp_type=timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for translation: {str(oom)}",
            ) from oom
        except GpuContextLostError as gpu_err:
            logger.critical(
                "GPU context lost during translation: %s", gpu_err, exc_info=True
            )
            return _gpu_context_lost_response()
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Optional stabilization (post-process) applied here for API
        if stabilize:
            try:
                result = stabilize_timestamps(
                    result, demucs=demucs, vad=vad, vad_threshold=vad_threshold
                )
            except Exception as stab_exc:  # noqa: BLE001
                logger.error("Stabilization failed: %s", stab_exc, exc_info=True)
        logger.info("Translation completed successfully")
        logger.debug("Translation result: %s", result)

        # Validate response_format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported response_format")

        # Format response
        return ResponseFormatter.format_translation(result, response_format)

    finally:
        file_handler.cleanup(temp_filepath)
