"""Handles initial environment setup, .env file loading, and debug flag detection.

This module is responsible for:
- Locating project root and user-specific .env files.
- Performing an initial load of .env files to determine if debug output is enabled.
- Providing utility functions and constants for conditional debug printing.
- Exposing paths and existence flags for .env files to be used by constants.py
  for the main .env loading sequence.

Environment variable precedence (highest → lowest):
  1. Shell environment variables  (e.g. ``WHISPER_MODEL=… python -m …``)
  2. User config  (~/.config/insanely-fast-whisper-rocm/.env)
  3. Project .env  (repo root .env — acts as default/template only)

All .env files are loaded with ``override=False`` so that values already present
in the process environment are never replaced.  To override the model at runtime,
pass the variable before the command::

    WHISPER_MODEL=openai/whisper-small insanely-fast-whisper-wyoming
    WHISPER_MODEL=openai/whisper-small insanely-fast-whisper-rocm
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Use proper Python logging for environment loading
logger = logging.getLogger(__name__)

# Determine Project Root based on this file's location
# Assumes this file is in insanely_fast_whisper_rocm/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_ENV_FILE = PROJECT_ROOT / ".env"

USER_CONFIG_DIR = Path.home() / ".config" / "insanely-fast-whisper-rocm"
USER_ENV_FILE = USER_CONFIG_DIR / ".env"

_cli_debug_mode = "--debug" in sys.argv

# Temporarily load .env files to check LOG_LEVEL for initial debug print decision.
# override=False: shell env vars already set in the process take precedence.
# Load order: project first (defaults), then user config (can override project
# defaults but not shell vars because both use override=False and the user .env
# is loaded second — values already set by the project load won't be replaced,
# so we load user first in the pre-check to mirror constants.py load order).

_project_root_env_exists_temp = PROJECT_ROOT_ENV_FILE.exists()
if not USER_CONFIG_DIR.exists():
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_user_env_exists_temp = USER_ENV_FILE.exists()

# Pre-load: user config first so it wins over project defaults (both override=False,
# so shell vars win over both).
if _user_env_exists_temp:
    load_dotenv(USER_ENV_FILE, override=False)
if _project_root_env_exists_temp:
    load_dotenv(PROJECT_ROOT_ENV_FILE, override=False)

_env_log_level_temp = os.getenv("LOG_LEVEL", "").upper()
_env_debug_mode_temp = _env_log_level_temp == "DEBUG"

SHOW_DEBUG_PRINTS = _cli_debug_mode or _env_debug_mode_temp

# Configure logging early if debug mode is detected
if SHOW_DEBUG_PRINTS:
    logging.basicConfig(
        level=logging.INFO,  # Keep root logger at INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Only enable DEBUG for our application's loggers
    logging.getLogger("insanely_fast_whisper_rocm").setLevel(logging.DEBUG)
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    # MIOpen warnings are printed directly to stderr, not through Python logging,
    # so we can't suppress them here. They're harmless workspace allocation attempts.


def debug_print(message: str) -> None:
    """Log environment loading messages at DEBUG level if enabled.

    Args:
        message: The message to log.
    """
    if SHOW_DEBUG_PRINTS:
        logger.debug(message)


# Main load: same order as the pre-load above (user first, project second).
# override=False ensures shell env vars set before process start are never
# replaced.  The project .env provides fallback defaults only.
if _user_env_exists_temp:
    debug_print(f"Loading user .env: {USER_ENV_FILE}")
    load_dotenv(USER_ENV_FILE, override=False)
if _project_root_env_exists_temp:
    debug_print(f"Loading project .env: {PROJECT_ROOT_ENV_FILE}")
    load_dotenv(PROJECT_ROOT_ENV_FILE, override=False)
else:
    debug_print(f"No project .env found at: {PROJECT_ROOT_ENV_FILE}")

# Expose pre-checked existence and paths for constants.py to use for the main
# load. PROJECT_ROOT_ENV_EXISTS is still useful if constants.py wants to know
# if it was loaded.
PROJECT_ROOT_ENV_EXISTS = _project_root_env_exists_temp
USER_ENV_EXISTS = _user_env_exists_temp
