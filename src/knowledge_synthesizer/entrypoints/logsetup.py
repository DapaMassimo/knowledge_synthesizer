"""Application logging: a bounded in-memory buffer the UI can render, plus optional console.

Services log via ``logging.getLogger(__name__)`` (stdlib only); the entrypoints call
``configure_logging`` once to attach handlers under the ``knowledge_synthesizer`` logger.
"""

from __future__ import annotations

import logging
import os
from collections import deque

_ROOT = "knowledge_synthesizer"
_BUFFER: deque[str] = deque(maxlen=1000)
_CONFIGURED = False
# Third-party libraries that spam warnings/info (transformers' lazy `__path__` access,
# RapidOCR per-image info, HF Hub rate-limit notices, ...).
_NOISY = ("transformers", "torch", "httpx", "httpcore", "RapidOCR", "huggingface_hub")


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _BUFFER.append(self.format(record))


def _quiet_noisy_libraries() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.ERROR)


def _ensure_working_ssl() -> None:
    """Work around python-build-standalone OpenSSL failing on a mismatched system config."""
    import ssl

    try:
        ssl.create_default_context()
    except ssl.SSLError:
        os.environ["OPENSSL_CONF"] = os.devnull
        logging.getLogger(_ROOT).warning(
            "System OpenSSL config failed to initialize; set OPENSSL_CONF=%s to recover.",
            os.devnull,
        )


def configure_logging(level: str = "INFO", *, to_console: bool = False) -> bool:
    """Attach the buffer (and optionally a console) handler once. Returns True if newly set up."""
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level.upper())
    if _CONFIGURED:
        return False

    _quiet_noisy_libraries()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s", "%H:%M:%S")
    buffer_handler = _BufferHandler()
    buffer_handler.setFormatter(formatter)
    logger.addHandler(buffer_handler)
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    _CONFIGURED = True
    _ensure_working_ssl()  # after handlers so the warning lands in the buffer
    return True


def get_logs() -> list[str]:
    return list(_BUFFER)


def clear_logs() -> None:
    _BUFFER.clear()
