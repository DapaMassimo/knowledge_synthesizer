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
# Third-party libraries that spam warnings (e.g. transformers' lazy `__path__` access).
_NOISY = ("transformers", "torch", "httpx", "httpcore")


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _BUFFER.append(self.format(record))


def _quiet_noisy_libraries() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_logging(level: str = "INFO", *, to_console: bool = False) -> bool:
    """Attach the buffer (and optionally a console) handler once. Returns True if newly set up."""
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level.upper())
    _quiet_noisy_libraries()
    if _CONFIGURED:
        return False

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
    return True


def get_logs() -> list[str]:
    return list(_BUFFER)


def clear_logs() -> None:
    _BUFFER.clear()
