"""Application logging: a bounded in-memory buffer the UI can render, plus optional console.

Services log via ``logging.getLogger(__name__)`` (stdlib only); the entrypoints call
``configure_logging`` once to attach handlers under the ``knowledge_synthesizer`` logger.
"""

from __future__ import annotations

import logging
from collections import deque

_ROOT = "knowledge_synthesizer"
_BUFFER: deque[str] = deque(maxlen=1000)
_CONFIGURED = False


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _BUFFER.append(self.format(record))


def configure_logging(level: str = "INFO", *, to_console: bool = False) -> bool:
    """Attach the buffer (and optionally a console) handler once. Returns True if newly set up."""
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level.upper())
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
