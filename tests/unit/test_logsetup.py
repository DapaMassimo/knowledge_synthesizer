import logging

import pytest

from knowledge_synthesizer.entrypoints.logsetup import clear_logs, configure_logging, get_logs

pytestmark = pytest.mark.unit


def test_configured_logger_captures_messages_into_the_buffer() -> None:
    configure_logging("INFO")
    clear_logs()

    logging.getLogger("knowledge_synthesizer.application.test").info("hello %d", 42)

    logs = get_logs()
    assert any("hello 42" in line for line in logs)


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    handler_count = len(logging.getLogger("knowledge_synthesizer").handlers)
    configure_logging("DEBUG")  # second call must not add duplicate handlers
    assert len(logging.getLogger("knowledge_synthesizer").handlers) == handler_count


def test_clear_logs_empties_the_buffer() -> None:
    configure_logging("INFO")
    logging.getLogger("knowledge_synthesizer").info("something")
    clear_logs()
    assert get_logs() == []
