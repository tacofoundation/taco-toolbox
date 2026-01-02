"""Tests for _logging.py"""

import logging

import pytest

from tacotoolbox._logging import (
    disable_logging,
    get_logger,
    setup_basic_logging,
)


class TestGetLogger:

    def test_prefixes_bare_module_name(self):
        logger = get_logger("mymodule")
        assert logger.name == "tacotoolbox.mymodule"

    def test_main_becomes_tacotoolbox(self):
        logger = get_logger("__main__")
        assert logger.name == "tacotoolbox"

    def test_already_prefixed_unchanged(self):
        logger = get_logger("tacotoolbox.create")
        assert logger.name == "tacotoolbox.create"


class TestSetupBasicLogging:

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Clean tacotoolbox logger before/after each test."""
        logger = logging.getLogger("tacotoolbox")
        original_level = logger.level
        original_handlers = logger.handlers[:]
        original_propagate = logger.propagate
        yield
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    def test_sets_level(self):
        setup_basic_logging(level=logging.WARNING)
        logger = logging.getLogger("tacotoolbox")
        assert logger.level == logging.WARNING

    def test_does_not_duplicate_handlers(self):
        setup_basic_logging()
        initial_count = len(logging.getLogger("tacotoolbox").handlers)
        setup_basic_logging()
        setup_basic_logging()
        assert len(logging.getLogger("tacotoolbox").handlers) == initial_count

    def test_disables_propagation(self):
        setup_basic_logging()
        assert logging.getLogger("tacotoolbox").propagate is False


class TestDisableLogging:

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        logger = logging.getLogger("tacotoolbox")
        original_level = logger.level
        yield
        logger.setLevel(original_level)

    def test_level_above_critical(self):
        disable_logging()
        logger = logging.getLogger("tacotoolbox")
        assert logger.level > logging.CRITICAL