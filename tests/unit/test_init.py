"""Tests for tacotoolbox __init__.py"""

import logging

import pytest

import tacotoolbox
from tacotoolbox import verbose


class TestVerbose:

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        logger = logging.getLogger("tacotoolbox")
        original_level = logger.level
        original_handlers = logger.handlers[:]
        yield
        logger.handlers = original_handlers
        logger.setLevel(original_level)

    def test_true_sets_info_level(self):
        verbose(True)
        assert logging.getLogger("tacotoolbox").level == logging.INFO

    def test_info_string_sets_info_level(self):
        verbose("info")
        assert logging.getLogger("tacotoolbox").level == logging.INFO

    def test_debug_string_sets_debug_level(self):
        verbose("debug")
        assert logging.getLogger("tacotoolbox").level == logging.DEBUG

    def test_false_disables_logging(self):
        verbose(False)
        assert logging.getLogger("tacotoolbox").level > logging.CRITICAL

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid verbose level"):
            verbose("invalid")


class TestVersion:

    def test_version_is_string(self):
        assert isinstance(tacotoolbox.__version__, str)

    def test_version_not_fallback(self):
        # Package is installed, should not be fallback
        assert tacotoolbox.__version__ != "0.0.0"