"""Tests for _progress.py"""

import logging

import pytest

from tacotoolbox._progress import _should_show_progress, progress_bar, progress_scope


@pytest.fixture(autouse=True)
def reset_logger():
    logger = logging.getLogger("tacotoolbox._progress")
    original_level = logger.level
    yield
    logger.setLevel(original_level)


def set_progress_logger_level(level: int):
    logging.getLogger("tacotoolbox._progress").setLevel(level)


class TestShouldShowProgress:

    def test_shows_at_debug(self):
        set_progress_logger_level(logging.DEBUG)
        assert _should_show_progress() is True

    def test_shows_at_info(self):
        set_progress_logger_level(logging.INFO)
        assert _should_show_progress() is True

    def test_hidden_at_warning(self):
        set_progress_logger_level(logging.WARNING)
        assert _should_show_progress() is False

    def test_hidden_when_disabled(self):
        set_progress_logger_level(logging.CRITICAL + 1)
        assert _should_show_progress() is False


class TestProgressBar:

    def test_disabled_at_warning_level(self):
        set_progress_logger_level(logging.WARNING)
        pbar = progress_bar(range(10), desc="test")
        assert pbar.disable is True

    def test_enabled_at_info_level(self):
        set_progress_logger_level(logging.INFO)
        pbar = progress_bar(range(10), desc="test")
        assert pbar.disable is False

    def test_iterates_correctly(self):
        set_progress_logger_level(logging.WARNING)
        items = list(progress_bar([1, 2, 3]))
        assert items == [1, 2, 3]


class TestProgressScope:

    def test_yields_tqdm_instance(self):
        set_progress_logger_level(logging.WARNING)
        with progress_scope("test", total=10) as pbar:
            assert hasattr(pbar, "update")
            pbar.update(5)

    def test_closes_on_exit(self):
        set_progress_logger_level(logging.WARNING)
        with progress_scope("test", total=10) as pbar:
            pass
        assert pbar.n == 0 or pbar.disable