"""Tests for distributed utilities."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDistributedHelpers:
    def test_rank_without_init(self):
        from distributed import get_rank
        assert get_rank() == 0

    def test_world_size_without_init(self):
        from distributed import get_world_size
        assert get_world_size() == 1

    def test_is_main_process_without_init(self):
        from distributed import is_main_process
        assert is_main_process() is True

    def test_current_os(self):
        from distributed import CURRENT_OS
        assert CURRENT_OS in ('Windows', 'Linux', 'Darwin')

    def test_cleanup_without_init(self):
        from distributed import cleanup_ddp
        # Should not raise even if not initialized
        cleanup_ddp()
