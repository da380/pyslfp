"""
Tests for run-time thread control.
"""

import ducc0
from threadpoolctl import threadpool_info
import pyshtools.backends.ducc0_wrapper as ducc_wrapper

from pyslfp.parallel import physical_core_count, set_num_threads


def test_physical_core_count_is_positive():
    assert physical_core_count() >= 1


def test_set_num_threads_applies_to_transforms_and_blas():
    applied = set_num_threads(2)
    assert applied == 2
    assert ducc0.misc.thread_pool_size() == 2
    assert ducc_wrapper.nthreads == 2
    for info in threadpool_info():
        assert info["num_threads"] <= 2

    # Restore single-threaded operation for the rest of the test session
    assert set_num_threads(1) == 1
    assert ducc0.misc.thread_pool_size() == 1


def test_set_num_threads_default_is_physical_cores():
    applied = set_num_threads(None)
    assert applied == physical_core_count()
    set_num_threads(1)
