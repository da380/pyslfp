"""Fixtures shared by the viscoelastic tests."""

import pytest

from .body import homogeneous_setting


@pytest.fixture(scope="module")
def setting():
    """The homogeneous test body of `body.py`: (material, earth, state, load,
    tau_nd)."""
    return homogeneous_setting()
