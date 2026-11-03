"""The package exposes its version."""

import re

import pyslfp


def test_the_version_is_exposed():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", pyslfp.__version__)
