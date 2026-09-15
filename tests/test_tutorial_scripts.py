"""The tutorial scripts run headless from start to finish. They read the
real datasets and solve at the degrees a reader would use, so they are
slow and excluded from the default run."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).parent.parent / "tutorials" / "scripts").glob("*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_tutorial_script_runs(script):
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, script.name],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
