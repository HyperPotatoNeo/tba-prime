from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "src/zeroband/eval.py", "@configs/eval/debug.toml"]


@pytest.fixture(scope="module")
def process(run_process: Callable[[Command, Environment], ProcessResult]) -> ProcessResult:
    return run_process(CMD, {})


def test_no_error(process: ProcessResult):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"
