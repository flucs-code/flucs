"""
Safety checks for mutually exclusive flucs operating modes.
"""

import sys
from itertools import combinations

import pytest

import flucs.flucs as flucs_cli

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Operating modes
# ------------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("first_mode", "second_mode"),
    list(
        combinations(
            [
                ("--run",),
                ("--init", "TestFourierSystem"),
                ("--list",),
                ("--timing", "1"),
                ("--clean",),
                ("--reconstruct", "restart.nc"),
                ("--postprocess",),
            ],
            2,
        ),
    ),
    ids=lambda mode: mode[0].removeprefix("--"),
)
def test_conflicting_operating_modes(
    tmp_path,
    monkeypatch,
    capsys,
    first_mode,
    second_mode,
):
    """
    Reject conflicting modes before execution or modification of run files.
    """
    # Create recognizable files that must remain untouched.
    original_files = {
        "input.toml": b"Original input\n",
        "output.log": b"Original output\n",
        "restart.nc": b"Original restart\n",
    }
    for filename, contents in original_files.items():
        (tmp_path / filename).write_bytes(contents)

    # Prevent a broken exclusivity check from starting a solver or profiler.
    for run_function in ("run_flucs", "run_flucs_under_nsys"):
        monkeypatch.setattr(
            flucs_cli,
            run_function,
            lambda *_: pytest.fail("Solver execution must not begin"),
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["flucs", "--io_path", str(tmp_path), *first_mode, *second_mode],
    )

    # Enter the real CLI and require an operating-mode conflict.
    with pytest.raises(SystemExit) as error:
        flucs_cli.main()

    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    # Preserve existing contents and do not create additional files.
    saved_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert saved_files == original_files
