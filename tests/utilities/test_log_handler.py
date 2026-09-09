"""
Tests for FLUCS log handling.
"""

import sys
from io import StringIO

import pytest

from flucs.utilities.log_handler import FlucsLogHandler

pytestmark = pytest.mark.core


def test_log_handler_redirects_and_restores_streams(capsys):
    # Create in-memory text stream to prevent writing
    log = StringIO()

    # Record current output streams.
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Check that output is redirected and that exception handling works
    with pytest.raises(RuntimeError, match="sentinel"):
        with FlucsLogHandler(log):  # Temporarily redirect stdout and stderr
            print("standard output")
            print("standard error", file=sys.stderr)
            raise RuntimeError("sentinel")

    # Verify retoration of stdout and stderr
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr

    # Returns output captured during the test
    captured = capsys.readouterr()

    # Check that only the redirected messages reached standard output
    assert captured.out == "standard output\nstandard error\n"
    assert captured.err == ""

    # Check that the log contains the expected output
    assert "standard output" in log.getvalue()
    assert "standard error" in log.getvalue()
    assert "RuntimeError: sentinel" in log.getvalue()
