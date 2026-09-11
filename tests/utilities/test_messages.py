"""
Tests for user-facing message utilities.
"""

import pytest

from flucs.utilities.messages import flucsprint, format_seconds

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    ("seconds", "verbose", "expected"),
    [
        pytest.param(0, False, "00:00:00:00", id="zero"),
        pytest.param(90061, False, "01:01:01:01", id="compact"),
        pytest.param(1, True, "1 second", id="singular"),
        pytest.param(
            90061,
            True,
            "1 day, 1 hour, 1 minute, 1 second",
            id="verbose",
        ),
    ],
)
def test_format_seconds(seconds, verbose, expected):
    assert format_seconds(seconds, verbose=verbose) == expected


def test_flucsprint_prefixes_and_validation(capsys):
    # Print ordinary message and warning with source
    flucsprint("ordinary", "message")
    flucsprint("look here", source="Setup", message_type="warning")

    # Check that the output is as expected
    assert capsys.readouterr().out == (
        "ordinary message\n\n[Setup] WARNING: look here\n\n"
    )

    # Check input validation for message_type
    with pytest.raises(ValueError, match="Invalid message_type 'debug'"):
        flucsprint("invalid", message_type="debug")
