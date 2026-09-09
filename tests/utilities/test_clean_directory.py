"""
Tests for interactive directory cleanup.
"""

import pytest

from flucs.utilities.clean_directory import clean_directory

pytestmark = pytest.mark.core


def test_clean_directory_without_candidates(tmp_path, monkeypatch, capsys):
    # Set up a file that should not be deleted
    (tmp_path / "keep.txt").write_text("keep")

    # Patch user confirmation response
    monkeypatch.setattr(
        "builtins.input",
        lambda _: pytest.fail("cleanup unexpectedly requested confirmation"),
    )

    # Clean directory using normal function
    clean_directory(tmp_path, ("output.*", "restart.*"))

    # Checks
    assert (tmp_path / "keep.txt").exists()
    assert "No output.* or restart.* files found." in capsys.readouterr().out

@pytest.mark.parametrize(
    ("response", "files_deleted"),
    [
        pytest.param("no", False, id="declined"),
        pytest.param("YES", True, id="confirmed"),
    ],
)
def test_clean_directory_confirmation(
    tmp_path,
    monkeypatch,
    capsys,
    response,
    files_deleted,
):
    # Set up potentially matching files and directories
    output = tmp_path / "output.nc"
    restart = tmp_path / "restart.nc"
    keep = tmp_path / "keep.txt"
    matching_directory = tmp_path / "output.directory"
    
    for path in (output, restart, keep):
        path.write_text(path.name)
    matching_directory.mkdir()

    # Patch user confirmation response
    monkeypatch.setattr("builtins.input", lambda _: response)

    # Clean directory using normal function
    clean_directory(tmp_path, ("output.*", "*.nc"))

    # Checks
    assert output.exists() is not files_deleted
    assert restart.exists() is not files_deleted
    assert keep.exists()

    assert matching_directory.is_dir()
    
    captured = capsys.readouterr().out
    assert captured.count(" - output.nc") == 1
    assert ("Cleanup complete." in captured) is files_deleted
