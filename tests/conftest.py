# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Global fixtures for all tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TextIO

import pytest


@pytest.fixture
def temp_file(tmp_path: Path) -> Iterator[tuple[TextIO, Path]]:
    """
    Set-up and tear-down for a temporary file.

    Python temporary files are incredibly brittle and hard-to-use in any
    reliable fashion. On Windows, they cannot be opened a second time,
    and therefore cannot be used for the purpose of these tests that
    need to write to the file and then allow mypy to read from them.
    Therefore, we create "actual" files.
    """
    tmp_filepath = tmp_path / "mock_file.py"
    tmp_file = open(tmp_filepath, "w")
    yield tmp_file, tmp_filepath
    tmp_file.close()
    tmp_filepath.unlink()  # clean-up
