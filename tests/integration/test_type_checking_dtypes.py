"""
Tests for type checking dtypes in NDArray.
"""

from __future__ import annotations

import itertools
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import pytest
from _numpy_salar_types import (
    complex_dtypes,
    floating_dtypes,
    numpy_scalar_types,
    signedinteger_dtypes,
    unsignedinteger_dtypes,
)

project_root = Path(__file__).resolve().parents[2]


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
    # tmp_filepath.unlink()  # clean-up


def assert_type_check_passes(
    test_string: str,
    temp_file: TextIO,
    temp_file_path: Path,
    fail_msg: str = "Type check test failed",
) -> subprocess.CompletedProcess[Any]:
    """
    Assert that the given string passes the type check. Return output.

    :param test_string: File contents to check. Must be formatted as a
        syntactically valid Python file.
    :param temp_file: Opened file-like object to write the test string
        to and to feed to type checker.
    :param temp_file_path: The path of the test file.
    :param fail_msg: A failure message to display should the test fail.
    :return:
    """
    # write test contents to file
    temp_file.write(test_string)
    temp_file.close()

    output = subprocess.run(
        f"mypy --config-file pyproject.toml {str(temp_file_path)}",
        capture_output=True,
        env=os.environ,
        cwd=project_root,  # use pyproject.toml as config file
        shell=True,
    )
    if output.returncode != 0:
        pytest.fail(
            f"{fail_msg}:"
            f"\nstdout:\n{output.stdout.decode('utf-8')}"
            f"\nstderr:\n{output.stderr.decode('utf-8')}\n"
        )
    return output


# prepare parameter for parametrized test
_dtype_parents_map = [(k, v) for k, v in numpy_scalar_types.items()]


@pytest.mark.parametrize("dtype, parents", _dtype_parents_map)
def test_type_checking_dtypes_dtype_hierarchy(
    dtype: str, parents: list[str], temp_file: tuple[TextIO, Path]
) -> None:
    """Test that types can be used where their parent type is expected"""
    # check that there is something to test and type exists
    if not parents:
        return  # no parents means nothing to test
    if not hasattr(np, dtype.removesuffix("[Any]").removesuffix("[Any, Any]")):
        pytest.skip(f"System does not support dtype {dtype}.")

    # create atest file
    test_string = (
        "from typing import Any\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
    )
    for parent_dtype in parents:
        random_id = random.randint(0, 999999)
        test_string += (
            f"x_{random_id}: NDArray[Shape[int], np.{dtype}] = "
            f"np.array([1], dtype=np.{dtype})\n"
            f"y_{random_id}: NDArray[Shape[int], np.{parent_dtype}] = "
            f"x_{random_id}\n\n"
        )
    # run type check and fail if it throws errors
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg=f"Failed test for actual type {dtype}",
    )


_dtype_list = [
    signedinteger_dtypes,
    unsignedinteger_dtypes,
    floating_dtypes,
    complex_dtypes,
]


@pytest.mark.parametrize("dtype_list", _dtype_list)
def test_type_checking_dtypes_interoperable_types(
    dtype_list: dict[str, list[str]], temp_file: tuple[TextIO, Path]
) -> None:
    """Check interoperability of types of the same class"""
    # remove non-supported dtypes
    supported_dtypes = dtype_list["C-type"]
    for dtype in dtype_list["sized alias"]:
        if hasattr(np, dtype):
            supported_dtypes.append(dtype)

    # create all possible combinations
    dtype_combinations = itertools.combinations_with_replacement(
        supported_dtypes, 2
    )

    # create a test string containing all possible combinations
    test_string = (
        "from numdantic import NDArray, Shape\n" "import numpy as np\n\n"
    )
    for actual_type, target_type in dtype_combinations:
        # type-ignore combinations that must not pass
        actual_type_ = np.dtype(getattr(np, actual_type)).itemsize
        target_type_ = np.dtype(getattr(np, target_type)).itemsize
        compatible = actual_type_ == target_type_
        ignore = "  # type: ignore" if not compatible else ""

        # append to test string
        random_id = random.randint(0, 999999)
        test_string += (
            f"x_{random_id}: NDArray[Shape[int], np.{actual_type}] = "
            f"np.array([1], dtype=np.{actual_type})\n"
            f"y_{random_id}: NDArray[Shape[int], np.{target_type}] = "
            f"x_{random_id}{ignore}\n\n"
        )
    # run type check
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg="Failed test for dtype interoperability",
    )
