"""
Tests for type checking dtypes in NDArray.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import TextIO

import numpy as np
import pytest
from util import (
    COMPLEX_TYPES,
    FLOATING_TYPES,
    INT_TYPES,
    NUMPY_SCALAR_TYPES,
    UINT_TYPES,
    assert_type_check_passes,
)

# prepare parameters for parametrized test
_DTYPE_TO_PARENTS_MAP = [(k, v) for k, v in NUMPY_SCALAR_TYPES.items()]
_DTYPE_LIST = [INT_TYPES, UINT_TYPES, FLOATING_TYPES, COMPLEX_TYPES]


@pytest.mark.parametrize("dtype, parents", _DTYPE_TO_PARENTS_MAP)
def test_type_checking_dtypes_dtype_hierarchy(
    dtype: str, parents: list[str], temp_file: tuple[TextIO, Path]
) -> None:
    """Test that types can be used where their parent type is expected"""
    # check that there is something to test and type exists
    if not parents:
        return  # no parents means nothing to test
    if not hasattr(np, dtype.removesuffix("[Any]").removesuffix("[Any, Any]")):
        pytest.skip(f"System does not support dtype {dtype}.")

    # create a test file
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


@pytest.mark.parametrize("dtype_list", _DTYPE_LIST)
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
