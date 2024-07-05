# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Tests for type checking dtypes in NDArray.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TextIO

import numpy as np
import pytest
from util import (
    COMPLEX_TYPES,
    FLOATING_TYPES,
    INT_TYPES,
    NP_SCALAR_TYPES_PARENTS,
    UINT_TYPES,
    assert_type_check_passes,
)

# prepare parameters for parametrized test
_DTYPE_TO_PARENTS_MAP = [(k, v) for k, v in NP_SCALAR_TYPES_PARENTS.items()]
_DTYPE_LIST = [INT_TYPES, UINT_TYPES, FLOATING_TYPES, COMPLEX_TYPES]
_ALL_DTYPES = list(itertools.chain.from_iterable(_DTYPE_LIST))


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
    for i, parent_dtype in enumerate(parents):
        test_string += (
            f"x_{i:02d}: NDArray[Shape[int], np.{dtype}] = "
            f"np.array([1], dtype=np.{dtype})\n"
            f"y_{i:02d}: NDArray[Shape[int], np.{parent_dtype}] = "
            f"x_{i:02d}\n\n"
        )
    # run type check and fail if it throws errors
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg=f"Failed test for actual type {dtype}",
    )


@pytest.mark.parametrize("dtype_list", _DTYPE_LIST)
def test_type_checking_dtypes_interoperable_types(
    dtype_list: list[str], temp_file: tuple[TextIO, Path]
) -> None:
    """Check interoperability of types of the same class"""
    # remove non-supported dtypes
    supported_dtypes: list[str] = []
    for dtype in dtype_list:
        if hasattr(np, dtype):
            supported_dtypes.append(dtype)

    # create all possible combinations
    dtype_combinations = itertools.combinations_with_replacement(
        supported_dtypes, 2
    )

    # create a test string containing all possible combinations
    test_string = (
        "from numdantic import NDArray, Shape\nimport numpy as np\n\n"
    )
    for i, type_tuple in enumerate(dtype_combinations):
        actual_type = type_tuple[0]
        target_type = type_tuple[1]
        # type-ignore combinations that must not pass
        actual_type_size = np.dtype(getattr(np, actual_type)).itemsize
        target_type_size = np.dtype(getattr(np, target_type)).itemsize
        compatible = actual_type_size == target_type_size
        ignore = "  # type: ignore" if not compatible else ""

        # append to test string
        test_string += (
            f"x_{i:02d}: NDArray[Shape[int], np.{actual_type}] = "
            f"np.array([1], dtype=np.{actual_type})\n"
            f"y_{i:02d}: NDArray[Shape[int], np.{target_type}] = "
            f"x_{i:02d}{ignore}\n\n"
        )
    # run type check
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg="Failed test for dtype interoperability",
    )


@pytest.mark.parametrize("dtype_list", _DTYPE_LIST)
def test_type_checking_dtypes_incompatible_types(
    dtype_list: list[str], temp_file: tuple[TextIO, Path]
) -> None:
    """Check that incompatible types raise type checking errors"""
    # remove non-supported dtypes
    actual_dtypes: list[str] = []
    for dtype in dtype_list:
        if hasattr(np, dtype):
            actual_dtypes.append(dtype)

    # find all target types that are not in the list of actual types
    target_dtypes: list[str] = []
    for dtype in _ALL_DTYPES:
        if dtype not in actual_dtypes and hasattr(np, dtype):
            target_dtypes.append(dtype)

    # create combinations
    dtype_combinations = itertools.product(target_dtypes, actual_dtypes)

    # prepare test string
    test_string = (
        "from numdantic import NDArray, Shape\nimport numpy as np\n\n"
    )
    for i, type_tuple in enumerate(dtype_combinations):
        actual_type = type_tuple[0]
        target_type = type_tuple[1]
        # append to test string
        test_string += (
            f"x_{i:02d}: NDArray[Shape[int], np.{actual_type}] = "
            f"np.array([1], dtype=np.{actual_type})\n"
            f"y_{i:02d}: NDArray[Shape[int], np.{target_type}] = "
            f"x_{i:02d}  # type: ignore\n\n"
        )
    # run type check
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg="Failed test for incompatible dtypes; types were compatible",
    )


# Four out of the eight generic dtypes behave contravariantly, so we must
# split the test for variance into two categories:
_COVARIANT_GENERICS = [
    ("generic", INT_TYPES + UINT_TYPES + FLOATING_TYPES + COMPLEX_TYPES),
    ("number[Any]", INT_TYPES + UINT_TYPES + FLOATING_TYPES + COMPLEX_TYPES),
    ("integer[Any]", INT_TYPES + UINT_TYPES),
    ("inexact[Any]", FLOATING_TYPES + COMPLEX_TYPES),
]
_CONTRAVARIANT_GENERICS = [
    ("signedinteger[Any]", INT_TYPES),
    ("unsignedinteger[Any]", UINT_TYPES),
    ("floating[Any]", FLOATING_TYPES),
    ("complexfloat[Any, Any]", COMPLEX_TYPES),
]


@pytest.mark.parametrize("generic_dtype, target_dtypes", _COVARIANT_GENERICS)
def test_type_checking_dtypes_generics_covariant(
    generic_dtype: str,
    target_dtypes: list[str],
    temp_file: tuple[TextIO, Path],
) -> None:
    """Generics should behave covariantly (and these actually do)"""
    supported_dtypes = []
    for dtype in target_dtypes:
        if hasattr(np, dtype):
            supported_dtypes.append(dtype)

    # create a test string
    test_string = (
        "from typing import Any\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
    )
    for i, target_dtype in enumerate(supported_dtypes):
        test_string += (
            f"x_{i:02d}: NDArray[Shape[int], np.{generic_dtype}] = "
            f"np.array([1], dtype=np.{generic_dtype})\n"
            f"y_{i:02d}: NDArray[Shape[int], np.{target_dtype}] = "
            f"x_{i:02d}  # type: ignore\n\n"
        )

    # run type check
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg=(
            f"Failed test for {generic_dtype}: type unexpectedly behaved "
            f"contravariantly."
        ),
    )


@pytest.mark.xfail(reason="dtypes behave contravariantly.", strict=True)
@pytest.mark.parametrize(
    "generic_dtype, target_dtypes", _CONTRAVARIANT_GENERICS
)
def test_type_checking_dtypes_generics_contravariant(
    generic_dtype: str,
    target_dtypes: list[str],
    temp_file: tuple[TextIO, Path],
) -> None:
    """Generics should behave covariantly (but these do not!)"""
    supported_dtypes = []
    for dtype in target_dtypes:
        if hasattr(np, dtype):
            supported_dtypes.append(dtype)

    # create a test string
    test_string = (
        "from typing import Any\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
    )
    for i, target_dtype in enumerate(supported_dtypes):
        test_string += (
            f"x_{i:02d}: NDArray[Shape[int], np.{generic_dtype}] = "
            f"np.array([1], dtype=np.{generic_dtype})\n"
            f"y_{i:02d}: NDArray[Shape[int], np.{target_dtype}] = "
            f"x_{i:02d}  # type: ignore\n\n"
        )

    # run type check
    assert_type_check_passes(
        test_string,
        *temp_file,
        fail_msg=(
            f"Failed test for {generic_dtype}: type unexpectedly behaved "
            f"contravariantly."
        ),
    )
