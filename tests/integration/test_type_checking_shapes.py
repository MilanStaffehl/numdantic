# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Tests for type checking shapes of NDArray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

import pytest
from util import ASSIGN_IGNORE, IS_PRE_NUMPY_2_1, assert_type_check_passes

if TYPE_CHECKING:
    from pathlib import Path


def test_type_checking_shapes_exact_matches(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that exactly matching shape annotations pass type check"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_int: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_int: NDArray[tuple[int, int], np.int32] = x_int\n\n"
        "x_lit: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_lit: NDArray[tuple[L[2], L[2]], np.int32] = x_lit\n\n"
        "x_nt: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_nt: NDArray[tuple[AxisLen, AxisLen], np.int32] = x_nt\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_mismatched_shapes(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that type checkers raise an error when shapes are mismatched"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_int: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_int: NDArray[tuple[int], np.int32] = x_int"
        "  # type: ignore\n\n"
        "x_lit: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_lit: NDArray[tuple[L[5], L[5]], np.int32] = x_lit"  # wrong len
        "  # type: ignore\n\n"
        "y_lit_: NDArray[tuple[L[2]], np.int32] = x_lit"  # wrong dims
        "  # type: ignore\n\n"
        "x_nt: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_nt: NDArray[tuple[AxisLen], np.int32] = x_nt"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="Literal assignment to int-based shape is invalid in numpy < 2.1",
    strict=True,
)
def test_type_checking_shapes_literal_with_int(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that literals are compatible with generic integer shapes"""
    # fmt: off
    test_string = (
        "from typing import Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "x_1: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[int, int], np.int32] = x_1\n\n"
        "x_2: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[L[2], L[2]], np.int32] = x_2"
        "  # type: ignore\n\n"  # other way around is still illegal
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="NewType assignment to int is invalid in numpy < 2.1",
    strict=True,
)
def test_type_checking_shapes_named_axis_with_int(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that named axes are compatible with generic integer shapes"""
    # fmt: off
    test_string = (
        "from typing import NewType\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[int, int], np.int32] = x_1\n\n"
        "x_2: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[AxisLen, AxisLen], np.int32] = x_2"
        "  # type: ignore\n\n"  # other way around is still illegal
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_named_axis_with_literal(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that named axes are incompatible with literal shapes"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[L[2], L[2]], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[AxisLen, AxisLen], np.int32] = x_2"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_mixed_shape_annotations(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that multiple types can be used in one shape type"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x: NDArray[tuple[AxisLen, int, L[2]], np.int32] = "
        f"np.array([[[1, 2]], [[3, 4]]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y: NDArray[tuple[AxisLen, int, L[2]], np.int32] = x\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_switched_named_axes(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that switched named axes cause an error"""
    # fmt: off
    test_string = (
        "from typing import NewType\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisOne = NewType('AxisOne', int)\n"
        "AxisTwo = NewType('AxisTwo', int)\n\n"
        "x: NDArray[tuple[AxisTwo, AxisOne], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y: NDArray[tuple[AxisOne, AxisTwo], np.int32] = x"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_indeterminate_dimensionality(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that shapes can be tuple[int, ...]"""
    # fmt: off
    test_string = (
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "x: NDArray[tuple[int, ...], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y: NDArray[tuple[int, ...], np.int32] = x\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="Shape type is still invariant.",
    strict=True,
)
def test_type_checking_shapes_mixing_indeterminate_dims(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test mixing shapes of indeterminate dimensionality with other shapes"""
    # fmt: off
    test_string = (
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "x_1: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[int, ...], np.int32] = x_1\n\n"
        "x_2: NDArray[tuple[int, ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_2: NDArray[tuple[int, int], np.int32] = x_2"
        "  # type: ignore\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)
