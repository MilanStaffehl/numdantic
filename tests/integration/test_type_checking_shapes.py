# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Tests for type checking shapes of NDArray.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, TextIO

import pytest
from util import ASSIGN_IGNORE, IS_PRE_NUMPY_2_1, assert_type_check_passes

if TYPE_CHECKING:
    from pathlib import Path


def test_type_checking_shapes_exact_matches(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that exactly matching shape annotations pass type check"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_int: NDArray[tuple[int, int], np.int32] = "  # int
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_int: NDArray[tuple[int, int], np.int32] = x_int\n\n"
        "x_lit: NDArray[tuple[L[2], L[2]], np.int32] = "  # Literal
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_lit: NDArray[tuple[L[2], L[2]], np.int32] = x_lit\n\n"
        "x_nt: NDArray[tuple[AxisLen, AxisLen], np.int32] = "  # named
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_nt: NDArray[tuple[AxisLen, AxisLen], np.int32] = x_nt\n\n"
        "x_int_ell: NDArray[tuple[int, ...], np.int32] = "  # int, ...
        f"np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_int_ell: NDArray[tuple[int, ...], np.int32] = x_int_ell\n\n"
        "x_lit_ell: NDArray[tuple[L[2], ...], np.int32] = "  # L, ...
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_lit_ell: NDArray[tuple[L[2], ...], np.int32] = x_lit_ell\n\n"
        "x_nt_ell: NDArray[tuple[AxisLen, ...], np.int32] = "  # n, ...
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_nt_ell: NDArray[tuple[AxisLen, ...], np.int32] = x_nt_ell\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_mismatched_shapes(
    temp_file: tuple[TextIO, Path],
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
        "y_int: NDArray[tuple[int], np.int32] = x_int"  # wrong dims
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
    temp_file: tuple[TextIO, Path],
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
    temp_file: tuple[TextIO, Path],
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
    temp_file: tuple[TextIO, Path],
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
    temp_file: tuple[TextIO, Path],
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
    temp_file: tuple[TextIO, Path],
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


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="Shape type is invariant in numpy < 2.1.",
    strict=True,
)
def test_type_checking_shapes_assignment_to_int_ellipses(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that all shapes can be assigned to tuple[int, ...]."""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[int, ...], np.int32] = x_1\n\n"
        "x_2: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[int, ...], np.int32] = x_2\n\n"
        "x_3: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_3: NDArray[tuple[int, ...], np.int32] = x_3\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="Shape type is invariant in numpy < 2.1.",
    strict=True,
)
def test_type_checking_shapes_assignment_to_literal_ellipses(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that only literals can be assigned to tuple[L[2], ...]."""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[L[2], ...], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[L[2], ...], np.int32] = x_2\n\n"  # works!
        "x_3: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_3: NDArray[tuple[L[2], ...], np.int32] = x_3"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="Shape type is invariant in numpy < 2.1.",
    strict=True,
)
def test_type_checking_shapes_assignment_to_named_axes_ellipses(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that only named axes can be assigned to tuple[AxisLen, ...]."""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[int, int], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[AxisLen, ...], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[tuple[L[2], L[2]], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[AxisLen, ...], np.int32] = x_2"
        "  # type: ignore\n\n"
        "x_3: NDArray[tuple[AxisLen, AxisLen], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_3: NDArray[tuple[AxisLen, ...], np.int32] = x_3\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="Literal assignment to int-based shape is invalid in numpy < 2.1",
    strict=True,
)
def test_type_checking_shapes_literal_ellipsis_with_int_ellipsis(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that literal ellipses are compatible with integer ellipses"""
    # fmt: off
    test_string = (
        "from typing import Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "x_1: NDArray[tuple[L[2], ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[int, ...], np.int32] = x_1\n\n"
        "x_2: NDArray[tuple[int, ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_2: NDArray[tuple[L[2], ...], np.int32] = x_2"
        "  # type: ignore\n\n"  # other way around is still illegal
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


@pytest.mark.xfail(
    IS_PRE_NUMPY_2_1,
    reason="NewType assignment to int is invalid in numpy < 2.1",
    strict=True,
)
def test_type_checking_shapes_named_axis_ellipsis_with_int_ellipsis(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that named axes ellipses are compatible with integer ellipsis"""
    # fmt: off
    test_string = (
        "from typing import NewType\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[AxisLen, ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[int, ...], np.int32] = x_1\n\n"
        "x_2: NDArray[tuple[int, ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_2: NDArray[tuple[AxisLen, ...], np.int32] = x_2"
        "  # type: ignore\n\n"  # other way around is still illegal
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_named_axis_ellipsis_with_literal_ellipsus(
    temp_file: tuple[TextIO, Path],
) -> None:
    """Test that named axes ellipses are incompatible with literal ellipses"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[tuple[AxisLen, ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_1: NDArray[tuple[L[2], ...], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[tuple[L[2], ...], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){ASSIGN_IGNORE}\n"
        "y_2: NDArray[tuple[AxisLen, ...], np.int32] = x_2"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


_TARGET_TYPES = ["int, int", "L[2], L[2]", "AxisLen, AxisLen"]
_SOURCE_TYPES = ["int, ...", "L[2], ...", "AxisLen, ..."]
_TEST_CASES = itertools.product(_SOURCE_TYPES, _TARGET_TYPES)


@pytest.mark.parametrize("source_type, target_type", _TEST_CASES)
def test_type_checking_shapes_cannot_assign_ellipses_to_explicit_shape(
    source_type: str, target_type: str, temp_file: tuple[TextIO, Path]
) -> None:
    """Test that ellipsis expressions cannot be assigned to explicit shapes"""
    # Assignment passes even on np 2.2 if the source type is
    # tuple[int, ...], so we must not ignore it:
    if source_type == "int, ...":
        assign_ign = ""
    else:
        assign_ign = ASSIGN_IGNORE
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        f"x: NDArray[tuple[{source_type}], np.int32] = "
        f"np.array([[1, 2], [3, 4]], dtype=np.int32){assign_ign}\n"
        f"y: NDArray[tuple[{target_type}], np.int32] = x"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)
