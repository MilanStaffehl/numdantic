# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Tests for type checking shapes of NDArray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

from util import assert_type_check_passes

if TYPE_CHECKING:
    from pathlib import Path


def test_type_checking_shapes_exact_matches(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that exactly matching shape annotations pass type check"""
    # fmt: off
    test_string = (
        "from typing import NewType, Literal as L\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_int: NDArray[Shape[int, int], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_int: NDArray[Shape[int, int], np.int32] = x_int\n\n"
        "x_lit: NDArray[Shape[L[2], L[2]], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_lit: NDArray[Shape[L[2], L[2]], np.int32] = x_lit\n\n"
        "x_nt: NDArray[Shape[AxisLen, AxisLen], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_nt: NDArray[Shape[AxisLen, AxisLen], np.int32] = x_nt\n\n"
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
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_int: NDArray[Shape[int, int], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_int: NDArray[Shape[int], np.int32] = x_int"
        "  # type: ignore\n\n"
        "x_lit: NDArray[Shape[L[2], L[2]], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_lit: NDArray[Shape[L[5], L[5]], np.int32] = x_lit"  # wrong len
        "  # type: ignore\n\n"
        "y_lit_: NDArray[Shape[L[2]], np.int32] = x_lit"  # wrong dims
        "  # type: ignore\n\n"
        "x_nt: NDArray[Shape[AxisLen, AxisLen], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_nt: NDArray[Shape[AxisLen], np.int32] = x_nt"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_literal_with_int(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that literals are incompatible with generic integer shapes"""
    # fmt: off
    test_string = (
        "from typing import Literal as L\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "x_1: NDArray[Shape[L[2], L[2]], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_1: NDArray[Shape[int, int], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[Shape[int, int], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_2: NDArray[Shape[L[2], L[2]], np.int32] = x_2"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)


def test_type_checking_shapes_named_axis_with_int(
    temp_file: tuple[TextIO, Path]
) -> None:
    """Test that named axes are incompatible with generic integer shapes"""
    # fmt: off
    test_string = (
        "from typing import NewType\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[Shape[AxisLen, AxisLen], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_1: NDArray[Shape[int, int], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[Shape[int, int], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_2: NDArray[Shape[AxisLen, AxisLen], np.int32] = x_2"
        "  # type: ignore\n\n"
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
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x_1: NDArray[Shape[AxisLen, AxisLen], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_1: NDArray[Shape[L[2], L[2]], np.int32] = x_1"
        "  # type: ignore\n\n"
        "x_2: NDArray[Shape[L[2], L[2]], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y_2: NDArray[Shape[AxisLen, AxisLen], np.int32] = x_2"
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
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "AxisLen = NewType('AxisLen', int)\n\n"
        "x: NDArray[Shape[AxisLen, int, L[2]], np.int32] = "
        "np.array([[[1, 2]], [[3, 4]]], dtype=np.int32)\n"
        "y: NDArray[Shape[AxisLen, int, L[2]], np.int32] = x\n\n"
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
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
        "AxisOne = NewType('AxisOne', int)\n"
        "AxisTwo = NewType('AxisTwo', int)\n\n"
        "x: NDArray[Shape[AxisTwo, AxisOne], np.int32] = "
        "np.array([[1, 2], [3, 4]], dtype=np.int32)\n"
        "y: NDArray[Shape[AxisOne, AxisTwo], np.int32] = x"
        "  # type: ignore\n\n"
    )
    # fmt: on
    assert_type_check_passes(test_string, *temp_file)
