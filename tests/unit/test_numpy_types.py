# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Unit tests for the _numpy_types.py module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing
import pytest
from pydantic_core import PydanticCustomError, ValidationError

from numdantic import _numpy_types

if TYPE_CHECKING:
    from pytest_mock import MockerFixture, MockType
    from pytest_subtests import SubTests


CastFuncType: TypeAlias = Callable[
    [Sequence[Any] | numpy.typing.NDArray[np.generic]],
    numpy.typing.NDArray[np.generic],
]


@pytest.fixture
def patch_pydantic_error(mocker: MockerFixture) -> Iterator[MockType]:
    """Mock the PydanticCustomError class and return its mock"""

    def return_mock_error(
        err_type: str, msg: str, _ctx: dict[str, Any]
    ) -> tuple[str, str]:
        return err_type, msg

    # patch where the class is actually called
    mock_error = mocker.patch("numdantic._numpy_types.PydanticCustomError")
    mock_error.side_effect = return_mock_error
    yield mock_error


# Tests for _validate_array_shape implicitly test _validate_array_axis as well
def test_validate_array_shape_valid_shape_generic(
    patch_pydantic_error: MockType,
) -> None:
    """Test validation with a valid shape for generic axis length"""
    test_array = np.array([[1, 2], [3, 4]])
    expected_shape = ("int", "int")
    error_list = []  # type: ignore
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    # check output and that no error was reported
    np.testing.assert_equal(test_array, output)
    assert len(error_list) == 0
    patch_pydantic_error.assert_not_called()


def test_validate_array_shape_invalid_shape_generic(
    patch_pydantic_error: MockType,
) -> None:
    """Test validation with an invalid shape for generic axis length"""
    test_array = np.array([1, 2, 3])
    expected_shape = ("int", "int")
    error_list = []  # type: ignore
    # check that an error is generated
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    np.testing.assert_equal(test_array, output)  # array unchanged
    expected_error = (
        "array_dimensions",
        "Mismatched dimensions: got {received_dim}D array, "
        "expected {expected_dim}D array",
    )
    patch_pydantic_error.assert_called_with(
        *expected_error,
        {"received_dim": 1, "expected_dim": 2},
    )
    assert len(error_list) == 1
    assert error_list[0] == expected_error


def test_validate_array_shape_valid_shape_literal(
    patch_pydantic_error: MockType,
) -> None:
    """Test validation with a valid shape for literal axis length"""
    test_array = np.array([[1, 2], [3, 4]])
    expected_shape = (2, 2)
    error_list = []  # type: ignore
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    # check output and that no error was reported
    np.testing.assert_equal(test_array, output)
    assert len(error_list) == 0
    patch_pydantic_error.assert_not_called()


def test_validate_array_shape_invalid_shape_literal_dims(
    patch_pydantic_error: MockType,
) -> None:
    """Input array has invalid dimensions when typed with literal ints"""
    test_array = np.array([1, 2, 3])
    expected_shape = (2, 2)
    error_list = []  # type: ignore
    # check that two errors are generated: one for dims, one for axis len
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    np.testing.assert_equal(test_array, output)  # array unchanged
    assert len(error_list) == 2

    # error for dimensions
    expected_error = (
        "array_dimensions",
        "Mismatched dimensions: got {received_dim}D array, "
        "expected {expected_dim}D array",
    )
    patch_pydantic_error.assert_any_call(
        *expected_error,
        {"received_dim": 1, "expected_dim": 2},
    )
    assert error_list[0] == expected_error

    # error for shape
    expected_error = (
        "array_shape",
        "Mismatched shapes: got shape {received_shape}, expected "
        "{expected_shape}",
    )
    patch_pydantic_error.assert_called_with(
        *expected_error,
        {"received_shape": (3,), "expected_shape": (2, 2)},
    )
    assert error_list[1] == expected_error


def test_validate_array_shape_invalid_shape_literal_len(
    patch_pydantic_error: MockType,
) -> None:
    """Input array has invalid axis length when typed with literal ints"""
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    expected_shape = (2, 2)
    error_list = []  # type: ignore
    # check that only one error due to axis length is generated
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    np.testing.assert_equal(test_array, output)  # array unchanged
    assert len(error_list) == 1
    expected_error = (
        "array_shape",
        "Mismatched shapes: got shape {received_shape}, expected "
        "{expected_shape}",
    )
    patch_pydantic_error.assert_called_with(
        *expected_error,
        {"received_shape": (2, 3), "expected_shape": (2, 2)},
    )
    assert error_list[0] == expected_error


def test_validate_array_shape_valid_shape_named_axes(
    patch_pydantic_error: MockType,
) -> None:
    """Test that named axes of same length pass without error"""
    test_array = np.array([[1, 2], [3, 4]])
    expected_shape = ("AxisLen", "AxisLen")
    error_list = []  # type: ignore
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    np.testing.assert_equal(test_array, output)
    assert len(error_list) == 0
    patch_pydantic_error.assert_not_called()


def test_validate_array_shape_invalid_shape_named_axes(
    patch_pydantic_error: MockType,
) -> None:
    """Test that two named axes of different length cause an error"""
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    expected_shape = ("AxisLen", "AxisLen")
    error_list = []  # type: ignore
    output = _numpy_types._validate_array_shape(
        test_array, expected_shape, error_list
    )
    np.testing.assert_equal(test_array, output)
    # check error due to mismatched axis lengths
    assert len(error_list) == 1
    expected_error = (
        "array_axis_length",
        "Invalid axis length: Axis {failed_at_index} is "
        "typed to be of same length as axis {first_occurrence}. "
        "Expected length {expected_len}, got {received_len}. "
        "Expected shape: {expected_shape}",
    )
    expected_ctx = {
        "failed_at_index": 1,
        "first_occurrence": 0,
        "expected_len": 2,
        "received_len": 3,
        "expected_shape": expected_shape,
    }
    patch_pydantic_error.assert_called_with(
        *expected_error,
        expected_ctx,
    )
    assert error_list[0] == expected_error


def test_validate_array_shape_valid_shape_named_axes_different_names(
    patch_pydantic_error: MockType, subtests: SubTests
) -> None:
    """Test that named axes of different length pass without error"""
    # check that it makes no difference if two differently named axes
    # have the same or different length
    test_arrays = {
        "same_len": np.array([[1, 2], [3, 4]]),
        "different_len": np.array([[1, 2, 3], [4, 5, 6]]),
    }
    expected_shape = ("AxisOne", "AxisTwo")
    for msg, test_array in test_arrays.items():
        with subtests.test(msg=msg):
            error_list = []  # type: ignore
            output = _numpy_types._validate_array_shape(
                test_array, expected_shape, error_list
            )
            # check output and that no error was reported
            np.testing.assert_equal(test_array, output)
            assert len(error_list) == 0
            patch_pydantic_error.assert_not_called()


@pytest.mark.parametrize("expected_dtype", [np.uint64, np.unsignedinteger])
def test_validate_array_dtype_valid_dtype(
    expected_dtype: np.generic,
    patch_pydantic_error: MockType,
    subtests: SubTests,
) -> None:
    """When dtypes match, an error is never raised"""
    for validation_mode in [True, False]:
        with subtests.test(msg=f"strict mode: {validation_mode}"):
            test_array = np.array([[1, 2], [3, 4]], dtype=np.uint64)
            error_list = []  # type: ignore
            output = _numpy_types._validate_array_dtype(
                test_array, expected_dtype, error_list, validation_mode
            )
            np.testing.assert_equal(test_array, output)
            assert len(error_list) == 0
            patch_pydantic_error.assert_not_called()


def test_validate_array_dtype_invalid_dtype_specific_dtype_no_cast(
    patch_pydantic_error: MockType,
) -> None:
    """Given dtype does not match specific expected dtype, can't cast"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.float64)
    expected_dtype = np.uint64

    # lax mode: types cannot be cast
    error_list = []  # type: ignore
    output = _numpy_types._validate_array_dtype(
        test_array,
        expected_dtype,  # type: ignore[arg-type]  # issue #7
        error_list,
        False,
    )
    np.testing.assert_equal(test_array, output)
    assert len(error_list) == 1
    expected_error = (
        "array_dtype",
        "Mismatched dtypes: cannot safely cast {received_dtype} to "
        "{expected_dtype}",
    )
    patch_pydantic_error.assert_called_with(
        *expected_error,
        {"received_dtype": "float64", "expected_dtype": "uint64"},
    )
    assert error_list[0] == expected_error

    # strict mode: types cannot be cast
    error_list = []
    output = _numpy_types._validate_array_dtype(
        test_array,
        expected_dtype,  # type: ignore[arg-type]  # issue #7
        error_list,
        True,
    )
    np.testing.assert_equal(test_array, output)
    assert len(error_list) == 1
    expected_error = (
        "array_dtype",
        "Mismatched dtypes: got {received_dtype}, expected {expected_dtype}",
    )
    patch_pydantic_error.assert_called_with(
        *expected_error,
        {"received_dtype": "float64", "expected_dtype": "uint64"},
    )
    assert error_list[0] == expected_error


def test_validate_array_dtype_invalid_dtype_specific_dtype_can_cast(
    patch_pydantic_error: MockType,
) -> None:
    """Given dtype does not match specific expected dtype, but can cast"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected_dtype = np.float64

    # lax mode: types cannot be cast
    error_list = []  # type: ignore
    output = _numpy_types._validate_array_dtype(
        test_array,
        expected_dtype,  # type: ignore[arg-type]  # issue #7
        error_list,
        False,
    )
    np.testing.assert_equal(test_array, output)
    assert output.dtype == np.dtype(expected_dtype)
    assert len(error_list) == 0
    patch_pydantic_error.assert_not_called()


@pytest.fixture
def cast_function() -> Iterator[CastFuncType]:
    """Return the cast function retrievable from _get_cast_function"""
    # reason for type ignore: issue #7
    yield _numpy_types._get_cast_function(np.int64)  # type: ignore[arg-type]


def test_cast_to_array_from_array(cast_function: CastFuncType) -> None:
    """Test that functions returns arrays as-is (even if dtype is wrong)"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    output = cast_function(test_array)
    np.testing.assert_equal(test_array, output)
    assert output.dtype == np.dtype(np.float32)  # not changed by cast


def test_cast_to_array_nested_list(cast_function: CastFuncType) -> None:
    """Test that functions casts nested lists to array of correct dtype"""
    test_sequence = [[1.0, 2.0], [3.0, 4.0]]  # elements are floats!
    output = cast_function(test_sequence)
    np.testing.assert_equal(np.array([[1, 2], [3, 4]], dtype=np.int64), output)
    assert output.dtype == np.dtype(np.int64)


def test_cast_to_array_nested_tuple(cast_function: CastFuncType) -> None:
    """Test that functions casts nested tuples to array of correct dtype"""
    test_sequence = ((1.0, 2.0), (3.0, 4.0))  # elements are floats!
    output = cast_function(test_sequence)
    np.testing.assert_equal(np.array([[1, 2], [3, 4]], dtype=np.int64), output)
    assert output.dtype == np.dtype(np.int64)


def test_cast_to_array_nested_mixed_sequence(
    cast_function: CastFuncType,
) -> None:
    """Test that functions casts mixed sequences to array of correct dtype"""
    test_sequence = [(1.0, 2.0), (3.0, 4.0)]  # list of tuples
    output = cast_function(test_sequence)
    np.testing.assert_equal(np.array([[1, 2], [3, 4]], dtype=np.int64), output)
    assert output.dtype == np.dtype(np.int64)


def test_cast_to_array_nested_incompatible_element_types(
    cast_function: CastFuncType,
) -> None:
    """Test that an error is raised when sequence elements are incompatible"""
    test_sequence = [["str", 1], [0.2, "abc"]]
    with pytest.raises(PydanticCustomError) as excinfo:
        cast_function(test_sequence)
    expected_msg = (
        "Received sequence has inhomogeneous part or invalid element types. "
        "Original exception:\ninvalid literal for int() with base 10: 'str'"
    )
    assert str(excinfo.value) == expected_msg


def test_cast_to_array_nested_inhomogeneous_sequence(
    cast_function: CastFuncType,
) -> None:
    """Test that an error is raised for inhomogeneous sequences"""
    test_sequence = [[1, 2, 3], [1, 2], 5]
    with pytest.raises(PydanticCustomError) as excinfo:
        cast_function(test_sequence)
    expected_msg = (
        "Received sequence has inhomogeneous part or invalid element types. "
        "Original exception:\nsetting an array element with a sequence. The "
        "requested array has an inhomogeneous shape after 1 dimensions. The "
        "detected shape was (3,) + inhomogeneous part."
    )
    assert str(excinfo.value) == expected_msg


def test_raise_validation_error_single_error() -> None:
    """Test raising a Validation error from one error"""
    input_array = np.array([1, 2], dtype=np.int32)
    errors = [PydanticCustomError("err_type", "error_message")]
    with pytest.raises(ValidationError) as excinfo:
        _numpy_types._raise_validation_error(input_array, errors, "model name")
    expected_msg = (
        "1 validation error for model name\n  error_message [type=err_type, "
        "input_value=array([1, 2], dtype=int32), input_type=ndarray]"
    )
    assert str(excinfo.value) == expected_msg


def test_raise_validation_error_two_errors() -> None:
    """Test raising a Validation error from two errors"""
    input_array = np.array([1, 2], dtype=np.int32)
    errors = [
        PydanticCustomError("err_type_1", "error_message_1"),
        PydanticCustomError("err_type_2", "error_message_2"),
    ]
    with pytest.raises(ValidationError) as excinfo:
        _numpy_types._raise_validation_error(input_array, errors, "model name")
    expected_msg = (
        "2 validation errors for model name\n  error_message_1 "
        "[type=err_type_1, input_value=array([1, 2], dtype=int32), "
        "input_type=ndarray]\n  error_message_2 [type=err_type_2, "
        "input_value=array([1, 2], dtype=int32), input_type=ndarray]"
    )
    assert str(excinfo.value) == expected_msg


def test_raise_validation_error_no_model_name() -> None:
    """Test raising a Validation error without a model name"""
    input_array = np.array([1, 2], dtype=np.int32)
    errors = [PydanticCustomError("err_type", "error_message")]
    with pytest.raises(ValidationError) as excinfo:
        _numpy_types._raise_validation_error(input_array, errors, None)
    expected_msg = (
        "1 validation error for NDArray validation\n  error_message "
        "[type=err_type, input_value=array([1, 2], dtype=int32), "
        "input_type=ndarray]"
    )
    assert str(excinfo.value) == expected_msg
