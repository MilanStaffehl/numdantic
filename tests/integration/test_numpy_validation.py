# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Test the integration between the custom numpy validator and pydantic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NewType

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from numdantic import NDArray, Shape

if TYPE_CHECKING:
    from pytest_subtests import SubTests

# custom axis length type, based on int
AxisLen = NewType("AxisLen", int)
AxisLen2 = NewType("AxisLen2", int)


class BasicTestModel(BaseModel):
    """Test base model for a 2x2 array of int32 dtype"""

    matrix: NDArray[Shape[Literal[2], Literal[2]], np.int32]


class FloatTestModel(BaseModel):
    """Test base model for a 2x2 array of int32 dtype"""

    matrix: NDArray[Shape[Literal[2], Literal[2]], np.float32]


class StrictTestModel(BaseModel):
    """Test model for a 2x2 array of int32 dtype, with strict validation"""

    model_config = ConfigDict(strict=True)
    matrix: NDArray[Shape[Literal[2], Literal[2]], np.int32]


class UnspecificDtypeModel(BaseModel):
    """Test model for a 2x2 array, but with generic floating point type"""

    matrix: NDArray[Shape[Literal[2], Literal[2]], np.floating]  # type: ignore


class UnspecifiedAxisLengthModel(BaseModel):
    """Test model for a 3D array of unspecified axis length"""

    matrix: NDArray[Shape[int, int, int], np.int32]


class CustomAxisTypeModel(BaseModel):
    """Test custom types as axis length"""

    matrix: NDArray[Shape[AxisLen, AxisLen], np.int32]


class ComplexAxisLengthModel(BaseModel):
    """Model mixing complex axes lengths"""

    matrix: NDArray[Shape[int, AxisLen, AxisLen2, AxisLen, AxisLen2], np.int32]


def test_numpy_simple_array_validation() -> None:
    """Test that the validator can validate a correct array"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    my_model = BasicTestModel(matrix=test_array)
    serialization = my_model.model_dump()
    assert "matrix" in serialization
    np.testing.assert_array_equal(test_array, serialization["matrix"])


def test_numpy_validation_with_wrong_object_type() -> None:
    """Wrong types raise aValidationError"""
    with pytest.raises(ValidationError) as excinfo:
        BasicTestModel(matrix=1.234)  # type: ignore
    expected_msg = (
        "1 validation error for BasicTestModel\nmatrix\n  Input must be a "
        "sequence or a numpy array, received float instead"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_with_wrong_array_dtype_lax_compatible() -> None:
    """Validator in lax mode will silently cast compatible dtypes"""
    test_array = np.array([[1, 2], [2, 3]], dtype=np.int16)
    my_model = BasicTestModel(matrix=test_array)  # type: ignore[arg-type]
    serialization = my_model.model_dump()
    assert "matrix" in serialization
    np.testing.assert_array_equal(test_array, serialization["matrix"])
    assert serialization["matrix"].dtype == np.int32


def test_numpy_validation_with_wrong_array_dtype_lax_incompatible() -> None:
    """Validator in lax mode will raise exception for incompatible dtypes"""
    test_array = np.array([[1, 2], [2, 3]], dtype=np.int64)
    with pytest.raises(ValidationError) as excinfo:
        BasicTestModel(matrix=test_array)  # type: ignore[arg-type]
    # assert error message
    expected_msg = (
        "1 validation error for BasicTestModel\nmatrix\n  Mismatched dtypes: "
        "cannot safely cast int64 to int32"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_with_wrong_array_dtype_strict() -> None:
    """Validator in strict mode will raise exception for incompatible dtypes"""
    test_array = np.array([[1, 2], [2, 3]], dtype=np.int16)
    with pytest.raises(ValidationError) as excinfo:
        StrictTestModel(matrix=test_array)  # type: ignore[arg-type]
    # assert error message
    expected_msg = (
        "1 validation error for StrictTestModel\nmatrix\n  Mismatched dtypes: "
        "got int16, expected int32"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_with_wrong_array_shape() -> None:
    """Validator will raise exception for wrong shape"""
    test_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    with pytest.raises(ValidationError) as excinfo:
        BasicTestModel(matrix=test_array)
    # assert error message
    expected_msg = (
        "1 validation error for BasicTestModel\nmatrix\n  Mismatched shapes: "
        "got shape (2, 3), expected (2, 2)"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_with_variable_axis_length(
    subtests: SubTests,
) -> None:
    """Unspecified axes lengths are accepted"""
    # 3D-array is the only requirement, so we test different ones
    arrays = [
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        np.array([[[1], [2]], [[3], [4]]], dtype=np.int32),
        np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 1], [2, 3]]],
            dtype=np.int32,
        ),
        np.array([[[1]]], dtype=np.int32),
    ]
    for i in range(len(arrays)):
        with subtests.test(msg=f"test array {i}", i=i):
            test_array = arrays[i]
            my_model = UnspecifiedAxisLengthModel(matrix=test_array)
            serialization = my_model.model_dump()
            assert "matrix" in serialization
            np.testing.assert_array_equal(test_array, serialization["matrix"])


def test_numpy_validation_dimensionality_mismatch() -> None:
    """Mismatch of dimensionality of numpy arrays"""
    test_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    with pytest.raises(ValidationError) as excinfo:
        BasicTestModel(matrix=test_array)
    # assert error message
    expected_msg = (
        "1 validation error for BasicTestModel\nmatrix\n  Mismatched "
        "dimensions: got 3D array, expected 2D array"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_dim_mismatch_generic_axis_length() -> None:
    """Wrong dimension should be caught even for generic axis lengths"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(ValidationError) as excinfo:
        UnspecifiedAxisLengthModel(matrix=test_array)
    # assert error message
    expected_msg = (
        "1 validation error for UnspecifiedAxisLengthModel\nmatrix\n  "
        "Mismatched dimensions: got 2D array, expected 3D array"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_broader_types(subtests: SubTests) -> None:
    """Subclasses of required dtype will pass validation"""
    arrays: list[npt.NDArray[np.generic]] = [
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        np.array([[1, 2], [3, 4]], dtype=np.float64),
        np.array([[1, 2], [3, 4]], dtype=np.float16),
    ]
    for i, array in enumerate(arrays):
        for strict in [True, False]:
            with subtests.test(
                msg=f"{array.dtype}, strict: {strict}", i=i, strict=strict
            ):
                my_model = UnspecificDtypeModel(matrix=array, strict=strict)  # type: ignore
                serialization = my_model.model_dump()
                assert "matrix" in serialization
                assert serialization["matrix"].dtype is array.dtype


def test_numpy_validation_from_sequence(subtests: SubTests) -> None:
    """Test creation and validation from sequence"""
    # simple scenario that works
    for seq_type in (list, tuple):
        with subtests.test(msg=f"{seq_type.__name__}"):
            test_sequence = seq_type([[1, 2], [3, 4]])
            my_model = BasicTestModel(matrix=test_sequence)
            serialization = my_model.model_dump()
            assert "matrix" in serialization
            assert serialization["matrix"].dtype is np.dtype(np.int32)
            assert serialization["matrix"].shape == (2, 2)

            # scenario that has an inhomogeneity
            test_sequence = [[1, 2], [3, 4, 5]]
            with pytest.raises(ValidationError) as excinfo:
                BasicTestModel(matrix=test_sequence)
            expected_msg = (
                "1 validation error for BasicTestModel\nmatrix\n  Received "
                "sequence has inhomogeneous part or invalid element types. "
                "Original exception:\nsetting an array element with a "
                "sequence. The requested array has an inhomogeneous shape "
                "after 1 dimensions. The detected shape was (2,) + "
                "inhomogeneous part."
            )
            assert expected_msg in str(excinfo.value)


def test_numpy_validation_from_mixed_sequences() -> None:
    """Test creation and validation from mixed nested sequences"""
    test_sequence = [(1, 2), (3, 4)]
    my_model = BasicTestModel(matrix=test_sequence)  # type: ignore[arg-type]
    serialization = my_model.model_dump()
    assert "matrix" in serialization
    assert serialization["matrix"].dtype is np.dtype(np.int32)
    assert serialization["matrix"].shape == (2, 2)


def test_numpy_validation_from_sequence_numeric_types() -> None:
    """Test that different numeric types in sequences cause no issues"""
    test_sequence = [(1, 0), (1.5, 0.2)]  # mixed sequence
    my_model = BasicTestModel(matrix=test_sequence)  # type: ignore[arg-type]
    serialization = my_model.model_dump()
    # check that model type is respected
    assert serialization["matrix"].dtype is np.dtype(np.int32)
    expected_result = np.array([[1, 0], [1, 0]])
    np.testing.assert_array_equal(serialization["matrix"], expected_result)

    # test the same, but for a model with a floating point type
    new_model = FloatTestModel(matrix=test_sequence)  # type: ignore[arg-type]
    serialization = new_model.model_dump()
    assert serialization["matrix"].dtype is np.dtype(np.float32)
    expected_result = np.array([[1.0, 0.0], [1.5, 0.2]], dtype=np.float32)
    np.testing.assert_array_equal(serialization["matrix"], expected_result)


def test_numpy_validation_from_sequence_incompatible_element_types() -> None:
    """Sequences containing incompatible types raise ValidationError"""
    test_sequence = [(1, 2), ("a", "b")]  # correct shape, mismatched types
    with pytest.raises(ValidationError) as excinfo:
        BasicTestModel(matrix=test_sequence)  # type: ignore[arg-type]
    expected_msg = (
        "1 validation error for BasicTestModel\nmatrix\n  Received sequence "
        "has inhomogeneous part or invalid element types. Original exception:"
        "\ninvalid literal for int() with base 10: 'a'"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_new_type_as_axis_length(subtests: SubTests) -> None:
    """Test that int-based new types can be used as axis length"""
    arrays = [
        np.array([[1, 2], [3, 4]], dtype=np.int32),  # shape (2, 2)
        np.ones((3, 3), dtype=np.int32),  # shape (3, 3)
        np.zeros((4, 4), dtype=np.int32),  # shape (4, 4)
        np.array([[1]], dtype=np.int32),  # shape (1, 1)
    ]
    for i in range(len(arrays)):
        with subtests.test(msg=f"test array {i}", i=i, shape=arrays[i].shape):
            test_array = arrays[i]
            my_model = CustomAxisTypeModel(matrix=test_array)
            serialization = my_model.model_dump()
            assert "matrix" in serialization
            np.testing.assert_array_equal(test_array, serialization["matrix"])


def test_numpy_validation_new_type_as_axis_length_dims() -> None:
    """Axis length is respected even when using NewType as axis length"""
    test_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    with pytest.raises(ValidationError) as excinfo:
        CustomAxisTypeModel(matrix=test_array)
    expected_msg = (
        "1 validation error for CustomAxisTypeModel\nmatrix\n  Mismatched "
        "dimensions: got 3D array, expected 2D array"
    )
    assert expected_msg in str(excinfo.value)


def test_numpy_validation_new_type_as_axis_length_shape(
    subtests: SubTests,
) -> None:
    """Same NewType twice is interpreted as axes of same length"""
    arrays = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),  # shape (2, 3)
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),  # shape (3, 2)
    ]
    for i in range(len(arrays)):
        with subtests.test(msg=f"test array {i}", i=i, shape=arrays[i].shape):
            test_array = arrays[i]
            with pytest.raises(ValidationError) as excinfo:
                CustomAxisTypeModel(matrix=test_array)
            expected_msg = (
                f"1 validation error for CustomAxisTypeModel\nmatrix\n  "
                f"Invalid axis length: Axis 1 is typed to be of same length "
                f"as axis 0. Expected length {2 + i}, got {3 - i}. Expected "
                f"shape: ('AxisLen', 'AxisLen')"
            )
            assert expected_msg in str(excinfo.value)


def test_numpy_validation_new_type_as_axis_length_error_msg() -> None:
    """Test accurate error message for more complex shapes"""
    # correct case
    test_array = np.ones((4, 1, 2, 1, 2), dtype=np.int32)
    my_model = ComplexAxisLengthModel(matrix=test_array)
    serialization = my_model.model_dump()
    np.testing.assert_equal(test_array, serialization["matrix"])

    # singular invalid case
    test_array = np.ones((4, 1, 2, 1, 3), dtype=np.int32)
    with pytest.raises(ValidationError) as excinfo:
        ComplexAxisLengthModel(matrix=test_array)
    expected_msg = (
        "1 validation error for ComplexAxisLengthModel\nmatrix\n  Invalid "
        "axis length: Axis 4 is typed to be of same length as axis 2. "
        "Expected length 2, got 3. Expected shape: ('int', 'AxisLen', "
        "'AxisLen2', 'AxisLen', 'AxisLen2')"
    )
    assert expected_msg in str(excinfo.value)

    # double invalid case
    test_array = np.ones((4, 1, 2, 3, 4), dtype=np.int32)
    with pytest.raises(ValidationError) as excinfo:
        ComplexAxisLengthModel(matrix=test_array)
    expected_msg_part_one = (
        "2 validation errors for ComplexAxisLengthModel\nmatrix\n  Invalid "
        "axis length: Axis 3 is typed to be of same length as axis 1. "
        "Expected length 1, got 3. Expected shape: "
        "('int', 'AxisLen', 'AxisLen2', 'AxisLen', 'AxisLen2')"
    )
    assert expected_msg_part_one in str(excinfo.value)
    expected_msg_part_two = (
        "matrix\n  Invalid axis length: Axis 4 is typed to be of same length "
        "as axis 2. Expected length 2, got 4. Expected shape: "
        "('int', 'AxisLen', 'AxisLen2', 'AxisLen', 'AxisLen2')"
    )
    assert expected_msg_part_two in str(excinfo.value)


def test_numpy_array_json_schema_model_dump() -> None:
    """Model dump should contain array as nested list"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    my_model = BasicTestModel(matrix=test_array)
    model_dump = my_model.model_dump_json()
    expected = '{"matrix":[[1,2],[3,4]]}'
    assert expected == model_dump


def test_numpy_array_json_schema_validation_from_json() -> None:
    """Validate model from JSON data directly"""
    # from valid JSON data
    json_data = '{"matrix": [[1, 2], [3, 4]]}'
    test_model = BasicTestModel.model_validate_json(json_data)
    serialization = test_model.model_dump()
    assert "matrix" in serialization
    assert isinstance(serialization["matrix"], np.ndarray)
    assert (2, 2) == serialization["matrix"].shape

    # from invalid JSON data (shape mismatched)
    json_data = '{"matrix": [1, 2, 3, 4]}'
    with pytest.raises(ValidationError) as excinfo:
        BasicTestModel.model_validate_json(json_data)
    expected_msg_pt1 = (
        "2 validation errors for BasicTestModel\nmatrix\n  Mismatched "
        "dimensions: got 1D array, expected 2D array"
    )
    assert expected_msg_pt1 in str(excinfo.value)
    expected_msg_pt2 = (
        "matrix\n  Mismatched shapes: got shape (4,), expected (2, 2)"
    )
    assert expected_msg_pt2 in str(excinfo.value)


def test_numpy_array_json_schema_from_own_serialization() -> None:
    """Check consistency: should be able to use own serialization"""
    test_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
    my_model = BasicTestModel(matrix=test_array)
    model_dump = my_model.model_dump_json()
    new_model = BasicTestModel.model_validate_json(model_dump)
    serialization = new_model.model_dump()
    assert "matrix" in serialization
    assert isinstance(serialization["matrix"], np.ndarray)
    assert (2, 2) == serialization["matrix"].shape
