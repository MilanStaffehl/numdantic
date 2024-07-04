# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Type definitions and core schema for numpy array typing and validation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    LiteralString,
    NoReturn,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    get_args,
    get_origin,
)

import numpy as np
import numpy.typing
from pydantic_core import (
    InitErrorDetails,
    PydanticCustomError,
    ValidationError,
    core_schema,
)

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

# Basic types for static typing with numpy
_ShapeTypeVarTuple = TypeVarTuple("_ShapeTypeVarTuple")
_ScalarTypeVar = TypeVar("_ScalarTypeVar", bound=np.generic, covariant=True)

# Accessible types
Shape: TypeAlias = tuple[*_ShapeTypeVarTuple]
_ShapeTypeVar = TypeVar("_ShapeTypeVar", bound=Shape)  # type: ignore
ShapeLike: TypeAlias = tuple[int, ...]  # generic shape-like

_ExtendedShapeLike: TypeAlias = tuple[int | str, ...]
_GenericNDArrayType: TypeAlias = numpy.typing.NDArray[np.generic]


# pydantic validation for arrays
def _raise_validation_error(
    x: _GenericNDArrayType,
    stack: list[PydanticCustomError],
    model_name: str | None,
) -> NoReturn:
    """
    Raise a validation error built from the list of errors.

    This function takes a stack of PydanticCustomErrors and converts
    them into one canonic pydantic ``ValidationError`` which is then
    raised. The error will contain accurate information about where
    and why the error occurred.

    :param x: The input that failed its validation and thus caused the
        ValidationError.
    :param stack: A list of PydanticCustomError instances which holds
        information about the validation failures.
    :param model_name: The name of the model or the validator in which
        the validation failed and from which the error will be raised.
        Can be set to None, in which case the location of the error
        will be left unspecified and instead the ValidationError will
        read ``validation error for NDArray validation``.
    :raises ValidationError: Always raised, built from the input.
    :return: Never returns, always raises ValidationError.
    """
    if model_name is None:
        model_name = "NDArray validation"

    init_error_details = []
    for error in stack:
        init_error_details.append(InitErrorDetails(type=error, input=x))
    raise ValidationError.from_exception_data(model_name, init_error_details)


def _validate_array_shape(
    x: _GenericNDArrayType,
    expected_shape: _ExtendedShapeLike,
    error_stack: list[PydanticCustomError],
) -> _GenericNDArrayType:
    """
    Validate that the array ``x`` has the ``expected_shape``.

    Function checks that the array ``x`` has the dimensionality and
    shape dictated by the ``expected_shape`` tuple of either int-bound
    types or Literals. This includes the dimensionality, the axis length
    of the individual axes (if provided by the expected shape), and the
    ratio of axes of the same type when specified using a subtype of
    ``int`` (i.e. if two axes are typed to have the same length, the
    function will check that they indeed have the same length).

    If the array does not meet the requirements, the function appends
    an appropriate ``PydanticCustomError`` to the ``error_stack``. This
    enables gathering multiple errors before finally raising a pydantic
    ``ValidationError`` with the list and :func:`_raise_validation_error`.

    Note that the expected shape is a tuple of either integers or strings,
    where every entry of the tuple corresponds to the length of the axis
    it corresponds to. The entries of the tuple can be either be an actual
    axis length (i.e. integers) or strings, which signify an axis of
    unspecified length. In the latter case, strings that are occurring
    multiple times will cause this function to check that the
    corresponding axes are of the same length. The only exception to
    this is if the string is ``int``, in which case the axis length is
    not checked at all.

    :param x: The array to validate. Must be a numpy array.
    :param expected_shape: The shape that the array is supposed to have.
        This must be a list of integers (for axes with a specific length)
        or strings (for axes of unspecified length).
    :param error_stack: A list of ``PydanticCustomError``s, onto which
        new errors can be appended when they are found.
    :return: The array ``x``.
    """
    # validate dimensionality
    if len(expected_shape) != len(x.shape):
        error_ctx = {
            "received_dim": len(x.shape),
            "expected_dim": len(expected_shape),
        }
        error_stack.append(
            PydanticCustomError(
                "array_dimensions",
                "Mismatched dimensions: got {received_dim}D array, "
                "expected {expected_dim}D array",
                error_ctx,
            )
        )

    # validate shape
    viewed_axes: dict[str, int] = {}
    for i in range(len(expected_shape)):
        if not _validate_array_axis(
            expected_shape, x.shape, i, viewed_axes, error_stack
        ):
            break
    return x


def _validate_array_axis(
    expected_shape: _ExtendedShapeLike,
    received_shape: tuple[int, ...],
    axis_index: int,
    known_axes: dict[str, int],
    error_stack: list[PydanticCustomError],
) -> bool:
    """
    Validate the length of a given axis.

    The function will take the expected and received shapes and will
    compare the axis of the specified axis index. In order to be able to
    check axis lengths of named axes, the function also requires a dict
    mapping axes names to their expected lengths. If validation fails,
    the function creates an appropriate ``PydanticCustomError`` and
    appends it to the given error stack.

    Note that the expected shape is a tuple of either integers, which
    can be used to specify explicit axis length, or strings which can
    be used to ensure that axes of the same name have the same length,
    namely the length that the ``known_axes`` dictionary maps the name
    to. Axes named 'int' are ignored.

    If the function encounters a named axis that is not in ``known_axes``,
    it will update the dictionary with the name and length of the axis.

    :param expected_shape: Tuple of integers or strings which details
        the expected shape of the array. Integers are used for explicit
        axis lengths, strings are used for axes that must have the same
        length.
    :param received_shape: The actual shape of the received array.
    :param axis_index: The axis index which to check for.
    :param known_axes: A dictionary mapping the names of named axes to
        their respective lengths. If the function encounters a named
        axis that is not in ``known_axes``, it will update the dictionary
        with name and length of that axis.
    :param error_stack: A list of ``PydanticCustomError`` instances.
        When the function encounters a validation problem, an appropriate
        error is appended to this list.
    :return: Whether validation should be terminated immediately after
        this axis. This is useful to immediately raise gathered errors
        from ``error_stack``.
    """
    axis_len_or_name = expected_shape[axis_index]
    # axis of specific length
    if isinstance(axis_len_or_name, int):
        if received_shape[axis_index] != axis_len_or_name:
            error_ctx = {
                "received_shape": received_shape,
                "expected_shape": expected_shape,
            }
            error_stack.append(
                PydanticCustomError(
                    "array_shape",
                    "Mismatched shapes: got shape {received_shape}, "
                    "expected {expected_shape}",
                    error_ctx,
                )
            )
            return False  # avoid adding same error multiple times
        return True

    # named axes
    if axis_len_or_name == "int":
        # skip over axes of unspecified length
        return True
    # check if named axis length has been determined before
    elif axis_len_or_name in known_axes.keys():
        if received_shape[axis_index] != known_axes[axis_len_or_name]:
            # update error context with more info
            expected_len = known_axes[axis_len_or_name]
            axis_len_ctx = {
                "failed_at_index": axis_index,
                "first_occurrence": list(received_shape).index(expected_len),
                "expected_len": expected_len,
                "received_len": received_shape[axis_index],
                "expected_shape": expected_shape,
            }
            error_stack.append(
                PydanticCustomError(
                    "array_axis_length",
                    "Invalid axis length: Axis {failed_at_index} is "
                    "typed to be of same length as axis {first_occurrence}. "
                    "Expected length {expected_len}, got {received_len}. "
                    "Expected shape: {expected_shape}",
                    axis_len_ctx,
                )
            )
    else:
        known_axes.update({axis_len_or_name: received_shape[axis_index]})
    return True


def _validate_array_dtype(
    x: _GenericNDArrayType,
    expected_dtype: np.generic,
    error_stack: list[PydanticCustomError],
    strict: bool,
) -> _GenericNDArrayType:
    """
    Validate that the array ``x`` has the ``expected_dtype``.

    Function checks that the array ``x`` has the dtype ``expected_dtype``
    and, depending on whether strict mode is active, either attempt to
    cast wrong dtypes to the expected dtype (``strict = False``). If
    the cast fails or ``strict = True`` was set, the function will
    append a  ``PydanticCustomError`` with information on the error to
    the ``error_stack``. This enables gathering multiple errors before
    finally raising a pydantic ``ValidationError`` with the list and
    :func:`_raise_validation_error`.

    :param x: The array to validate. Must be a numpy array.
    :param expected_dtype: The dtype that the array is supposed to have.
        This must be a numpy dtype.
    :param error_stack: A list of ``PydanticCustomError``s, onto which
        new errors can be appended when they are found.
    :param strict: Whether to attempt casting to ``expected_dtype`` if
        the dtype does not match it, or append a ``PydanticCustomError``
        to the ``error_stack`` immediately and skip all further validation.
        Set to True for immediate failure, and to False to enable casting.
    :return: The array ``x``, possibly cast to ``expected_dtype``.
    """
    error_ctx = {
        "received_dtype": str(x.dtype),
        "expected_dtype": expected_dtype.__name__,  # type: ignore
    }
    # validate dtype
    if not np.issubdtype(x.dtype.type, expected_dtype):
        if strict:
            # raise immediately
            error_stack.append(
                PydanticCustomError(
                    "array_dtype",
                    "Mismatched dtypes: got {received_dtype}, expected "
                    "{expected_dtype}",
                    error_ctx,
                )
            )
        # attempt to cast
        elif not np.can_cast(x.dtype, expected_dtype):
            error_stack.append(
                PydanticCustomError(
                    "array_dtype",
                    "Mismatched dtypes: cannot safely cast "
                    "{received_dtype} to {expected_dtype}",
                    error_ctx,
                )
            )
        else:
            # safe to cast, cast to expected dtype
            x = x.astype(expected_dtype)
    return x


def _get_array_validator(
    expected_shape: _ExtendedShapeLike,
    expected_dtype: np.generic,
    strict: bool,
    model_name: str | None,
) -> Callable[[_GenericNDArrayType], _GenericNDArrayType]:
    """
    Return a validation function for numpy arrays.

    The function wraps the validation function that can validate a
    numpy array for shape ``real_shape`` and dtype ``real_type``. The
    function returned takes only the array and validates it according
    to the specified shape and type. If shape, dimensionality or dtype
    do not match, the function gathers all information about what is
    wrong and finally raises a ValidationError.

    Since the validator is supposed to behave differently in strict and
    lax mode, the function can be returned with either the behavior for
    strict mode (failing if the dtype is not ``expected_type``), or
    with the behavior for lax mode (silently casting the dtype to the
    ``expected_type`` if safely possible, and only failing if this is
    not possible. This behavior is controlled with the ``strict``
    parameter: if set to True, the returned function will behave as a
    validator for strict mode, otherwise it will behave as a validator
    for lax mode.

    :param expected_shape: The shape of the array that is expected.
        This must be a tuple of integers or None. Integers denote axis
        lengths that must be matched exactly, while None can be used
        for undetermined axis length. The length of the tuple determines
        the dimensionality of the array which must be met.
    :param expected_dtype: The dtype of the array as a subtype of
        ``numpy.generic``. This can be a generic type such as
        ``numpy.floating``; the validation function will validate that
        the actual dtype of an array passed to it is a subtype of this
        type.
    :param strict: Whether to return a schema for strict mode (in which
        any dtype that is not a valid subdtype of the expected dtype
        will raise an exception) or in lax mode (in which case mismatched
        dtypes will be cast to the expected type).
    :param model_name: The name of the model which uses this validator.
        This will be used as the title of any ``ValidationError`` raised
        by this function if validation fails. Optional, if set to None,
        the title will be "NDArray validation".
    :return: A callable that takes an actual numpy array and validates
        that it has the given dimensionality, shape, and dtype.
    """

    def validate_array(x: _GenericNDArrayType) -> _GenericNDArrayType:
        """
        Validate the given array to have the expected shape and dtype.

        The given array will be checked for its dimensionality, shape,
        and dtype. Dimensionality must match exactly, shape must match
        only for those axes where an explicit length is required. The
        dtype of the array must be a valid subtype of the expected type.

        If any of these requirements are not met, the function will
        gather them and eventually raise a pydantic ``ValidationError``
        including information on all conditions that were not met.

        If ``strict`` is set to True in the wrapping function, then a
        mismatch in dtype will always raise a ValidationRrror. Otherwise,
        a safe type cast will be attempted and a ValidationError is only
        raised if casting is unsafe.

        :param x: The array to validate. Must be a numpy array.
        :raises ValidationError: If the array does not meet the
            expectations on dimensionality, shape, and dtype.
        :return: The array, provided the validation is successful.
        """
        # collect validation errors until validation is complete
        validation_errors: list[PydanticCustomError] = []

        # validate array
        x = _validate_array_shape(x, expected_shape, validation_errors)
        x = _validate_array_dtype(x, expected_dtype, validation_errors, strict)

        # check if there were any errors
        if validation_errors:
            _raise_validation_error(x, validation_errors, model_name)
        return x

    return validate_array


def _get_cast_function(
    real_type: np.generic,
) -> Callable[[Sequence[Any] | _GenericNDArrayType], _GenericNDArrayType]:
    """
    Return a function that may cast a sequence to a numpy array.

    The returned function will accept as its only argument a sequence.
    It will try to convert it to a numpy array of the given data type.
    Shape is inferred from the shape of the input sequence.

    :param real_type: The dtype of the array that the returned function
        should produce.
    :return: A function that takes as argument a sequence and turns it
        into a numpy array of dtype ``real_type``.
    """

    def cast_to_array(
        x: Sequence[Any] | _GenericNDArrayType,
    ) -> _GenericNDArrayType:
        """
        Turn given sequence into a numpy array.

        The function takes an arbitrary sequence, possible including
        other (nested) sequences, and turns it into a numpy array of
        the dtype that was fixed by the wrapper function
        :func:`_get_cast_schema`. The shape of the array is determined
        from the structure of the (nested) sequence, i.e. the structure
        of the sequence will be retained as the array shape if possible.
        If the sequence cannot be turned into an array due to
        inhomogeneity, the function raises a pydantic ``ValidationError``.

        When given an array, this array is returned as-is immediately.

        :param x: A sequence or nested sequence of numbers, strings or
            bytes, or an array.
        :raises PydanticCustomError: If ``x`` is neither a sequence nor
            an array.
        :return: A numpy array created from the sequence.
        """
        if isinstance(x, np.ndarray):
            return x
        if not isinstance(x, Sequence):
            msg: LiteralString = (
                "Input must be a sequence or a numpy array, received "
                "{type_x} instead"
            )
            raise PydanticCustomError(
                "input_type", msg, {"type_x": type(x).__name__}
            )
        try:
            return np.array(x, dtype=real_type)
        except ValueError as exc:
            raise PydanticCustomError(
                "from_sequence",
                "Received sequence has inhomogeneous part or invalid element "
                "types. Original exception:\n{exc}",
                {"exc": str(exc)},
            )

    return cast_to_array


class _NDArrayPydanticAnnotation:
    """
    Annotation class enabling numpy array validation in pydantic.

    This class can be annotated to the NDArray-type of the numdantic
    library, to enable validation of numpy arrays with pydantic:

    .. code:: python

        typing.Annotated[NDArray, _NDArrayPydanticAnnotation]

    When supplied to a pydantic ``BaseModel``, this annotated type wil
    correctly validate an array passed to it according to the type
    provided:

    >>> from numdantic import NDArray, Shape
    >>> from pydantic import BaseModel
    >>> import numpy as np
    >>> class MatrixModel(BaseModel):
    ...     matrix: NDArray[Shape[int, int], np.int32]
    ...
    >>> x = np.array([[1, 2], [3, 4]])
    >>> my_model = MatrixModel(matrix=x)
    >>> my_model.model_dump()
    {"matrix": array([[1, 2], [3, 4]])}

    Additionally, the annotation will allow pydantic to coerce the data
    type of the array to the correct type, provided this will not cause
    the loss of information (i.e. when converting floating point numbers
    to integers).
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        Return pydantic core schema for numpy array validation.

        :param _source_type: The type given in the model.
        :param _handler: Handler to call into the next CoreSchema
            schema generation function.
        :return: Pydantic core schema for numpy array validation.
        """
        # get shape as actual tuple:
        shape_type = get_args(_source_type)[0]
        expected_shape = tuple(
            [
                get_args(x)[0] if get_origin(x) == Literal else x.__name__
                for x in get_args(shape_type)
            ]
        )
        # similarly, get the dtype:
        dtype_type = get_args(get_args(_source_type)[1])[0]

        # construct validator functions
        cast_func = _get_cast_function(dtype_type)
        validator_lax = _get_array_validator(
            expected_shape, dtype_type, False, _handler.field_name
        )
        validator_strict = _get_array_validator(
            expected_shape, dtype_type, True, _handler.field_name
        )

        # construct validator schema
        array_validator_schema = core_schema.lax_or_strict_schema(
            lax_schema=core_schema.no_info_plain_validator_function(
                validator_lax
            ),
            strict_schema=core_schema.no_info_plain_validator_function(
                validator_strict
            ),
        )

        # construct final schema
        array_schema = core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(cast_func),
                array_validator_schema,
            ]
        )

        # serialization to JSON format
        json_serializer = core_schema.plain_serializer_function_ser_schema(
            lambda x: x.tolist(), when_used="json"
        )

        # build the core schema and return it
        return core_schema.json_or_python_schema(
            python_schema=array_schema,
            json_schema=array_schema,
            serialization=json_serializer,
        )


# define a new type alias
NDArray: TypeAlias = Annotated[
    np.ndarray[_ShapeTypeVar, np.dtype[_ScalarTypeVar]],
    _NDArrayPydanticAnnotation,
]
