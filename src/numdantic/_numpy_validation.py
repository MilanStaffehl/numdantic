# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Pydantic core schema for numpy array validation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    LiteralString,
    NoReturn,
    TypeAlias,
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
    from types import GenericAlias

    from pydantic import GetCoreSchemaHandler


# Types
_ExtendedShapeLike: TypeAlias = tuple[int | str, ...]
_GenericNDArrayType: TypeAlias = numpy.typing.NDArray[np.generic]


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


def _validate_array_indeterminate_shape(
    x: _GenericNDArrayType,
    first_axis_len_or_type: str | int,
    error_stack: list[PydanticCustomError],
) -> _GenericNDArrayType:
    """
    Validate that shape of array ``x`` with indeterminate dimensions.

    Function checks that the array ``x``, typed to have indeterminate
    dimensionality, has the shape dictated by ``first_axis_len_or_type``
    which can either be the name of the axis type or the integer length
    that all axes must have. The function accepts arbitrary dimensions
    for ``x``, but it will check that when ``first_axis_len_or_type`` is
    an integer, that all axes have that exact length, and when it is a
    type name that is not ``"int"``, that all axes have the same length
    as the first axis. If ``first_axis_len_or_type`` is ``"int"``, the
    function treats the shape of the array as completely arbitrary and
    returns it as-is without verification.

    If the array does not meet the requirements, the function appends
    an appropriate ``PydanticCustomError`` to the ``error_stack``. This
    enables gathering multiple errors before finally raising a pydantic
    ``ValidationError`` with the list and :func:`_raise_validation_error`.

    Note that there are only three recognized cases:

    - ``first_axis_len_or_type`` is ``"int"``: No validation.
    - ``first_axis_len_or_type`` is any integer: All axes must have this
      length; dimensionality is not checked.
    - ``first_axis_len_or_type`` is any string other than ``"int"```:
      All axes must have the same length as the first axis.

    Also note that this function makes no attempt to check that ``x``
    is actually typed as having indeterminate shape. This must happen
    prior to calling this function.

    :param x: The array to validate. Must be a numpy array.
    :param first_axis_len_or_type: The type name or literal length that
        is expected for the first axis of ``x``. This can, but does not
        have to be, the actual type or length of the first axis of ``x``.
        When this is an integer, all axes of ``x`` wil be checked to
        have this length. When this is any string except ``"int"``, all
        axes of ``x`` will be checked to have the same length as the
        first axis of ``x``. If this is the string literal ``"int"``,
        then ``x`` is returned without verification.
    :param error_stack: A list of ``PydanticCustomError``s, onto which
        new errors can be appended when they are found.
    :return: The array ``x``.
    """
    # validate dimensionality
    if first_axis_len_or_type == "int":
        # dimensionality and shape arbitrary; no validation possible
        return x
    elif isinstance(first_axis_len_or_type, int):
        # literal axis length, all axes must have this length
        expected_shape = tuple([first_axis_len_or_type for _ in x.shape])
        err_msg = (
            f"All axes were typed to have fixed length "
            f"{first_axis_len_or_type}. Expected shape: {expected_shape}, "
            f"actual shape: {x.shape}."
        )
    else:
        # named axes, all axes must have length of first axis
        expected_shape = tuple([x.shape[0] for _ in x.shape])
        err_msg = (
            f"All axes were typed to have length {x.shape[0]} of first axis. "
            f"Expected shape: {expected_shape}, actual shape: {x.shape}."
        )

    if not x.shape == expected_shape:
        error_stack.append(
            PydanticCustomError(
                "array_dimensions",
                "Mismatched dimensions: {err_msg}",
                {"err_msg": err_msg},
            )
        )
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
    not possible). This behavior is controlled with the ``strict``
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
    :param strict: Whether to return a schema for strict mode or lax
        mode. In strict mode, any dtype that is not a valid subdtype of
        the expected dtype will raise an exception. In lax mode,
        mismatched dtypes will be cast to the expected type.
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
        if expected_shape[1] == "...":
            x = _validate_array_indeterminate_shape(
                x, expected_shape[0], validation_errors
            )
        else:
            x = _validate_array_shape(x, expected_shape, validation_errors)
        x = _validate_array_dtype(x, expected_dtype, validation_errors, strict)

        # check if there were any errors
        if validation_errors:
            _raise_validation_error(x, validation_errors, model_name)
        return x

    return validate_array


def _get_cast_function(
    real_type: np.generic,
    strict: bool = False,
) -> Callable[[Sequence[Any] | _GenericNDArrayType], _GenericNDArrayType]:
    """
    Return a function that may cast a sequence to a numpy array.

    The returned function will accept as its only argument a sequence.
    It will try to convert it to a numpy array of the given data type.
    Shape is inferred from the shape of the input sequence.

    :param real_type: The dtype of the array that the returned function
        should produce.
    :param strict: Whether to return a cast function for strict mode or
        lax mode. In strict mode, any type except a numpy array will
        raise a ValidationError. In lax mode, sequences are cast into
        a numpy array of the correct dtype.
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

    def disallow_casting(
        x: Sequence[Any] | _GenericNDArrayType,
    ) -> _GenericNDArrayType:
        """
        Return the given object only if it is a numpy array.

        The function takes an arbitrary sequence, possibly including
        other (nested) sequences, or a numpy array and rejects any
        input that is not a numpy array by raising a ValidationError.
        When given an array, this array is returned as-is immediately.

        :param x: A sequence or nested sequence of numbers, strings or
            bytes, or an array.
        :raises ValidationError: If ``x`` is not a numpy array.
        :return: A numpy array created from the sequence.
        """
        if not isinstance(x, np.ndarray):
            msg: LiteralString = (
                "Input must be a numpy array in strict mode, received "
                "{type_x} instead."
            )
            raise PydanticCustomError(
                "input_type", msg, {"type_x": type(x).__name__}
            )
        # input is already a numpy array, return as-is
        return x

    if strict:
        return disallow_casting
    return cast_to_array


def _transform_shape_type(shape_type: GenericAlias) -> _ExtendedShapeLike:
    """
    Transform a generic alias for shape into a tuple of str and int.

    Function takes a shape tuple as it appears in a type annotation of
    a pydantic model field (which is a tuple of types, including ``int``,
    new types based on ``int``, ``Literal`` integers, and possibly the
    Python built-in ``Ellipsis`` type. It converts this tuple into a
    tuple of strings and integers, where literal integers are converted
    to their integer value, and all other types are replaced with their
    type name (e.g. ``int``, ``...`` for ellipses, etc.).

    :param shape_type: Tuple of types as they appear in a shape annotation.
        Can include ``int```, new types based on ``int``, integer literals,
        and ``...`` (built-in ``Ellipsis`` type).
    :return: The tuple of types as string of their names, and as integers
        for integer literals.
    """
    extended_shape = []
    for x in get_args(shape_type):
        if get_origin(x) is Literal:
            extended_shape.append(get_args(x)[0])
        elif x is Ellipsis:
            extended_shape.append("...")
        else:
            extended_shape.append(x.__name__)
    return tuple(extended_shape)


class NDArrayPydanticAnnotation:
    """
    Annotation class enabling numpy array validation in pydantic.

    This class can be annotated to the NDArray-type of the numdantic
    library, to enable validation of numpy arrays with pydantic:

    .. code:: python

        typing.Annotated[NDArray, _NDArrayPydanticAnnotation]

    When supplied to a pydantic ``BaseModel``, this annotated type wil
    correctly validate an array passed to it according to the type
    provided:

    >>> from numdantic import NDArray
    >>> from pydantic import BaseModel
    >>> import numpy as np
    >>> class MatrixModel(BaseModel):
    ...     matrix: NDArray[tuple[int, int], np.int32]
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
        expected_shape = _transform_shape_type(shape_type)
        # similarly, get the dtype:
        dtype_type = get_args(get_args(_source_type)[1])[0]

        # construct casting and validator functions
        cast_func_lax = _get_cast_function(dtype_type, False)
        cast_func_strict = _get_cast_function(dtype_type, True)
        validator_lax = _get_array_validator(
            expected_shape, dtype_type, False, _handler.field_name
        )
        validator_strict = _get_array_validator(
            expected_shape, dtype_type, True, _handler.field_name
        )

        # construct combined casting and validator schema
        lax_schema = core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(cast_func_lax),
                core_schema.no_info_plain_validator_function(validator_lax),
            ]
        )
        strict_schema = core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(cast_func_strict),
                core_schema.no_info_plain_validator_function(validator_strict),
            ]
        )

        # construct final schema
        array_schema = core_schema.lax_or_strict_schema(
            lax_schema=lax_schema, strict_schema=strict_schema
        )

        # TODO: write proper serialization and deserialization schemas (#86)
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
