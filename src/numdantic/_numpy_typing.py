# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Type definitions for numpy array typing.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias, TypeVar, TypeVarTuple

import numpy as np

from ._numpy_validation import NDArrayPydanticAnnotation

# Basic types for static typing with numpy
_ShapeTypeVarTuple = TypeVarTuple("_ShapeTypeVarTuple")
_ScalarTypeVar = TypeVar("_ScalarTypeVar", bound=np.generic, covariant=True)

# Accessible types
Shape: TypeAlias = tuple[*_ShapeTypeVarTuple]
ShapeLike: TypeAlias = tuple[int, ...]  # generic shape-like
_ShapeType = TypeVar("_ShapeType", bound=ShapeLike, covariant=True)

# define a new type alias
NDArray: TypeAlias = Annotated[
    np.ndarray[_ShapeType, np.dtype[_ScalarTypeVar]],
    NDArrayPydanticAnnotation,
]
