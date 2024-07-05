# numdantic

Typing support for [`numpy`](https://numpy.org/) arrays, compatible with [`pydantic`](https://docs.pydantic.dev/latest/) validation out-of-the-box.

|     |     |
| --- | :-: |
| CI/CD | [![Build](https://github.com/MilanStaffehl/numdantic/actions/workflows/publish_release.yml/badge.svg)](https://github.com/MilanStaffehl/numdantic/actions/workflows/publish_release.yml) [![Tests](https://github.com/MilanStaffehl/numdantic/actions/workflows/tests.yml/badge.svg)](https://github.com/MilanStaffehl/numdantic/actions/workflows/tests.yml) [![Code Quality](https://github.com/MilanStaffehl/numdantic/actions/workflows/linting.yml/badge.svg)](https://github.com/MilanStaffehl/numdantic/actions/workflows/linting.yml) ![Covergae badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/MilanStaffehl/84965933a22ab4f94b02d8563982025d/raw/926382768a39dfd4029b26bdbecb9cab1acdff0d/numdantic_coverage.json) |
| PyPI | ![PyPI - Version](https://img.shields.io/pypi/v/numdantic?logo=pypi&logoColor=yellow&label=PyPI&color=blue) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/numdantic?logo=Python&logoColor=yellow&label=Python&color=blue) |
| Dev | [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |

## Table of contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Concepts](#concepts)
  - [Axes of unspecified length](#axes-of-unspecified-length)
  - [Axes of specific length](#axes-of-specific-length)
  - [Named axes](#named-axes)
  - [Generic scalar types](#generic-scalar-types)
  - [Behavior in lax vs. strict mode](#behavior-in-lax-vs-strict-mode)
  - [Arrays from sequences](#arrays-from-sequences)
- [Limitations](#limitations)
  - [Wrong shapes or dtypes in assignments](#wrong-shapes-or-dtypes-in-assignments)
  - [Mixing of named axes and generic axes](#mixing-of-named-axes-and-generic-axes)
  - [Using Python built-ins as dtype](#using-python-built-ins-as-dtype)
- [Tips & tricks](#tips--tricks)
- [Alternatives](#alternatives)
- [Contributing](#contributing)
  - [Code of conduct](#code-of-conduct)
  - [Bug reports](#bug-reports)
  - [Feature requests](#feature-requests)
  - [Pull requests](#pull-requests)
- [Authors](#authors)

## About

`numpy` is a widely used Python library for data science and numerical computation. With the Python typing system becoming increasingly popular, the need for proper ways to type `numpy` arrays is rising. While the developers of `numpy` are dedicated to making their own typing system for their library, development is slow and the currently available solutions do not utilize the possibilities of the Python typing system to their full extent.

This project aims to do two things:

1. Provide typing support for `numpy` arrays that works with most static type checkers, including shape typing.
2. Provide, using the same types, support for validation of `numpy` arrays with `pydantic`.

## Features

- **Typing support**: Annotate your code with type hints for arrays, including shape and dtype.
- **Type checking**: Compatible with static type checkers such as `mypy` and `pyright`.
- **Validation**: Ready for use in `pydantic` models out-of-the-box! Validation includes:
  - Shape (dimensionality and axis lengths)
  - dtypes
  - Axis length for same-length axes
- **Minimalistic**: There are exactly two types that make `numdantic` work and that you need to learn about - nothing more!

## Installation

`numdantic` requires Python 3.11 or higher to work. To install `numdantic`, simply run:

```shell
pip install numdantic
```

## Usage

To get started, import the `NDArray` and `Shape` types from the package. `NDArray` takes two type parameters: a shape, and a `numpy` scalar type. You can use it to annotate a variable as an array like this:

```Python
import numpy as np
from numdantic import NDArray, Shape

# annotating a 2D array
matrix: NDArray[Shape[int, int], np.int32] = np.random.rand(2, 2)
```

This variable is now typed as a 2D array, with its axes having unspecified length, and of dtype `np.int32`. Static type checker such as `mypy` will now be able to check if you are using the variable correctly. Both shape and dtype are a lot more flexible than shown here though; you can learn about specifying axis lengths and same-length-axes in the [_Concepts_](#concepts) section below.

If you wish to use a `numpy` array inside of a `pydantic` base model, you can use `NDArray` as an annotation for the corresponding field, and it will work out-of-the-box:

```Python
import numpy as np
from numdantic import NDArray, Shape
from pydantic import BaseModel

class MyModel(BaseModel):
    matrix: NDArray[Shape[int, int], np.int32]

# this will pass validation
MyModel(matrix=np.array([[1, 2], [3, 4]], dtype=np.int32))

# this will raise a ValidationError due to wrong dimensions
MyModel(matrix=np.array([1, 2, 3, 4], dtype=np.int32))

# this will raise a ValidationError due to wrong dtype
MyModel(matrix=np.array([[1, 2], [3, 4]], dtype=np.float64))
```

Learn more about what else `numdantic` can do below!

## Concepts

### Axes of unspecified length

If you wish to annotate an array that has a specific dimensionality, but whose axes can have arbitrary length, you can type its shape using the `int` type. The rationale behind this is that array shapes are tuples of integers, and therefore, they are typed as such. For example, you can specify 1D, 2D, and 3D arrays like this:

```Python
import numpy as np
from numdantic import NDArray, Shape

array1d: NDArray[Shape[int], np.int32] = ...
array2d: NDArray[Shape[int, int], np.int32] = ...
array3d: NDArray[Shape[int, int, int], np.int32] = ...
```

Type checkers and `pydantic` will not verify the length of the axes. This means that both an array of shape `(2, 2)` and `(100, 100)` would pass validation as `array2d`, but an array of shape `(1, )` or `(2, 2, 2)` would cause an error due to a mismatch in dimensions.

### Axes of specific length

Often you will know what lengths your axes will have and want to make sure that these axes lengths are respected throughout your program. To type an axis with a specific length, you can use literal integers:

```Python
from typing import Literal as L
import numpy as np
from numdantic import NDArray, Shape

widescreen_image: NDArray[Shape[L[1080], L[720], L[4]], np.int64]
```

Type checkers will accept this notation. Moreover, when used inside of a `pydantic` base model, this will ensure that the axes of the array are checked for their length; if any axis has a length that does not match its annotation, a `ValidationError` will be raised accordingly.

> [!NOTE]
>
> There is one significant caveat to this approach: you cannot mix unspecified axes and axes of specific length, i.e. attempting to assign `widescreen_image` to a variable typed to have shape `Shape[int, int, int]` does not work and will cause type checkers to report an error. Once you opt for literals in your shapes, you will have to commit to them. Learn more about this limitation and why it occurs in the section about [Limitations](#limitations).

### Named axes

When you have axes that carry specific meaning, you often want to give them a specific name or handle. This is often useful for two reasons:

1. Documentation: giving the axis a name makes it easier to understand what the array represents and what each axis means.
2. Type safety: named axes offer the opportunity to detect when an array has its axes in the wrong order.

Borrowing the classic example for such a scenario, let us assume you wish to annotate a frame from a video. You can do so as shown above if you know the exact screen resolution, but often you wish to keep the exact axis length unspecified to support multiple resolutions. You do, however, want to make sure that the data is ordered correctly, to avoid one developer ordering it like `(width, height, RGBA)` and another as `(height, width, RGBA)`, possibly leading to hard-to-track bugs.

You can achieve this in `numdantic` by defining a `NewType` based on `int` and using it as an axis length:

```Python
from typing import NewType
import numpy as np
from numdantic import NDArray, Shape

# named axes
Width = NewType("Width", int)
Height = NewType("Height", int)
RGBAColor = NewType("RGBAColor", int)

# annotate frame
video_frame: NDArray[Shape[Width, Height, RGBAColor], np.int64]
```

This annotation will ensure that the axes are always ordered the correct way. For example, attempting to use `video_frame` in a function that accepts an array typed as `NDArray[Shape[Height, Width, RGBAColor], np.int64]` will cause type checkers to raise an error.

> [!NOTE]
> There is one significant caveat to this approach: you cannot mix unspecified axes and named axes, i.e. attempting to assign `video_frame` to a variable typed to have shape `Shape[int, int, int]` does not work and will cause type checkers to report an error. Once you opt for named axes, you will have to commit to them. Learn more about this limitation and why it occurs in the section about [Limitations](#limitations).

Named axes have a secondary benefit that only comes into play when validating them with `pydantic`: If you use two or more axes of the same name in a base model field annotation, `pydantic` will check that they all have the same length:

```Python
from typing import NewType
import numpy as np
from numdantic import NDArray, Shape

# named axes
Side = NewType("Side", int)

class Square(BaseModel):
    # both axes must have same length
    vertices: NDArray[Shape[Side, Side], np.int32]

# this will work (shape is (2, 2))
Square(vertices=np.array([[1, 2], [1, 2]], dtype=np.int32))

# this will raise a ValidationError (shape is (2, 3))
Square(vertices=np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32))
```

### Mixing of shape types

It is possible and intended that you mix axes of unspecified length, axes of specific length, and named axes within the same array. This is what makes `numdantic` flexible and useful. For example, here is how you could mix the different shape types in a single annotation:

```Python
from typing import NewType, Literal as L
import numpy as np
from numdantic import NDArray, Shape

SomeAxis = NewType("SomeAxis", int)

x: NDArray[Shape[int, SomeAxis, L[2], SomeAxis], np.int32]
```

For each of these axes, type checkers and `pydantic` will apply the rules described above. In a `pydantic` model this would mean that the second and fourth axes of `x` must have the same length and the third axis must be of length 2.

### Generic scalar types

Often, it is not crucial what precision your arrays dtype has. You might not care if your array has dtype `int32` or `int64`, just that it is an integer. Or perhaps you do not care about the dtype at all. For this purpose, `numdantic` allows you to specify broader dtypes in your annotations, using the generic scalar types provided by `numpy`:

```Python
import numpy as np
from numdantic import NDArray, Shape
from typing import Any

# an array that will accept any number
any_number: NDArray[Shape[int, int], np.number[Any]]

# an array that will accept any positive integer
pos_ints: NDArray[Shape[int, int], np.unsignedinteger[Any]]
```

See the `numpy` documentation for [scalar types](https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types) for an overview over what scalar types `numpy` offers.

> [!IMPORTANT]
> As generics, these scalar types must be supplied with a type parameter. This type parameter specifies the size of the type (for example 32 bit vs 64 bit). Type checkers actually convert all scalar types to one of these generics. For example, `numpy.int64` is converted into `numpy.signedinteger[numpy._typing._64Bit]` during type checking. In order to actually receive a type that is agnostic to the size of the dtype, it is required to use `Any` as type parameter - hence the use of `Any` in the example above.

> [!NOTE]
> Depending on your version of `numpy`, using these generic scalar types to *create* arrays may be deprecated. You will get a corresponding runtime warning if you use them to instantiate an array. As a result, you might also get such a warning when you use a scalar generic inside a `pydantic` validator and the validator attempts to create an array using the generic as dtype. In such a case, you can either suppress this warning, or choose another appropriate scalar dtype. The latter is recommended. In *type annotations*, the generic scalars are all fine and should cause no problems.

### Behavior in lax vs. strict mode

`pydantic` can run its validation in two modes: strict and lax mode. Depending on the mode chosen, inputs of a wrong type may be cast to the expected type, if possible (lax mode), or always raise an exception (strict mode). `numdantic` mirrors this behavior:

- In **strict** mode, dtype must match exactly. If the input has a different dtype than the field expects it to, a `ValidationError` is raised. If you have specified only a generic dtype, the input must have a dtype that is a valid subtype of this generic type.
- In **lax** mode, any mismatched dtype that can be safely cast to the target dtype is accepted and cast. Only if casting cannot be safely done, a `ValidationError` will be raised. If you have specified a generic dtype, casting will be performed only if the input has a dtype that is not a subtype of that generic dtype. Casting is performed directly to the generic dtype, meaning the resulting dtype is system-dependent!

Shapes are never cast. This is to avoid hard to track bugs caused by arrays being reshaped into shapes that do not cause runtime issues, but produce wrong results. If an array input has the wrong shape, it will always raise a `ValidationError`.

You can specify in which mode to run validation the usual way, i.e. by setting the mode to `strict` in a model, field, or globally using the `pydantic` API.

### Arrays from sequences

When using `NDArray` as a field type in a `pydantic` base model, you can also pass it (nested) sequences, which are then turned into an array of the specified dtype and validated according to shape and dtype:

```Python
import numpy as np
from pydantic import BaseModel
from numdantic import NDArray, Shape

class MyModel(BaseModel):
    array: NDArray[Shape[int, int], np.int64]

# create a model using a sequence
my_model = MyModel(array=[[1, 2], [3, 4]])
serialization = my_model.model_dump()

# check the output
assert isinstance(my_model["array"], np.ndarray)  # passes
assert my_model["array"].dtype is np.dtype(np.int64)  # passes
```

## Limitations

### Wrong shapes or dtypes in assignments

Due to how `numpy` handles the shape and dtype of its `ndarray` type, static type checkers will unfortunately not be able to detect a wrong shape or a wrong dtype in assignments. For example, the following code will not raise an error when running it through a type checker:

```Python
import numpy as np
from numdantic import NDArray, Shape

# This does not cause a type checking error!
x: NDArray[Shape[int, int], np.int32] = np.array([1, 2, 3], dtype=np.int32)
# This does not cause a type checking error either!
x: NDArray[Shape[int, int], np.int32] = np.array([[1, 2], [3, 4]], dtype=np.float32)
```

This is due to the fact that `numpy` functions usually have return types annotations where the array shape is always `Any`, and the dtype typically is `np.generic`. Some `numpy` functions might have type annotations that accurately represent at least the dtype of their return value, but this is by no means guaranteed. There is nothing much that can be done about this, until `numpy` changes its own typing system to more accurately reflect the return values dtype and shape.

Until then, you will either have to pay close attention to assignments, implement your own typing stubs for `numpy`, or write custom wrappers around `numpy` functions which you can then annotate appropriately, using `TypeVar` and `TypeVarTuple`.

Fortunately, there is a saving grace: type checkers _will_ detect mismatches in types further down the line, for example if you try to use an array typed with `numdantic` as a 2D array in a function that is typed with `numdantic` to only accept 3D arrays, a type checker will catch this mistake.

### Mixing of named axes, literal axes and generic axes

It is not possible to use named axes created with `NewType` in places that are typed using `int`:

```Python
from typing import NewType
import numpy as np
from numdantic import NDArray, Shape

# named axes
Width = NewType("Width", int)
Height = NewType("Height", int)

# annotate image
image: NDArray[Shape[Width, Height], np.int64] = np.random.rand(40, 20)

def transpose_image(
    img: NDArray[Shape[int, int], np.int32]
) -> NDArray[Shape[int, int], np.int32]:
    return img.transpose()

# this will raise am error when checked by a type checker!
transpose_image(image)
```

Similarly, you cannot mix shapes with literal integers such as `Shape[Literal[2], Literal[2]]` with named axes or generic axes typed with `int`. All these combinations will cause your type checker to issue an error.

This happens because the `TypeVarTuple` used to make shapes work within `numdantic` currently has no way of specifying variance; it is invariant in all its entries. As a result, type checkers are not able to accept subtypes of `int` inside of a `Shape` to be valid in place of actual `int` types.

If you wish to enjoy the documentation benefit of named axes and can forgo the benefit of base models checking for axes of the same name having the same length, it is recommended to use type aliases instead:

```Python
from typing import NewType, TypeAlias
import numpy as np
from numdantic import NDArray, Shape

# named axes using type aliases
width: TypeAlias = int
height: TypeAlias = int

# annotate image
image: NDArray[Shape[width, height], np.int64]
```

For Python 3.12+ it is of course recommended to instead use the [new type alias syntax](https://peps.python.org/pep-0695/) instead.

For axes of specific length, you can of course create similar type aliases of specific literals:

```PYthon
Width720: TypeAlias = Literal[720]
```

In the future, `numdantic` might try to solve this issue, but whether that is even feasible is yet to be determined.

### Using Python built-ins as dtype

Currently, `numdantic` does not allow using Python built-in types as dtype for array annotations. This is due to how `numpy` types their `ndarray` type: as dtype, they only allow subtypes of `np.generic`. This is despite the fact that `numpy` has, for quite some time now, also accepted built-in types such as `int` or `float` as dtypes. The reasoning probably is that these built-in types are converted into proper `numpy` dtypes before an `ndarray` is constructed.

In principle, this can be remedied by simply adding built-in types to the allowed dtypes in `numdantic` and then ignoring the complaints that type checkers will have, but this could lead to unforeseen consequences and sort of defeats the whole point of type checking. Therefore, `numdantic` accepts this limitation for now.

## Tips & tricks

Here are some miscellaneous tips and tricks for using `numdantic`:

- Can't figure out which dtypes are compatible with which other dtypes? You can run the helper script `scripts/compatible_dtypes.py` to get a handy table of dtype compatibility! You can even pipe the output into a valid `.rst` file. The table shows what array dtypes (rows) can be assigned to what target dtypes (columns), and it also includes useful information on your platform which can give an insight into the implementation of `numpy` dtypes on your machine.
- To get the most accurate type checking possible, use the [`numpy` plug-in](https://pydoc.dev/numpy/latest/numpy.typing.mypy_plugin.html) for `mypy`. This plug-in will supply `mypy` with implementation details of the different dtypes on your machine and makes type checking of dtypes behave more predictably.

## Alternatives

`numdantic` is _very_ rudimentary. For some projects, that might be just what you need, but if you find that `numdantic`does not fulfill your requirements, you might wish to check out these alternatives. They are much more advanced, support other data structures like pandas data frames as well, and are well maintained. Note however that their typing system differ from that of `numdantic` and therefore are not easily compatible.

- [`numpydantic`](https://github.com/p2p-ld/numpydantic) together with [`nptyping`](https://github.com/ramonhagenaars/nptyping) - Support for `numpy` arrays, `pandas` data frames, `dask` arrays, `hdf5` and `zarr`. Typing is simple and includes shape typing using string literals. Includes JSON schema generation and proper JSON serialization for all data structures.
- [`pydantic-numpy`](https://github.com/caniko/pydantic-numpy) - Support for `numpy` arrays, including loading from .npy and .npz files. Provides type factory for different shapes and dtypes.

## Contributing

`numdantic` is looking for your help! If you find a bug, have a request for a new feature, or wish to contribute, follow these guidelines.

### Code of conduct

This should go without saying, but any interactions regarding `numdantic`, public or private, with the developers, contributors, or users, should be respectful and polite. Any use of aggressive, hateful, sexist, racist, or otherwise derogatory or discriminating language will not be tolerated. Depending on the severity of the infraction, a warning may be issued first, but I reserve the right to block people from participating in the `numdantic` community without warning for severe infractions or if warnings are not leading to a correction of behavior. Actions such as trolling, doxxing, threatening or insulting members of the community will result in an immediate ban from participating in these communities.

Be nice, please. It isn't that hard.

### Bug reports

If you find a bug or something is not working as you would expect, open a bug report on the [GitHub issues](https://github.com/MilanStaffehl/numdantic/issues). Please make sure that your bug has not been reported before. If it has, join the conversation on the existing issue instead. When you open a new issue, make sure to provide a minimal example that is able to reproduce the bug on your machine. This makes fixing the bug much easier.

When you open a new bug report, use the `bug report` issue template and fill out the form as best as you can.

### Feature requests

If you have an idea for a new feature for `numdantic`, you can submit a feature request on the [GitHub issues](https://github.com/MilanStaffehl/numdantic/issues) using the `feature request` template. Fill out the form with your idea and give it an expressive title.

### Pull requests

If you wish to supply an implementation to a feature request or a bug report directly, you can do so by opening a pull request. You can additionally also provide code contributions for open issues that have the `help wanted` label. Go to the issue and comment that you would like to provide an implementation. Write your code on a new branch on a fork of the `numdantic` main repository, and when you are finished, create a pull request to the main repository. Your pull request will then be reviewed and you will receive feedback as soon as possible.

Please note that there are some requirements for your code contributions:

- `numdantic` follows a few code conventions. They are automatically enforced by `pre-commit`. In order to use `pre-commit`, clone the repository, and then run the following commands from the project root to get started:

  ```shell
  pip install -e .[dev]
  pre-commit install
  ```

- All functions, methods, classes and exceptions must have a docstring, describing their use and all parameters. Single-line docstrings are sufficient for tests, but not source code. See the existing docstrings for examples of what a docstring for `numdantic` should look like.

- Your code must be covered by tests. `numdantic` differentiates between unit tests that cover only internal code, and integration tests that also cover the integration with `mympy` and `pydantic`. Depending on your code, you may not be able to provide unit tests, since `numdantic` is strongly integrated with `pydantic` by design. Note that arrays are not patched in unit tests; you may use actual `numpy` arrays for testing.

- `numdantic` uses [conventional commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) with the [angular](https://github.com/angular/angular/) commit types. The possible scopes depend on the commit type:

  - For `feat`, `fix`, `refactor`, and `perf` commit types, scopes can be either `validation`, `typing` or `scripts`, depending on what changed. For `refactor` and `perf` type commits, the scope can additionally also be `tests` if the refactoring touches only test code. Note that fixes for tests belong to the `test` commit type, *not* the `fix` type!
  - For `docs` commit types, the scope should be the name of the file changed. If you worked on multiple files, use one commit per file.
  - For `ci` commit types either drop the scope or use `actions` if you changed something about the GitHub actions.
  - For `test` commit types, use either `unit` or `integration`, depending on which tests you changed. If you change both, split the changes into two commits. Note that fixes for tests also belong to the `test` commit type.
  - For `build` commit types, no scope is required. If your commit touches only the `pyproject.toml`, you can optionally use `pyproject` as the scope.

Thank you for helping `numdantic` improve! :heart:

## Authors

- Milan Staffehl
