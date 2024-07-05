"""
Create a table displaying what dtypes are compatible in NDArray assignments.

This script creates a table that shows how the various scalar types of
``numpy`` are compatible when used in an ``NDArray`` of ``numdantic``.
The table rows show the type of the array that is being assigned to a
variable of the dtype given by the column. If the assignment passes the
type check, it is marked with an X, otherwise it is marked with a period
(``.``) for visual clarity. The table is followed by some general info
about your machine, to give context to the table. This is important as
the implementation oif ``numpy`` scalar types is architecture and system
dependent. Depending on their availability, some scalr types might also
not be available on your machine and therefore are not present in the
table.

Note that this script must be run from a Python environment that has
both ``numpy`` and ``mypy`` installed.

The table is printed to stdout, so you can pipe it to a file. If you do
so, you can use the ``.rst`` file extension as the output is compatible
with reStructuredText.
"""

import os
import platform
import random
import re
import subprocess
from pathlib import Path

import numpy as np

import numdantic

project_root = Path(__file__).resolve().parents[1]


# list of types and their corresponding annotation
numpy_scalar_types = [
    # generic types
    "generic",
    "number[Any]",
    # integer types
    "integer[Any]",
    "signedinteger[Any]",
    "byte",
    "short",
    "intc",
    "int_",
    "long",
    "longlong",
    # unsigned integer types
    "unsignedinteger[Any]",
    "ubyte",
    "ushort",
    "uintc",
    "uint",
    "ulong",
    "ulonglong",
    # inexact generics
    "inexact[Any]",
    # floating point types
    "floating[Any]",
    "half",
    "single",
    "double",
    "longdouble",
    # complex floating point types
    "complexfloating[Any, Any]",
    "csingle",
    "cdouble",
    "clongdouble",
    # bool types
    "bool",
    "bool_",
    # sized aliases integers
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "intp",
    "uintp",
    # sized aliases inexact
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
# platform dependent types
platform_dependent = ("float96", "float128", "complex192", "complex256")
for platform_type in platform_dependent:
    if hasattr(np, platform_type):
        numpy_scalar_types.append(platform_type)


def compare_types_mypy() -> None:
    """Create a table of dtype compatability"""
    # create a table as a string
    table = "Compatibility table of dtypes\n=============================\n\n"
    type_names = [
        t.removesuffix("[Any]").removesuffix("[Any, Any]")
        for t in numpy_scalar_types
    ]

    # create a table header
    first_column_width = max([len(x) for x in type_names]) + 5
    table += "Actual type".ljust(first_column_width)
    for i in range(len(type_names)):
        table += f"{i:^4d}"
    table += "\n"
    table += "=" * (first_column_width - 1)
    table += " "
    for _ in range(len(type_names)):
        table += "=== "
    table += "\n"

    # create a string with all dtype combinations
    test_string = (
        "from typing import Any\n"
        "from numdantic import NDArray, Shape\n"
        "import numpy as np\n\n"
    )
    for actual_type in numpy_scalar_types:
        for target_type in numpy_scalar_types:
            rid = random.randint(0, 999999)
            test_string += (
                f"x_{rid}: NDArray[Shape[int], np.{actual_type}] = "
                f"np.array([1], dtype=np.{actual_type})\n"
                f"y_{rid}: NDArray[Shape[int], np.{target_type}] = x_{rid}\n\n"
            )

    # write the string to a file for mypy to parse
    tmp_filepath = Path(__file__).parent / "dtypes_test.py"
    with open(tmp_filepath, "w+") as file:
        file.write(test_string)

    # validate with mypy
    output = subprocess.run(
        f"mypy --config-file pyproject.toml {tmp_filepath}",
        capture_output=True,
        env=os.environ,
        cwd=project_root,  # use pyproject.toml as config file
        shell=True,
    )

    # clean-up
    tmp_filepath.unlink()

    # parse output
    current_line = 6
    for i, actual_type in enumerate(type_names):
        table += f"{i}: {actual_type}".ljust(first_column_width)
        for _ in type_names:
            pattern = re.compile(rf"dtypes_test\.py:{current_line}:")
            if re.search(pattern, str(output.stdout)):
                table += " .  "
            else:
                table += " X  "
            current_line += 3
        table += "\n"

    # diagnostic info about the system
    table += (
        f"\nTested on {platform.platform()} {platform.architecture()[0]} "
        f"({platform.machine()}) with Python {platform.python_version()}. "
        f"Used versions: numpy: {np.__version__}, numdantic: "
        f"{numdantic.__version__}."
    )

    # print the result
    print(table)


if __name__ == "__main__":
    compare_types_mypy()
