# Copyright (c) 2024 Milan Staffehl - subject to the MIT license.
"""
Helper variables containing info on numpy scalar dtypes, and helper functions.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, TextIO

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def assert_type_check_passes(
    test_string: str,
    temp_file: TextIO,
    temp_file_path: Path,
    fail_msg: str = "Type check test failed",
) -> subprocess.CompletedProcess[Any]:
    """
    Assert that the given string passes the type check. Return output.

    :param test_string: File contents to check. Must be formatted as a
        syntactically valid Python file.
    :param temp_file: Opened file-like object to write the test string
        to and to feed to type checker.
    :param temp_file_path: The path of the test file.
    :param fail_msg: A failure message to display should the test fail.
    :return:
    """
    # write test contents to file
    temp_file.write(test_string)
    temp_file.close()

    output = subprocess.run(
        f"mypy --config-file pyproject.toml {str(temp_file_path)}",
        capture_output=True,
        env=os.environ,
        cwd=_PROJECT_ROOT,  # use pyproject.toml as config file
        shell=True,
    )
    if output.returncode != 0:
        pytest.fail(
            f"{fail_msg}:"
            f"\nstdout:\n{output.stdout.decode('utf-8')}"
            f"\nstderr:\n{output.stderr.decode('utf-8')}\n"
        )
    return output


# Mapping of dtypes to their parent dtypes
NP_SCALAR_TYPES_PARENTS = {
    # generic types
    "generic": [],
    "number[Any]": ["generic"],
    "integer[Any]": ["generic", "number[Any]"],
    "signedinteger[Any]": ["generic", "number[Any]", "integer[Any]"],
    "unsignedinteger[Any]": ["generic", "number[Any]", "integer[Any]"],
    "inexact[Any]": ["generic", "number[Any]"],
    "floating[Any]": ["generic", "number[Any]", "inexact[Any]"],
    "complexfloating[Any, Any]": ["generic", "number[Any]", "inexact[Any]"],
    # integer types
    "byte": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "short": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "intc": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "int_": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "long": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "longlong": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "signedinteger[Any]",
    ],
    "intp": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    # unsigned integer types
    "ubyte": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "ushort": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "uintc": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "uint": ["generic", "number[Any]", "integer[Any]", "unsignedinteger[Any]"],
    "ulong": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "ulonglong": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "uintp": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    # floating point types
    "half": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "single": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "double": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "longdouble": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    # complex floating point types
    "csingle": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
    "cdouble": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
    "clongdouble": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
    # bool types
    "bool": ["generic"],
    "bool_": ["generic"],
    # sized aliases integers
    "int8": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "int16": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "int32": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "int64": ["generic", "number[Any]", "integer[Any]", "signedinteger[Any]"],
    "uint8": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "uint16": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "uint32": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    "uint64": [
        "generic",
        "number[Any]",
        "integer[Any]",
        "unsignedinteger[Any]",
    ],
    # sized aliases inexact
    "float16": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "float32": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "float64": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "float96": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "float128": ["generic", "number[Any]", "inexact[Any]", "floating[Any]"],
    "complex64": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
    "complex128": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
    "complex192": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
    "complex256": [
        "generic",
        "number[Any]",
        "inexact[Any]",
        "complexfloating[Any, Any]",
    ],
}

# lists of explicit dtypes, sorted by implementation
INT_TYPES = [
    "byte",
    "short",
    "intc",
    "int_",
    "long",
    "longlong",
    "int8",
    "int16",
    "int32",
    "int64",
]
UINT_TYPES = [
    "ubyte",
    "ushort",
    "uintc",
    "uint",
    "ulong",
    "ulonglong",
    "sized alias",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
FLOATING_TYPES = [
    "half",
    "single",
    "double",
    "longdouble",
    "float16",
    "float32",
    "float64",
    "float96",
    "float128",
]
COMPLEX_TYPES = [
    "csingle",
    "cdouble",
    "clongdouble",
    "complex64",
    "complex128",
    "complex192",
    "complex256",
]
