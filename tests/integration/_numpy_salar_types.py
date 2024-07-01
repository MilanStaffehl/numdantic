"""
Helper variables containing info on numpy scalar dtypes.
"""

# Mapping of dtypes to their parent dtypes
numpy_scalar_types = {
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
signedinteger_dtypes = {
    "C-type": ["byte", "short", "intc", "int_", "long", "longlong"],
    "sized alias": ["int8", "int16", "int32", "int64"],
}
unsignedinteger_dtypes = {
    "C-type": ["ubyte", "ushort", "uintc", "uint", "ulong", "ulonglong"],
    "sized alias": ["uint8", "uint16", "uint32", "uint64"],
}
floating_dtypes = {
    "C-type": ["half", "single", "double", "longdouble"],
    "sized alias": ["float16", "float32", "float64", "float96", "float128"],
}
complex_dtypes = {
    "C-type": ["csingle", "cdouble", "clongdouble"],
    "sized alias": ["complex64", "complex128", "complex192", "complex256"],
}
