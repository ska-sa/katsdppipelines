import attr

ObitType = attr.make_class("ObitType",
                        ["enum", "name", "description", "coerce"],
                        frozen=True, slots=True)

# Derived from ObitTypes.h
OBIT_TYPE_ENUM = [
    ObitType(0, "byte", "8-bit signed byte", int),
    ObitType(1, "short", "16-bit signed integer", int),
    ObitType(2, "int", "signed integer", int),
    ObitType(3, "oint", "FORTRAN integer", int),
    ObitType(4, "long", "32-bit signed integer", int),
    ObitType(5, "ubyte", "8-bit unsigned byte", int),
    ObitType(6, "ushort", "16-bit unsigned integer", int),
    ObitType(7, "uint", "FORTRAN unsigned integer", int),
    ObitType(8, "ulong", "32-bit unsigned integer", int),
    ObitType(9, "llong", "64-bit signed integer", int),
    ObitType(10, "float", "32-bit single precision float", float),
    ObitType(11, "double", "64-bit double precision float", float),
    ObitType(12, "complex", "32-bit single precision complex", complex),
    ObitType(13, "dcomplex", "64-bit double precision complex", complex),
    ObitType(14, "string", "fixed length string", str),
    ObitType(15, "bool", "boolean", bool),
    ObitType(16, "bits", "bits", int),
]

# Classify the Obit types
OBIT_INTS = range(10)
OBIT_FLOATS = range(10, 12)
OBIT_COMPLEXES = range(12, 14)
OBIT_STRINGS = range(14, 15)
OBIT_BOOLS = range(15, 16)
OBIT_BITS = range(16, 17)

OBIT_TYPE = (attr.make_class("ObitTypes",
                [t.name for t in OBIT_TYPE_ENUM])
                (*[t.enum for t in OBIT_TYPE_ENUM]))
