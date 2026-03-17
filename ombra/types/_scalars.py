"""GLSL scalar type classes."""

from ._base import ShaderType


class ScalarType(ShaderType):
    """Base for GLSL scalar types (``float``, ``int``, ``bool``, etc.)."""


class Float(ScalarType):
    glsl_name = "float"


class Double(ScalarType):
    glsl_name = "double"


class Int(ScalarType):
    glsl_name = "int"


class UInt(ScalarType):
    glsl_name = "uint"


class Bool(ScalarType):
    glsl_name = "bool"
