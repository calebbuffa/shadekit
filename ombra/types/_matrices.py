"""GLSL matrix type classes."""

from ._base import ShaderType


class MatrixType(ShaderType):
    """Base for GLSL matrix types (``mat2``, ``mat3``, ``mat4``).

    Class attributes:
        cols: Number of columns.
        rows: Number of rows.
    """

    cols: int
    rows: int


class Mat2(MatrixType):
    glsl_name = "mat2"
    cols = 2
    rows = 2


class Mat3(MatrixType):
    glsl_name = "mat3"
    cols = 3
    rows = 3


class Mat4(MatrixType):
    glsl_name = "mat4"
    cols = 4
    rows = 4
