"""GLSL type inference for binary operations and constructor validation."""

from __future__ import annotations

from . import _vectors
from ._base import ShaderMeta
from ._matrices import MatrixType
from ._scalars import Bool, Double, Float, Int, ScalarType, UInt
from ._vectors import VectorType

# Implicit scalar promotion: int → uint → float → double.
_PROMOTION: dict[type[ScalarType], int] = {Int: 0, UInt: 1, Float: 2, Double: 3}
_RANK_TO_SCALAR: dict[int, type[ScalarType]] = {v: k for k, v in _PROMOTION.items()}

# Lookup table: (component glsl_name, size) → VectorType class.
_VEC_TABLE: dict[tuple[str, int], type[VectorType]] = {}


def _ensure_vec_table() -> None:
    """Lazily populate ``_VEC_TABLE`` from the vector module classes."""
    if _VEC_TABLE:
        return

    for obj_name in dir(_vectors):
        obj = getattr(_vectors, obj_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, VectorType)
            and obj is not VectorType
        ):
            _VEC_TABLE[(obj.component_type.glsl_name, obj.size)] = obj


def _promote(a: type[ScalarType], b: type[ScalarType]) -> type[ScalarType]:
    """Return the promoted scalar type of *a* and *b*."""
    if a is b:
        return a
    ra, rb = _PROMOTION.get(a), _PROMOTION.get(b)
    if ra is None or rb is None:
        raise TypeError(f"Cannot promote {a} and {b}")
    return _RANK_TO_SCALAR[max(ra, rb)]


def _vec_for(scalar: type[ScalarType], size: int) -> type[VectorType]:
    """Return the vector type with the given *scalar* component and *size*."""
    _ensure_vec_table()
    key = (scalar.glsl_name, size)
    result = _VEC_TABLE.get(key)
    if result is None:
        raise TypeError(f"No vector type for {scalar} × {size}")
    return result


def infer_binary_type(op: str, left: ShaderMeta, right: ShaderMeta) -> ShaderMeta:
    """Infer the result type of ``left op right``.

    Supports:
    - Arithmetic: ``+  -  *  /``
    - Comparison: ``<  >  <=  >=  ==  !=``
    - Logical:    ``&&  ||  ^^``

    Raises :class:`TypeError` if the operation is invalid.
    """
    if op in ("&&", "||", "^^"):
        if left is not Bool or right is not Bool:
            raise TypeError(
                f"Logical {op} requires bool operands, got {left} and {right}"
            )
        return Bool

    if op in ("<", ">", "<=", ">=", "==", "!="):
        return _infer_comparison(op, left, right)

    if op in ("+", "-", "*", "/", "%"):
        return _infer_arithmetic(op, left, right)

    if op in ("&", "|", "^", "<<", ">>"):
        return _infer_bitwise(op, left, right)

    raise TypeError(f"Unknown operator: {op!r}")


def validate_constructor(target: ShaderMeta, args: list[ShaderMeta]) -> bool:
    """Check whether *args* can construct *target* via a GLSL constructor.

    Returns ``True`` if the constructor is valid, ``False`` otherwise.

    Examples::

        validate_constructor(Vec4, [Vec3, Float])   # True  (3+1=4)
        validate_constructor(Vec4, [Float])          # True  (fill)
        validate_constructor(Mat4, [Float])          # True  (diagonal)
        validate_constructor(Vec3, [Mat4])           # False
    """
    if issubclass(target, VectorType):
        return _validate_vec_ctor(target, args)
    if issubclass(target, MatrixType):
        return _validate_mat_ctor(target, args)
    return False


def _infer_comparison(op: str, left: ShaderMeta, right: ShaderMeta) -> ShaderMeta:
    if issubclass(left, ScalarType) and issubclass(right, ScalarType):
        return Bool
    if issubclass(left, VectorType) and issubclass(right, VectorType):
        if left.size != right.size:
            raise TypeError(f"Vector sizes must match for {op}: {left} vs {right}")
        return _vec_for(Bool, left.size)
    raise TypeError(f"Cannot compare {left} and {right} with {op}")


def _infer_arithmetic(op: str, left: ShaderMeta, right: ShaderMeta) -> ShaderMeta:
    # scalar and scalar
    if issubclass(left, ScalarType) and issubclass(right, ScalarType):
        return _promote(left, right)

    # vector and scalar / scalar and vector
    if issubclass(left, VectorType) and issubclass(right, ScalarType):
        return _vec_for(_promote(left.component_type, right), left.size)
    if issubclass(left, ScalarType) and issubclass(right, VectorType):
        return _vec_for(_promote(left, right.component_type), right.size)

    # vector and vector (component-wise, same size)
    if issubclass(left, VectorType) and issubclass(right, VectorType):
        if left.size != right.size:
            raise TypeError(f"Vector sizes must match for {op}: {left} vs {right}")
        return _vec_for(_promote(left.component_type, right.component_type), left.size)

    # matrix * vector / vector * matrix (only for *)
    if op == "*":
        if issubclass(left, MatrixType) and issubclass(right, VectorType):
            if left.cols != right.size:
                raise TypeError(
                    f"Matrix-vector multiply dimension mismatch: {left} * {right}"
                )
            return _vec_for(Float, left.rows)
        if issubclass(left, VectorType) and issubclass(right, MatrixType):
            if left.size != right.rows:
                raise TypeError(
                    f"Vector-matrix multiply dimension mismatch: {left} * {right}"
                )
            return _vec_for(Float, right.cols)

    # matrix and matrix
    if issubclass(left, MatrixType) and issubclass(right, MatrixType):
        if left.cols != right.cols or left.rows != right.rows:
            raise TypeError(f"Matrix sizes must match for {op}: {left} vs {right}")
        return left

    # matrix and scalar / scalar and matrix
    if issubclass(left, MatrixType) and issubclass(right, ScalarType):
        return left
    if issubclass(left, ScalarType) and issubclass(right, MatrixType):
        return right

    raise TypeError(f"Cannot apply {op} to {left} and {right}")


def _validate_vec_ctor(target: type[VectorType], args: list[ShaderMeta]) -> bool:
    if not args:
        return False
    # Single scalar -> fill all components.
    if len(args) == 1 and issubclass(args[0], ScalarType):
        return True
    # Mixed scalars/vectors — total component count must match target size.
    total = 0
    for a in args:
        if issubclass(a, ScalarType):
            total += 1
        elif issubclass(a, VectorType):
            total += a.size
        else:
            return False
    return total == target.size


def _validate_mat_ctor(target: type[MatrixType], args: list[ShaderMeta]) -> bool:
    if not args:
        return False
    # Single scalar -> diagonal matrix.
    if len(args) == 1 and issubclass(args[0], ScalarType):
        return True
    # N column vectors of size == rows.
    if all(issubclass(a, VectorType) and a.size == target.rows for a in args):
        return len(args) == target.cols
    return False


def _infer_bitwise(op: str, left: ShaderMeta, right: ShaderMeta) -> ShaderMeta:
    """Infer the result type of bitwise operations (&, |, ^, <<, >>).

    Bitwise ops are valid on integer scalar/vector types only.
    """
    _INT_SCALARS = (Int, UInt)

    # scalar and scalar
    if issubclass(left, ScalarType) and issubclass(right, ScalarType):
        if left not in _INT_SCALARS or right not in _INT_SCALARS:
            raise TypeError(
                f"Bitwise {op} requires integer types, got {left} and {right}"
            )
        return _promote(left, right)

    # vector and scalar / scalar and vector
    if issubclass(left, VectorType) and issubclass(right, ScalarType):
        if left.component_type not in _INT_SCALARS or right not in _INT_SCALARS:
            raise TypeError(
                f"Bitwise {op} requires integer types, got {left} and {right}"
            )
        return _vec_for(_promote(left.component_type, right), left.size)
    if issubclass(left, ScalarType) and issubclass(right, VectorType):
        if left not in _INT_SCALARS or right.component_type not in _INT_SCALARS:
            raise TypeError(
                f"Bitwise {op} requires integer types, got {left} and {right}"
            )
        return _vec_for(_promote(left, right.component_type), right.size)

    # vector and vector
    if issubclass(left, VectorType) and issubclass(right, VectorType):
        if left.size != right.size:
            raise TypeError(f"Vector sizes must match for {op}: {left} vs {right}")
        if (
            left.component_type not in _INT_SCALARS
            or right.component_type not in _INT_SCALARS
        ):
            raise TypeError(
                f"Bitwise {op} requires integer types, got {left} and {right}"
            )
        return _vec_for(_promote(left.component_type, right.component_type), left.size)

    raise TypeError(f"Cannot apply {op} to {left} and {right}")
