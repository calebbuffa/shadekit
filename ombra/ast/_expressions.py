"""GLSL expression AST nodes with operator overloading.

Every node is an :class:`Expr` subclass carrying an inferred
:class:`~ombra.types.ShaderMeta` type.  Python operators build the AST::

    from ombra.ast import Variable
    from ombra.types import Vec3, Float, Mat4

    pos   = Variable("pos", Vec3)
    scale = Variable("scale", Float)
    mvp   = Variable("mvp", Mat4)

    result = mvp * (pos * scale)
    # BinaryOp("*", mvp, BinaryOp("*", pos, scale))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ombra.types._arrays import ArrayType
from ombra.types._inference import infer_binary_type
from ombra.types._matrices import MatrixType
from ombra.types._scalars import Bool, Float, Int
from ombra.types._vectors import VectorType

if TYPE_CHECKING:
    from ombra.types._base import ShaderMeta


# Operator precedence (lower = binds tighter)

_PRECEDENCE: dict[str, int] = {
    "*": 3,
    "/": 3,
    "%": 3,
    "+": 4,
    "-": 4,
    "<<": 5,
    ">>": 5,
    "<": 6,
    ">": 6,
    "<=": 6,
    ">=": 6,
    "==": 7,
    "!=": 7,
    "&": 8,
    "^": 9,
    "|": 10,
    "&&": 11,
    "^^": 12,
    "||": 13,
}


class Expr:
    """Base for all GLSL expression AST nodes.

    Attributes:
        glsl_type: The inferred GLSL type of this expression (a class
            whose metaclass is :class:`ShaderMeta`).
    """

    __slots__ = ("glsl_type",)

    def __init__(self, glsl_type: ShaderMeta) -> None:
        self.glsl_type = glsl_type

    def __add__(self, other: Expr) -> BinaryOp:
        return _binop("+", self, other)

    def __radd__(self, other: Expr) -> BinaryOp:
        return _binop("+", other, self)

    def __sub__(self, other: Expr) -> BinaryOp:
        return _binop("-", self, other)

    def __rsub__(self, other: Expr) -> BinaryOp:
        return _binop("-", other, self)

    def __mul__(self, other: Expr) -> BinaryOp:
        return _binop("*", self, other)

    def __rmul__(self, other: Expr) -> BinaryOp:
        return _binop("*", other, self)

    def __truediv__(self, other: Expr) -> BinaryOp:
        return _binop("/", self, other)

    def __rtruediv__(self, other: Expr) -> BinaryOp:
        return _binop("/", other, self)

    def __mod__(self, other: Expr) -> BinaryOp:
        return _binop("%", self, other)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("-", self, self.glsl_type)

    def __and__(self, other: Expr) -> BinaryOp:
        return _binop("&", self, other)

    def __rand__(self, other: Expr) -> BinaryOp:
        return _binop("&", other, self)

    def __or__(self, other: Expr) -> BinaryOp:
        return _binop("|", self, other)

    def __ror__(self, other: Expr) -> BinaryOp:
        return _binop("|", other, self)

    def __xor__(self, other: Expr) -> BinaryOp:
        return _binop("^", self, other)

    def __rxor__(self, other: Expr) -> BinaryOp:
        return _binop("^", other, self)

    def __invert__(self) -> UnaryOp:
        return UnaryOp("~", self, self.glsl_type)

    def __lshift__(self, other: Expr) -> BinaryOp:
        return _binop("<<", self, other)

    def __rlshift__(self, other: Expr) -> BinaryOp:
        return _binop("<<", other, self)

    def __rshift__(self, other: Expr) -> BinaryOp:
        return _binop(">>", self, other)

    def __rrshift__(self, other: Expr) -> BinaryOp:
        return _binop(">>", other, self)

    def __lt__(self, other: Expr) -> BinaryOp:
        return _binop("<", self, other)

    def __le__(self, other: Expr) -> BinaryOp:
        return _binop("<=", self, other)

    def __gt__(self, other: Expr) -> BinaryOp:
        return _binop(">", self, other)

    def __ge__(self, other: Expr) -> BinaryOp:
        return _binop(">=", self, other)

    def __getattr__(self, name: str) -> FieldAccess:
        # Only intercept plausible GLSL swizzles/fields, not dunder attrs.
        if name.startswith("_"):
            raise AttributeError(name)
        if self.glsl_type is None:
            raise AttributeError(name)
        return FieldAccess(self, name)

    def __getitem__(self, index: Expr | int) -> IndexAccess:
        idx = _coerce(index)
        result_type = _infer_index_type(self.glsl_type)
        return IndexAccess(self, idx, result_type)

    def eq(self, other: object) -> BinaryOp:
        """GLSL ``==`` comparison.  Returns a :class:`BinaryOp`, not a Python bool."""
        return _binop("==", self, other)

    def ne(self, other: object) -> BinaryOp:
        """GLSL ``!=`` comparison.  Returns a :class:`BinaryOp`, not a Python bool."""
        return _binop("!=", self, other)


class Literal(Expr):
    """A compile-time constant value.

    ``Literal(1.0, Float)`` → ``"1.0"``
    """

    __slots__ = ("value",)

    def __init__(self, value: int | float | bool, glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.value = value


class Variable(Expr):
    """A named variable reference.

    ``Variable("pos", Vec3)`` → ``"pos"``
    """

    __slots__ = ("name",)

    def __init__(self, name: str, glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.name = name


class BinaryOp(Expr):
    """A binary operator expression.

    ``BinaryOp("*", a, b, Vec3)`` → ``"a * b"``
    """

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Expr, right: Expr, glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.op = op
        self.left = left
        self.right = right


class UnaryOp(Expr):
    """A unary operator expression (prefix).

    ``UnaryOp("-", x, Float)`` → ``"-x"``
    """

    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Expr, glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.op = op
        self.operand = operand


class FunctionCall(Expr):
    """A GLSL function call.

    ``FunctionCall("normalize", [v], Vec3)`` → ``"normalize(v)"``

    When created by a :class:`~ombra.decorators.GlslFunction`, the
    ``_glsl_fn`` attribute holds a back-reference so the stage decorator
    can auto-include the function definition.
    """

    __slots__ = ("func_name", "args", "_glsl_fn")

    def __init__(self, func_name: str, args: list[Expr], glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.func_name = func_name
        self.args = args
        self._glsl_fn: object | None = None


class ConstructorCall(Expr):
    """A GLSL type constructor.

    ``ConstructorCall(Vec4, [pos, 1.0_lit], Vec4)`` → ``"vec4(pos, 1.0)"``
    """

    __slots__ = ("target_type", "args")

    def __init__(
        self, target_type: ShaderMeta, args: list[Expr], glsl_type: ShaderMeta
    ) -> None:
        super().__init__(glsl_type)
        self.target_type = target_type
        self.args = args


class FieldAccess(Expr):
    """A field / swizzle access.

    ``FieldAccess(v, "xyz")`` → ``"v.xyz"``
    """

    __slots__ = ("obj", "field")

    def __init__(self, obj: Expr, field: str) -> None:
        super().__init__(_infer_swizzle_type(obj.glsl_type, field))
        self.obj = obj
        self.field = field


class IndexAccess(Expr):
    """An array subscript expression.

    ``IndexAccess(arr, i, Float)`` → ``"arr[i]"``
    """

    __slots__ = ("obj", "index")

    def __init__(self, obj: Expr, index: Expr, glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.obj = obj
        self.index = index


class Ternary(Expr):
    """A ternary conditional expression.

    ``Ternary(cond, a, b, Vec3)`` → ``"cond ? a : b"``
    """

    __slots__ = ("condition", "true_expr", "false_expr")

    def __init__(
        self,
        condition: Expr,
        true_expr: Expr,
        false_expr: Expr,
        glsl_type: ShaderMeta,
    ) -> None:
        super().__init__(glsl_type)
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr


class PostfixOp(Expr):
    """A postfix increment/decrement.

    ``PostfixOp("++", x, Int)`` → ``"x++"``
    """

    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Expr, glsl_type: ShaderMeta) -> None:
        super().__init__(glsl_type)
        self.op = op
        self.operand = operand


def _coerce(value: object) -> Expr:
    """Auto-coerce a Python literal to a :class:`Literal` AST node.

    Accepts ``Expr`` pass-through, ``float``, ``int``, and ``bool``.
    """
    if isinstance(value, Expr):
        return value

    if isinstance(value, bool):
        return Literal(value, Bool)
    if isinstance(value, float):
        return Literal(value, Float)
    if isinstance(value, int):
        return Literal(value, Int)
    raise TypeError(f"Cannot coerce {type(value).__name__} to GLSL expression")


def _binop(op: str, left: object, right: object) -> BinaryOp:
    """Create a :class:`BinaryOp` with inferred type, auto-coercing Python literals."""

    left_expr = _coerce(left)
    right_expr = _coerce(right)
    result_type = infer_binary_type(op, left_expr.glsl_type, right_expr.glsl_type)
    return BinaryOp(op, left_expr, right_expr, result_type)


# Valid GLSL swizzle characters and their component index.
_SWIZZLE_SETS = (
    {"x": 0, "y": 1, "z": 2, "w": 3},
    {"r": 0, "g": 1, "b": 2, "a": 3},
    {"s": 0, "t": 1, "p": 2, "q": 3},
)


def _infer_swizzle_type(parent_type: ShaderMeta, field: str) -> ShaderMeta:
    """Infer the result type of a swizzle or field access."""

    if not issubclass(parent_type, VectorType):
        # Not a vector — return parent type (struct field access, resolved later).
        return parent_type

    # Check if all characters belong to one swizzle set.
    for charset in _SWIZZLE_SETS:
        if all(c in charset for c in field):
            # Validate component indices are in range.
            for c in field:
                if charset[c] >= parent_type.size:
                    raise AttributeError(
                        f"Swizzle '{c}' out of range for {parent_type} (size {parent_type.size})"
                    )
            n = len(field)
            if n == 1:
                return parent_type.component_type
            # Return vector of same component type with swizzle length.
            from ombra.types._inference import _vec_for

            return _vec_for(parent_type.component_type, n)

    # Not a swizzle — treat as opaque field access, keep parent type.
    return parent_type


def _infer_index_type(parent_type: ShaderMeta) -> ShaderMeta:
    """Infer the element type when indexing into *parent_type*."""

    if isinstance(parent_type, ArrayType):
        return parent_type.base
    if issubclass(parent_type, VectorType):
        return parent_type.component_type
    if issubclass(parent_type, MatrixType):
        # mat4[0] yields a column vector
        from ombra.types._inference import _vec_for
        from ombra.types._scalars import Float

        return _vec_for(Float, parent_type.rows)
    # Unknown — return parent type (e.g. opaque buffer access)
    return parent_type


# ── Public helper constructors ───────────────────────────────────────


def ternary(condition: object, true_expr: object, false_expr: object) -> Ternary:
    """Build a ``cond ? a : b`` ternary expression with auto-coercion."""
    cond = _coerce(condition)
    t = _coerce(true_expr)
    f = _coerce(false_expr)
    return Ternary(cond, t, f, t.glsl_type)


def logical_not(operand: object) -> UnaryOp:
    """Build a GLSL ``!expr`` logical-NOT expression."""

    expr = _coerce(operand)
    return UnaryOp("!", expr, Bool)


def logical_and(left: object, right: object) -> BinaryOp:
    """Build a GLSL ``a && b`` expression."""
    return _binop("&&", left, right)


def logical_or(left: object, right: object) -> BinaryOp:
    """Build a GLSL ``a || b`` expression."""
    return _binop("||", left, right)


def pre_increment(operand: Expr) -> UnaryOp:
    """Build a GLSL ``++x`` expression."""
    return UnaryOp("++", operand, operand.glsl_type)


def pre_decrement(operand: Expr) -> UnaryOp:
    """Build a GLSL ``--x`` expression."""
    return UnaryOp("--", operand, operand.glsl_type)


def post_increment(operand: Expr) -> PostfixOp:
    """Build a GLSL ``x++`` expression."""
    return PostfixOp("++", operand, operand.glsl_type)


def post_decrement(operand: Expr) -> PostfixOp:
    """Build a GLSL ``x--`` expression."""
    return PostfixOp("--", operand, operand.glsl_type)
