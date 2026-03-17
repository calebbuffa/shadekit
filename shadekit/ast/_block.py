"""Statement block collector for pure-Python shader authoring.

:class:`Block` provides an ergonomic way to build ``main()`` bodies
using Python expressions and the shadekit AST, then return them from
``@prog.vertex`` / ``@prog.fragment`` / ``@prog.compute`` decorators.

Usage::

    from shadekit.glsl import Program, vec3, vec4, mat3, normalize
    from shadekit.types import Float, Vec3, Vec4, Mat4
    from shadekit.ast import Block

    prog = Program()
    u_mvp = prog.uniform("u_mvp", Mat4)
    v_normal = prog.varying("v_normal", Vec3)
    f_color = prog.output(0, "f_color", Vec4)

    @prog.fragment
    def fs():
        b = Block()
        n = b.var("n", Vec3, normalize(v_normal))
        b.set(f_color, vec4(n, 1.0))
        return b
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shadekit.ast._expressions import Expr, Variable, _coerce
from shadekit.ast._statements import (
    Assignment,
    Break,
    CompoundAssignment,
    Continue,
    Declaration,
    Discard,
    ExpressionStatement,
    Return,
    Stmt,
)

if TYPE_CHECKING:
    from shadekit.types._base import ShaderMeta


class Block:
    """Collects GLSL statements for ``@prog.vertex`` / ``@prog.fragment`` bodies.

    Each method appends a statement and (where applicable) returns a
    :class:`Variable` AST node so later statements can reference it.

    The block is iterable, so ``@prog.vertex`` recognises it as a
    ``list[Stmt]``.
    """

    __slots__ = ("_stmts",)

    def __init__(self) -> None:
        self._stmts: list[Stmt] = []

    def var(
        self,
        name: str,
        glsl_type: ShaderMeta,
        init: Expr | float | int | bool | None = None,
    ) -> Variable:
        """Declare a local variable and return a :class:`Variable` for later use.

        ``b.var("n", Vec3, normalize(v))`` emits ``vec3 n = normalize(v);``
        and returns a ``Variable("n", Vec3)`` node.
        """
        init_expr = _coerce(init) if init is not None else None
        self._stmts.append(Declaration(name, glsl_type, init_expr))
        return Variable(name, glsl_type)

    def const(
        self,
        name: str,
        glsl_type: ShaderMeta,
        init: Expr | float | int | bool,
    ) -> Variable:
        """Declare a ``const`` local variable."""
        self._stmts.append(Declaration(name, glsl_type, _coerce(init), const=True))
        return Variable(name, glsl_type)

    def set(self, target: Expr, value: Expr | float | int | bool) -> None:
        """Append ``target = value;``."""
        self._stmts.append(Assignment(target, _coerce(value)))

    def add_assign(
        self, op: str, target: Expr, value: Expr | float | int | bool
    ) -> None:
        """Append ``target op value;`` (e.g. ``+=``, ``-=``)."""
        self._stmts.append(CompoundAssignment(op, target, _coerce(value)))

    def return_(self, value: Expr | float | int | bool | None = None) -> None:
        """Append ``return value;`` (or ``return;`` if *value* is ``None``)."""
        self._stmts.append(Return(_coerce(value) if value is not None else None))

    def discard(self) -> None:
        """Append ``discard;``."""
        self._stmts.append(Discard())

    def break_(self) -> None:
        """Append ``break;``."""
        self._stmts.append(Break())

    def continue_(self) -> None:
        """Append ``continue;``."""
        self._stmts.append(Continue())

    def expr(self, e: Expr) -> None:
        """Append a bare expression statement (e.g. a function call)."""
        self._stmts.append(ExpressionStatement(e))

    def add(self, stmt: Stmt) -> None:
        """Append any :class:`Stmt` node directly."""
        self._stmts.append(stmt)

    def __iadd__(self, stmt: Stmt) -> "Block":
        """``block += stmt`` shorthand for :meth:`add`."""
        self._stmts.append(stmt)
        return self

    @property
    def stmts(self) -> list[Stmt]:
        """The collected statements."""
        return self._stmts

    def __iter__(self):
        return iter(self._stmts)

    def __len__(self) -> int:
        return len(self._stmts)

    def __repr__(self) -> str:
        return f"Block({len(self._stmts)} stmts)"
