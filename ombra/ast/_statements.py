"""GLSL statement AST nodes.

Statements are used to build function bodies. They reference
:class:`~ombra.ast._expressions.Expr` nodes for values.

::

    from ombra.ast import Variable, Literal, Assignment, Declaration, Return
    from ombra.types import Float, Vec3

    pos = Variable("pos", Vec3)
    result = Declaration("result", Vec3, pos * Literal(2.0, Float))
    ret = Return(Variable("result", Vec3))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ombra.types._base import ShaderMeta

from ombra.ast._expressions import Expr


class Stmt:
    """Base for all GLSL statement AST nodes."""

    __slots__ = ()


class Declaration(Stmt):
    """A variable declaration, optionally with initializer.

    ``Declaration("color", Vec3, some_expr)``
    → ``vec3 color = <expr>;``

    Set *const* to ``True`` for ``const`` declarations.
    """

    __slots__ = ("name", "glsl_type", "initializer", "const")

    def __init__(
        self,
        name: str,
        glsl_type: ShaderMeta,
        initializer: Expr | None = None,
        *,
        const: bool = False,
    ) -> None:
        self.name = name
        self.glsl_type = glsl_type
        self.initializer = initializer
        self.const = const


class Assignment(Stmt):
    """An assignment to a variable or field.

    ``Assignment(Variable("x", Float), some_expr)``
    → ``x = <expr>;``
    """

    __slots__ = ("target", "value")

    def __init__(self, target: Expr, value: Expr) -> None:
        self.target = target
        self.value = value


class CompoundAssignment(Stmt):
    """A compound assignment (``+=``, ``-=``, ``*=``, ``/=``, etc.).

    ``CompoundAssignment("+=", x, expr)`` → ``x += <expr>;``
    """

    __slots__ = ("op", "target", "value")

    def __init__(self, op: str, target: Expr, value: Expr) -> None:
        self.op = op
        self.target = target
        self.value = value


class Return(Stmt):
    """A return statement.

    ``Return(expr)`` → ``return <expr>;``
    ``Return()`` → ``return;``
    """

    __slots__ = ("value",)

    def __init__(self, value: Expr | None = None) -> None:
        self.value = value


class Discard(Stmt):
    """A ``discard;`` statement (fragment shaders)."""

    __slots__ = ()


class If(Stmt):
    """An if / else-if / else chain.

    ``If(cond, [then_stmts], [elif_clauses], [else_stmts])``
    """

    __slots__ = ("condition", "then_body", "elif_clauses", "else_body")

    def __init__(
        self,
        condition: Expr,
        then_body: Sequence[Stmt],
        elif_clauses: Sequence[tuple[Expr, Sequence[Stmt]]] | None = None,
        else_body: Sequence[Stmt] | None = None,
    ) -> None:
        self.condition = condition
        self.then_body = list(then_body)
        self.elif_clauses = (
            [(c, list(b)) for c, b in elif_clauses] if elif_clauses else []
        )
        self.else_body = list(else_body) if else_body else None


class For(Stmt):
    """A GLSL for-loop.

    ``For(init_stmt, condition_expr, update_expr, body_stmts)``
    """

    __slots__ = ("init", "condition", "update", "body")

    def __init__(
        self,
        init: Stmt,
        condition: Expr,
        update: Expr,
        body: Sequence[Stmt],
    ) -> None:
        self.init = init
        self.condition = condition
        self.update = update
        self.body = list(body)


class ExpressionStatement(Stmt):
    """A bare expression used as a statement (e.g. a function call).

    ``ExpressionStatement(call_expr)`` → ``foo(a, b);``
    """

    __slots__ = ("expr",)

    def __init__(self, expr: Expr) -> None:
        self.expr = expr


class While(Stmt):
    """A GLSL while-loop.

    ``While(cond, body)`` → ``while (cond) { ... }``
    """

    __slots__ = ("condition", "body")

    def __init__(self, condition: Expr, body: Sequence[Stmt]) -> None:
        self.condition = condition
        self.body = list(body)


class DoWhile(Stmt):
    """A GLSL do-while loop.

    ``DoWhile(body, cond)`` → ``do { ... } while (cond);``
    """

    __slots__ = ("body", "condition")

    def __init__(self, body: Sequence[Stmt], condition: Expr) -> None:
        self.body = list(body)
        self.condition = condition


class Switch(Stmt):
    """A GLSL switch statement.

    ``Switch(expr, cases, default_body)``
    → ``switch (expr) { case 0: ...; default: ...; }``

    Each case is a tuple of ``(value_expr, body_stmts)``.
    """

    __slots__ = ("expr", "cases", "default_body")

    def __init__(
        self,
        expr: Expr,
        cases: Sequence[tuple[Expr, Sequence[Stmt]]],
        default_body: Sequence[Stmt] | None = None,
    ) -> None:
        self.expr = expr
        self.cases = [(v, list(b)) for v, b in cases]
        self.default_body = list(default_body) if default_body else None


class Break(Stmt):
    """A ``break;`` statement."""

    __slots__ = ()


class Continue(Stmt):
    """A ``continue;`` statement."""

    __slots__ = ()
