from __future__ import annotations

from collections.abc import Callable

from shadekit.ast._expressions import (
    BinaryOp,
    ConstructorCall,
    Expr,
    FieldAccess,
    FunctionCall,
    IndexAccess,
    PostfixOp,
    Ternary,
    UnaryOp,
)
from shadekit.ast._statements import (
    Assignment,
    CompoundAssignment,
    Declaration,
    DoWhile,
    ExpressionStatement,
    For,
    If,
    Return,
    Stmt,
    Switch,
    While,
)
from shadekit.decorators._function import ShaderFunction


def walk_expr(expr: Expr, on_expr: Callable[[Expr], None]) -> None:
    """Depth-first walk of an expression tree, calling *on_expr* per node."""
    on_expr(expr)
    if isinstance(expr, BinaryOp):
        walk_expr(expr.left, on_expr)
        walk_expr(expr.right, on_expr)
    elif isinstance(expr, UnaryOp):
        walk_expr(expr.operand, on_expr)
    elif isinstance(expr, PostfixOp):
        walk_expr(expr.operand, on_expr)
    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            walk_expr(arg, on_expr)
    elif isinstance(expr, ConstructorCall):
        for arg in expr.args:
            walk_expr(arg, on_expr)
    elif isinstance(expr, FieldAccess):
        walk_expr(expr.obj, on_expr)
    elif isinstance(expr, IndexAccess):
        walk_expr(expr.obj, on_expr)
        walk_expr(expr.index, on_expr)
    elif isinstance(expr, Ternary):
        walk_expr(expr.condition, on_expr)
        walk_expr(expr.true_expr, on_expr)
        walk_expr(expr.false_expr, on_expr)


def walk_stmt(stmt: Stmt, on_expr: Callable[[Expr], None]) -> None:
    """Walk a statement, visiting all contained expressions via *on_expr*."""
    if isinstance(stmt, Return) and stmt.value is not None:
        walk_expr(stmt.value, on_expr)
    elif isinstance(stmt, Declaration) and stmt.initializer is not None:
        walk_expr(stmt.initializer, on_expr)
    elif isinstance(stmt, Assignment):
        walk_expr(stmt.target, on_expr)
        walk_expr(stmt.value, on_expr)
    elif isinstance(stmt, CompoundAssignment):
        walk_expr(stmt.target, on_expr)
        walk_expr(stmt.value, on_expr)
    elif isinstance(stmt, ExpressionStatement):
        walk_expr(stmt.expr, on_expr)
    elif isinstance(stmt, If):
        walk_expr(stmt.condition, on_expr)
        for s in stmt.then_body:
            walk_stmt(s, on_expr)
        for cond, body in stmt.elif_clauses:
            walk_expr(cond, on_expr)
            for s in body:
                walk_stmt(s, on_expr)
        if stmt.else_body:
            for s in stmt.else_body:
                walk_stmt(s, on_expr)
    elif isinstance(stmt, For):
        walk_stmt(stmt.init, on_expr)
        walk_expr(stmt.condition, on_expr)
        walk_expr(stmt.update, on_expr)
        for s in stmt.body:
            walk_stmt(s, on_expr)
    elif isinstance(stmt, While):
        walk_expr(stmt.condition, on_expr)
        for s in stmt.body:
            walk_stmt(s, on_expr)
    elif isinstance(stmt, DoWhile):
        for s in stmt.body:
            walk_stmt(s, on_expr)
        walk_expr(stmt.condition, on_expr)
    elif isinstance(stmt, Switch):
        walk_expr(stmt.expr, on_expr)
        for _, body in stmt.cases:
            for s in body:
                walk_stmt(s, on_expr)
        if stmt.default_body:
            for s in stmt.default_body:
                walk_stmt(s, on_expr)


def find_called_names(stmts: list[Stmt]) -> set[str]:
    """Return the set of function names called within *stmts*."""
    names: set[str] = set()

    def _on_expr(expr: Expr) -> None:
        if isinstance(expr, FunctionCall):
            names.add(expr.func_name)

    for stmt in stmts:
        walk_stmt(stmt, _on_expr)
    return names


def collect_shader_functions(stmts: list[Stmt]) -> list[object]:
    """Return all unique :class:`ShaderFunction` instances referenced in *stmts*.

    Walks the AST looking for :class:`FunctionCall` nodes with a
    ``_glsl_fn`` back-reference (set by :meth:`ShaderFunction.__call__`).
    Also transitively collects functions called by those functions.

    Returns a list (no duplicates, stable order).
    """

    seen: dict[str, ShaderFunction] = {}

    def _on_expr(expr: Expr) -> None:
        if isinstance(expr, FunctionCall) and expr._glsl_fn is not None:
            fn = expr._glsl_fn
            if isinstance(fn, ShaderFunction) and fn.name not in seen:
                seen[fn.name] = fn
                # Transitively collect functions called by this one.
                for s in fn.body:
                    walk_stmt(s, _on_expr)

    for stmt in stmts:
        walk_stmt(stmt, _on_expr)
    return list(seen.values())


def collect_transitive_deps(fn: object) -> list[object]:
    """Return *fn* plus all transitive :class:`ShaderFunction` dependencies.

    Walks the function body AST to find any ``FunctionCall`` nodes that
    reference other ``ShaderFunction`` instances (via ``_glsl_fn``),
    recursively collecting the full closure.  Returns in stable order
    (dependencies before dependents).

    This is a core utility so that any shader-language backend (not just
    GLSL) can resolve function dependency graphs.
    """

    visiting: set[str] = set()
    result: list[object] = []

    def _visit(f: object) -> None:
        if not isinstance(f, ShaderFunction) or f.name in visiting:
            return
        visiting.add(f.name)
        # Walk children first (post-order) so deps precede dependents.
        for stmt in f.body:
            walk_stmt(stmt, _on_expr)
        result.append(f)

    def _on_expr(expr: Expr) -> None:
        if (
            isinstance(expr, FunctionCall)
            and getattr(expr, "_glsl_fn", None) is not None
        ):
            _visit(expr._glsl_fn)

    _visit(fn)
    return result
