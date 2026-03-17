"""Constant folding — evaluate compile-time constant expressions.

Simplifies AST expressions where both operands are :class:`Literal`
nodes, and removes identity operations (``x * 1.0``, ``x + 0.0``, etc.).

Usage::

    from shadekit.compiler import fold_constants
    optimized_expr = fold_constants(expr)
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from shadekit.ast._expressions import (
    BinaryOp,
    ConstructorCall,
    Expr,
    FieldAccess,
    FunctionCall,
    IndexAccess,
    Literal,
    PostfixOp,
    Ternary,
    UnaryOp,
    Variable,
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

if TYPE_CHECKING:
    pass

# Arithmetic operators that can be folded.
_OPS: dict[str, object] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
}

# Comparison operators that can be folded.
_CMP_OPS: dict[str, object] = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def fold_expr(node: Expr) -> Expr:
    """Return a simplified copy of *node* with constants folded."""
    if isinstance(node, Literal) or isinstance(node, Variable):
        return node

    if isinstance(node, UnaryOp):
        operand = fold_expr(node.operand)
        if node.op == "-" and isinstance(operand, Literal):
            return Literal(-operand.value, operand.glsl_type)
        if operand is node.operand:
            return node
        return UnaryOp(node.op, operand, node.glsl_type)

    if isinstance(node, BinaryOp):
        left = fold_expr(node.left)
        right = fold_expr(node.right)
        return _fold_binary(node.op, left, right, node.glsl_type, node)

    if isinstance(node, FunctionCall):
        new_args = [fold_expr(a) for a in node.args]
        if all(a is o for a, o in zip(new_args, node.args)):
            return node
        return FunctionCall(node.func_name, new_args, node.glsl_type)

    if isinstance(node, ConstructorCall):
        new_args = [fold_expr(a) for a in node.args]
        if all(a is o for a, o in zip(new_args, node.args)):
            return node
        return ConstructorCall(node.target_type, new_args, node.glsl_type)

    if isinstance(node, FieldAccess):
        obj = fold_expr(node.obj)
        if obj is node.obj:
            return node
        fa = FieldAccess.__new__(FieldAccess)
        # Bypass __init__ which re-infers swizzle type.
        Expr.__init__(fa, node.glsl_type)
        fa.obj = obj
        fa.field = node.field
        return fa

    if isinstance(node, IndexAccess):
        obj = fold_expr(node.obj)
        idx = fold_expr(node.index)
        if obj is node.obj and idx is node.index:
            return node
        return IndexAccess(obj, idx, node.glsl_type)

    if isinstance(node, Ternary):
        cond = fold_expr(node.condition)
        t = fold_expr(node.true_expr)
        f = fold_expr(node.false_expr)
        # If condition is a literal bool, simplify.
        if isinstance(cond, Literal) and isinstance(cond.value, bool):
            return t if cond.value else f
        if cond is node.condition and t is node.true_expr and f is node.false_expr:
            return node
        return Ternary(cond, t, f, node.glsl_type)

    if isinstance(node, PostfixOp):
        operand = fold_expr(node.operand)
        if operand is node.operand:
            return node
        return PostfixOp(node.op, operand, node.glsl_type)

    return node


def _fold_binary(
    op: str, left: Expr, right: Expr, glsl_type: object, orig: BinaryOp
) -> Expr:
    """Try to simplify a binary operation."""
    # Both sides are literal → evaluate at build time.
    if isinstance(left, Literal) and isinstance(right, Literal):
        py_op = _OPS.get(op) or _CMP_OPS.get(op)
        if py_op is not None:
            try:
                result = py_op(left.value, right.value)  # type: ignore[operator]
            except (ZeroDivisionError, ArithmeticError):
                pass
            else:
                from shadekit.types._scalars import Bool

                if isinstance(result, bool):
                    return Literal(result, Bool)
                return Literal(result, left.glsl_type)

    # Identity rules — keep the surviving side's type.
    if op == "+" and _is_zero(right):
        return left
    if op == "+" and _is_zero(left):
        return right
    if op == "-" and _is_zero(right):
        return left
    if op == "*" and _is_one(right):
        return left
    if op == "*" and _is_one(left):
        return right
    if op == "*" and _is_zero(right):
        return right  # 0
    if op == "*" and _is_zero(left):
        return left  # 0
    if op == "/" and _is_one(right):
        return left

    # Nothing changed → return original.
    if left is orig.left and right is orig.right:
        return orig
    return BinaryOp(op, left, right, glsl_type)


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Literal) and expr.value == 0


def _is_one(expr: Expr) -> bool:
    return isinstance(expr, Literal) and expr.value == 1


def fold_stmt(stmt: Stmt) -> Stmt:
    """Return a copy of *stmt* with constant expressions folded."""
    if isinstance(stmt, Return):
        if stmt.value is not None:
            folded = fold_expr(stmt.value)
            if folded is stmt.value:
                return stmt
            return Return(folded)
        return stmt

    if isinstance(stmt, Declaration):
        if stmt.initializer is not None:
            folded = fold_expr(stmt.initializer)
            if folded is stmt.initializer:
                return stmt
            return Declaration(stmt.name, stmt.glsl_type, folded)
        return stmt

    if isinstance(stmt, Assignment):
        target = fold_expr(stmt.target)
        value = fold_expr(stmt.value)
        if target is stmt.target and value is stmt.value:
            return stmt
        return Assignment(target, value)

    if isinstance(stmt, ExpressionStatement):
        folded = fold_expr(stmt.expr)
        if folded is stmt.expr:
            return stmt
        return ExpressionStatement(folded)

    if isinstance(stmt, If):
        cond = fold_expr(stmt.condition)
        then_ = [fold_stmt(s) for s in stmt.then_body]
        elifs = [
            (fold_expr(c), [fold_stmt(s) for s in b]) for c, b in stmt.elif_clauses
        ]
        else_ = [fold_stmt(s) for s in stmt.else_body] if stmt.else_body else None
        return If(cond, then_, elifs, else_)

    if isinstance(stmt, For):
        init = fold_stmt(stmt.init)
        cond = fold_expr(stmt.condition)
        update = fold_expr(stmt.update)
        body_ = [fold_stmt(s) for s in stmt.body]
        return For(init, cond, update, body_)

    if isinstance(stmt, While):
        cond = fold_expr(stmt.condition)
        body_ = [fold_stmt(s) for s in stmt.body]
        return While(cond, body_)

    if isinstance(stmt, DoWhile):
        body_ = [fold_stmt(s) for s in stmt.body]
        cond = fold_expr(stmt.condition)
        return DoWhile(body_, cond)

    if isinstance(stmt, Switch):
        expr = fold_expr(stmt.expr)
        cases = [(fold_expr(v), [fold_stmt(s) for s in b]) for v, b in stmt.cases]
        default = (
            [fold_stmt(s) for s in stmt.default_body] if stmt.default_body else None
        )
        return Switch(expr, cases, default)

    if isinstance(stmt, CompoundAssignment):
        target = fold_expr(stmt.target)
        value = fold_expr(stmt.value)
        if target is stmt.target and value is stmt.value:
            return stmt
        return CompoundAssignment(stmt.op, target, value)

    return stmt


def fold_constants(stmts: list[Stmt]) -> list[Stmt]:
    """Fold constants across a list of statements."""
    return [fold_stmt(s) for s in stmts]
