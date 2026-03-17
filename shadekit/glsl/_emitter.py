"""GLSL code emitter — converts AST nodes to source strings.

Usage::

    from shadekit.compiler import emit
    from shadekit.ast import Variable, Literal
    from shadekit.types import Float, Vec3

    pos   = Variable("pos", Vec3)
    scale = Literal(2.0, Float)
    print(emit(pos * scale))  # "pos * 2.0"
"""

from __future__ import annotations

from shadekit.ast._expressions import (
    _PRECEDENCE,
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
    Break,
    CompoundAssignment,
    Continue,
    Declaration,
    Discard,
    DoWhile,
    ExpressionStatement,
    For,
    If,
    Return,
    Stmt,
    Switch,
    While,
)


def emit_expr(node: Expr, parent_precedence: int = 99) -> str:
    """Emit a GLSL expression string from an AST node.

    *parent_precedence* is used to insert parentheses only when needed.
    """
    if isinstance(node, Literal):
        return _emit_literal(node)
    if isinstance(node, Variable):
        return node.name
    if isinstance(node, BinaryOp):
        return _emit_binary(node, parent_precedence)
    if isinstance(node, UnaryOp):
        return _emit_unary(node)
    if isinstance(node, FunctionCall):
        return _emit_call(node)
    if isinstance(node, ConstructorCall):
        return _emit_constructor(node)
    if isinstance(node, FieldAccess):
        return _emit_field(node)
    if isinstance(node, IndexAccess):
        return _emit_index(node)
    if isinstance(node, Ternary):
        return _emit_ternary(node, parent_precedence)
    if isinstance(node, PostfixOp):
        return _emit_postfix(node)
    raise TypeError(f"Unknown expression node: {type(node).__name__}")


def _emit_literal(node: Literal) -> str:
    if isinstance(node.value, bool):
        return "true" if node.value else "false"
    if isinstance(node.value, float):
        s = repr(node.value)
        # Ensure there's always a decimal point for GLSL.
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    # int
    return str(node.value)


def _emit_binary(node: BinaryOp, parent_prec: int) -> str:
    prec = _PRECEDENCE.get(node.op, 99)
    left = emit_expr(node.left, prec)
    right = emit_expr(node.right, prec - 1)  # right-assoc needs tighter
    result = f"{left} {node.op} {right}"
    if prec >= parent_prec:
        return f"({result})"
    return result


def _emit_call(node: FunctionCall) -> str:
    args = ", ".join(emit_expr(a) for a in node.args)
    return f"{node.func_name}({args})"


def _emit_constructor(node: ConstructorCall) -> str:
    args = ", ".join(emit_expr(a) for a in node.args)
    type_name = node.target_type.glsl_name
    return f"{type_name}({args})"


def _emit_field(node: FieldAccess) -> str:
    obj = emit_expr(node.obj, 1)  # field access binds tightest
    return f"{obj}.{node.field}"


def _emit_index(node: IndexAccess) -> str:
    obj = emit_expr(node.obj, 1)
    idx = emit_expr(node.index)
    return f"{obj}[{idx}]"


def _emit_ternary(node: Ternary, parent_prec: int) -> str:
    # Ternary has the lowest precedence (15 in GLSL spec).
    cond = emit_expr(node.condition)
    t = emit_expr(node.true_expr)
    f = emit_expr(node.false_expr)
    result = f"{cond} ? {t} : {f}"
    if parent_prec <= 14:
        return f"({result})"
    return result


def _emit_postfix(node: PostfixOp) -> str:
    operand = emit_expr(node.operand, 1)
    return f"{operand}{node.op}"


def _emit_unary(node: UnaryOp) -> str:
    operand = emit_expr(node.operand, 2)  # prefix has high precedence
    if node.op in ("++", "--"):
        return f"{node.op}{operand}"
    return f"{node.op}{operand}"


def emit_stmt(node: Stmt, indent: int = 0) -> str:
    """Emit a GLSL statement string from an AST node.

    Returns a string with a trailing newline.
    """
    pad = "    " * indent
    if isinstance(node, Declaration):
        return _emit_decl(node, pad)
    if isinstance(node, Assignment):
        return f"{pad}{emit_expr(node.target)} = {emit_expr(node.value)};\n"
    if isinstance(node, CompoundAssignment):
        return f"{pad}{emit_expr(node.target)} {node.op} {emit_expr(node.value)};\n"
    if isinstance(node, Return):
        if node.value is not None:
            return f"{pad}return {emit_expr(node.value)};\n"
        return f"{pad}return;\n"
    if isinstance(node, Discard):
        return f"{pad}discard;\n"
    if isinstance(node, Break):
        return f"{pad}break;\n"
    if isinstance(node, Continue):
        return f"{pad}continue;\n"
    if isinstance(node, If):
        return _emit_if(node, indent)
    if isinstance(node, For):
        return _emit_for(node, indent)
    if isinstance(node, While):
        return _emit_while(node, indent)
    if isinstance(node, DoWhile):
        return _emit_do_while(node, indent)
    if isinstance(node, Switch):
        return _emit_switch(node, indent)
    if isinstance(node, ExpressionStatement):
        return f"{pad}{emit_expr(node.expr)};\n"
    raise TypeError(f"Unknown statement node: {type(node).__name__}")


def _emit_decl(node: Declaration, pad: str) -> str:
    type_name = node.glsl_type.glsl_name
    prefix = "const " if node.const else ""
    if node.initializer is not None:
        return (
            f"{pad}{prefix}{type_name} {node.name} = {emit_expr(node.initializer)};\n"
        )
    return f"{pad}{prefix}{type_name} {node.name};\n"


def _emit_if(node: If, indent: int) -> str:
    pad = "    " * indent
    lines: list[str] = []
    lines.append(f"{pad}if ({emit_expr(node.condition)}) {{\n")
    for s in node.then_body:
        lines.append(emit_stmt(s, indent + 1))
    for cond, body in node.elif_clauses:
        lines.append(f"{pad}}} else if ({emit_expr(cond)}) {{\n")
        for s in body:
            lines.append(emit_stmt(s, indent + 1))
    if node.else_body:
        lines.append(f"{pad}}} else {{\n")
        for s in node.else_body:
            lines.append(emit_stmt(s, indent + 1))
    lines.append(f"{pad}}}\n")
    return "".join(lines)


def _emit_for(node: For, indent: int) -> str:
    pad = "    " * indent
    # For-loop init is a declaration or assignment, strip trailing ";\n"
    init_str = emit_stmt(node.init, 0).rstrip(";\n").strip()
    cond_str = emit_expr(node.condition)
    update_str = emit_expr(node.update)
    lines: list[str] = []
    lines.append(f"{pad}for ({init_str}; {cond_str}; {update_str}) {{\n")
    for s in node.body:
        lines.append(emit_stmt(s, indent + 1))
    lines.append(f"{pad}}}\n")
    return "".join(lines)


def _emit_while(node: While, indent: int) -> str:
    pad = "    " * indent
    lines: list[str] = []
    lines.append(f"{pad}while ({emit_expr(node.condition)}) {{\n")
    for s in node.body:
        lines.append(emit_stmt(s, indent + 1))
    lines.append(f"{pad}}}\n")
    return "".join(lines)


def _emit_do_while(node: DoWhile, indent: int) -> str:
    pad = "    " * indent
    lines: list[str] = []
    lines.append(f"{pad}do {{\n")
    for s in node.body:
        lines.append(emit_stmt(s, indent + 1))
    lines.append(f"{pad}}} while ({emit_expr(node.condition)});\n")
    return "".join(lines)


def _emit_switch(node: Switch, indent: int) -> str:
    pad = "    " * indent
    lines: list[str] = []
    lines.append(f"{pad}switch ({emit_expr(node.expr)}) {{\n")
    for val, body in node.cases:
        lines.append(f"{pad}    case {emit_expr(val)}:\n")
        for s in body:
            lines.append(emit_stmt(s, indent + 2))
    if node.default_body:
        lines.append(f"{pad}    default:\n")
        for s in node.default_body:
            lines.append(emit_stmt(s, indent + 2))
    lines.append(f"{pad}}}\n")
    return "".join(lines)


def emit(node: Expr | Stmt, indent: int = 0) -> str:
    """Emit GLSL source from any AST node.

    Returns a string (expression without semicolon; statement with).
    """
    if isinstance(node, Stmt):
        return emit_stmt(node, indent)
    if isinstance(node, Expr):
        return emit_expr(node)
    raise TypeError(f"Cannot emit {type(node).__name__}")
