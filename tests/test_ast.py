"""Tests for shadekit Phase 3 — Expression AST, Statement AST, and GLSL emitter."""

from __future__ import annotations

import pytest

from shadekit.ast import (
    Assignment,
    BinaryOp,
    ConstructorCall,
    Declaration,
    Discard,
    ExpressionStatement,
    FieldAccess,
    For,
    FunctionCall,
    If,
    Literal,
    Return,
    UnaryOp,
    Variable,
)
from shadekit.glsl import emit
from shadekit.types import (
    Bool,
    Float,
    Int,
    Mat4,
    Vec2,
    Vec3,
    Vec4,
)


class TestLiteral:
    def test_float(self) -> None:
        n = Literal(1.0, Float)
        assert n.value == 1.0
        assert n.glsl_type is Float

    def test_int(self) -> None:
        n = Literal(42, Int)
        assert n.value == 42
        assert n.glsl_type is Int

    def test_bool(self) -> None:
        n = Literal(True, Bool)
        assert n.value is True
        assert n.glsl_type is Bool


class TestVariable:
    def test_basic(self) -> None:
        v = Variable("pos", Vec3)
        assert v.name == "pos"
        assert v.glsl_type is Vec3


class TestBinaryOp:
    def test_direct(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        op = BinaryOp("+", a, b, Float)
        assert op.op == "+"
        assert op.left is a
        assert op.right is b
        assert op.glsl_type is Float


class TestUnaryOp:
    def test_direct(self) -> None:
        x = Variable("x", Float)
        op = UnaryOp("-", x, Float)
        assert op.op == "-"
        assert op.operand is x
        assert op.glsl_type is Float


class TestFunctionCall:
    def test_normalize(self) -> None:
        v = Variable("v", Vec3)
        call = FunctionCall("normalize", [v], Vec3)
        assert call.func_name == "normalize"
        assert call.args == [v]
        assert call.glsl_type is Vec3


class TestConstructorCall:
    def test_vec4_from_vec3(self) -> None:
        pos = Variable("pos", Vec3)
        w = Literal(1.0, Float)
        ctor = ConstructorCall(Vec4, [pos, w], Vec4)
        assert ctor.target_type is Vec4
        assert len(ctor.args) == 2


class TestFieldAccess:
    def test_swizzle_single(self) -> None:
        v = Variable("v", Vec3)
        fa = v.x
        assert isinstance(fa, FieldAccess)
        assert fa.field == "x"
        assert fa.glsl_type is Float  # single component → scalar

    def test_swizzle_multi(self) -> None:
        v = Variable("v", Vec4)
        fa = v.xyz
        assert isinstance(fa, FieldAccess)
        assert fa.field == "xyz"
        assert fa.glsl_type is Vec3

    def test_swizzle_rgba(self) -> None:
        v = Variable("color", Vec4)
        fa = v.rgb
        assert fa.glsl_type is Vec3

    def test_swizzle_out_of_range(self) -> None:
        v = Variable("v", Vec2)
        with pytest.raises(AttributeError, match="out of range"):
            _ = v.z  # Vec2 has no z component

    def test_dunder_not_intercepted(self) -> None:
        v = Variable("v", Vec3)
        with pytest.raises(AttributeError):
            _ = v.__foo__


class TestOperatorOverloading:
    def test_add(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        result = a + b
        assert isinstance(result, BinaryOp)
        assert result.op == "+"
        assert result.glsl_type is Float

    def test_sub(self) -> None:
        a = Variable("a", Vec3)
        b = Variable("b", Vec3)
        result = a - b
        assert isinstance(result, BinaryOp)
        assert result.op == "-"

    def test_mul(self) -> None:
        a = Variable("a", Vec3)
        b = Variable("b", Float)
        result = a * b
        assert isinstance(result, BinaryOp)
        assert result.op == "*"

    def test_div(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        result = a / b
        assert isinstance(result, BinaryOp)
        assert result.op == "/"

    def test_neg(self) -> None:
        x = Variable("x", Float)
        result = -x
        assert isinstance(result, UnaryOp)
        assert result.op == "-"
        assert result.glsl_type is Float

    def test_comparison(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        lt = a < b
        assert isinstance(lt, BinaryOp)
        assert lt.op == "<"

    def test_chained_ops(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        c = Variable("c", Float)
        result = a * b + c
        assert isinstance(result, BinaryOp)
        assert result.op == "+"
        assert isinstance(result.left, BinaryOp)
        assert result.left.op == "*"

    def test_matrix_vector_mul(self) -> None:
        mvp = Variable("mvp", Mat4)
        pos = Variable("pos", Vec4)
        result = mvp * pos
        assert isinstance(result, BinaryOp)
        assert result.glsl_type is Vec4


class TestEmitExpr:
    def test_float_literal(self) -> None:
        assert emit(Literal(1.0, Float)) == "1.0"

    def test_int_literal(self) -> None:
        assert emit(Literal(42, Int)) == "42"

    def test_bool_literal_true(self) -> None:
        assert emit(Literal(True, Bool)) == "true"

    def test_bool_literal_false(self) -> None:
        assert emit(Literal(False, Bool)) == "false"

    def test_variable(self) -> None:
        assert emit(Variable("pos", Vec3)) == "pos"

    def test_binary_simple(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        assert emit(a + b) == "a + b"

    def test_binary_precedence(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        c = Variable("c", Float)
        # a * b + c — no parens needed
        assert emit(a * b + c) == "a * b + c"

    def test_binary_precedence_paren(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        c = Variable("c", Float)
        # a * (b + c) — parens needed
        result = a * (b + c)
        assert emit(result) == "a * (b + c)"

    def test_unary_neg(self) -> None:
        x = Variable("x", Float)
        assert emit(-x) == "-x"

    def test_function_call(self) -> None:
        v = Variable("v", Vec3)
        call = FunctionCall("normalize", [v], Vec3)
        assert emit(call) == "normalize(v)"

    def test_function_call_multi_args(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        c = Variable("c", Float)
        call = FunctionCall("clamp", [a, b, c], Float)
        assert emit(call) == "clamp(a, b, c)"

    def test_constructor(self) -> None:
        pos = Variable("pos", Vec3)
        w = Literal(1.0, Float)
        ctor = ConstructorCall(Vec4, [pos, w], Vec4)
        assert emit(ctor) == "vec4(pos, 1.0)"

    def test_field_access(self) -> None:
        v = Variable("v", Vec3)
        fa = FieldAccess(v, "xyz")
        assert emit(fa) == "v.xyz"

    def test_swizzle_via_attr(self) -> None:
        v = Variable("v", Vec4)
        assert emit(v.xyz) == "v.xyz"

    def test_complex_expression(self) -> None:
        mvp = Variable("mvp", Mat4)
        pos = Variable("pos", Vec3)
        w = Literal(1.0, Float)
        ctor = ConstructorCall(Vec4, [pos, w], Vec4)
        result = mvp * ctor
        assert emit(result) == "mvp * vec4(pos, 1.0)"


class TestEmitStmt:
    def test_declaration_no_init(self) -> None:
        d = Declaration("x", Float)
        assert emit(d) == "float x;\n"

    def test_declaration_with_init(self) -> None:
        val = Literal(0.0, Float)
        d = Declaration("x", Float, val)
        assert emit(d) == "float x = 0.0;\n"

    def test_assignment(self) -> None:
        x = Variable("x", Float)
        val = Literal(1.0, Float)
        a = Assignment(x, val)
        assert emit(a) == "x = 1.0;\n"

    def test_return_value(self) -> None:
        v = Variable("result", Vec3)
        r = Return(v)
        assert emit(r) == "return result;\n"

    def test_return_void(self) -> None:
        r = Return()
        assert emit(r) == "return;\n"

    def test_discard(self) -> None:
        d = Discard()
        assert emit(d) == "discard;\n"

    def test_expression_statement(self) -> None:
        call = FunctionCall("barrier", [], Float)
        es = ExpressionStatement(call)
        assert emit(es) == "barrier();\n"

    def test_if_simple(self) -> None:
        cond = Variable("cond", Bool)
        body = [Return(Variable("x", Float))]
        stmt = If(cond, body)
        expected = "if (cond) {\n    return x;\n}\n"
        assert emit(stmt) == expected

    def test_if_else(self) -> None:
        cond = Variable("cond", Bool)
        then_body = [Return(Literal(1.0, Float))]
        else_body = [Return(Literal(0.0, Float))]
        stmt = If(cond, then_body, else_body=else_body)
        expected = "if (cond) {\n    return 1.0;\n} else {\n    return 0.0;\n}\n"
        assert emit(stmt) == expected

    def test_if_elif(self) -> None:
        a = Variable("a", Bool)
        b = Variable("b", Bool)
        stmt = If(
            a,
            [Return(Literal(1.0, Float))],
            elif_clauses=[(b, [Return(Literal(2.0, Float))])],
            else_body=[Return(Literal(3.0, Float))],
        )
        expected = (
            "if (a) {\n"
            "    return 1.0;\n"
            "} else if (b) {\n"
            "    return 2.0;\n"
            "} else {\n"
            "    return 3.0;\n"
            "}\n"
        )
        assert emit(stmt) == expected

    def test_for_loop(self) -> None:
        init = Declaration("i", Int, Literal(0, Int))
        cond = Variable("i", Int) < Literal(10, Int)
        update = Variable("i", Int) + Literal(1, Int)
        body = [ExpressionStatement(FunctionCall("foo", [Variable("i", Int)], Float))]
        stmt = For(init, cond, update, body)
        expected = "for (int i = 0; i < 10; i + 1) {\n    foo(i);\n}\n"
        assert emit(stmt) == expected

    def test_indentation(self) -> None:
        d = Declaration("x", Float)
        assert emit(d, indent=2) == "        float x;\n"


class TestEmitDispatch:
    def test_expr_dispatch(self) -> None:
        v = Variable("x", Float)
        assert emit(v) == "x"

    def test_stmt_dispatch(self) -> None:
        r = Return()
        assert emit(r) == "return;\n"

    def test_unknown_type(self) -> None:
        with pytest.raises(TypeError, match="Cannot emit"):
            emit(object())  # type: ignore[arg-type]


class TestEdgeCases:
    def test_nested_field_access(self) -> None:
        v = Variable("obj", Vec4)
        # obj.xyz.xy
        assert emit(v.xyz.xy) == "obj.xyz.xy"

    def test_negated_expression_in_binary(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        # a + (-b)
        result = a + (-b)
        assert emit(result) == "a + -b"

    def test_double_negation(self) -> None:
        x = Variable("x", Float)
        result = -(-x)
        assert emit(result) == "--x"

    def test_mod_operator(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        result = a % b
        assert isinstance(result, BinaryOp)
        assert result.op == "%"
        assert emit(result) == "a % b"
