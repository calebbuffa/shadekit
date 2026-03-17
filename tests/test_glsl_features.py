"""Tests for ombra Phase 7 — GLSL feature completeness.

Covers:
- New expression nodes (IndexAccess, Ternary, PostfixOp)
- Bitwise / shift operators
- Equality helpers (.eq, .ne)
- Auto-coercion of Python literals (_coerce)
- New statement nodes (While, DoWhile, Switch, Break, Continue, CompoundAssignment)
- Const declarations
- Emitter output for all new nodes
- Optimizer folding for new nodes (ternary constant folding)
- Type inference for bitwise operators
- New type classes (samplers, images, atomic_uint, void)
- New builtins (hyperbolic, rounding, packing, derivatives, relational, etc.)
- Builder extensions, vertex inputs, shared vars, layout bindings, interpolation
- Assembler output for extensions, inputs, shared vars
- Program API updates (extension, vertex_input, shared, binding, interpolation)
"""

from __future__ import annotations

import pytest

from ombra.ast import (
    Assignment,
    BinaryOp,
    Break,
    CompoundAssignment,
    Continue,
    Declaration,
    DoWhile,
    IndexAccess,
    Literal,
    PostfixOp,
    Switch,
    Ternary,
    UnaryOp,
    Variable,
    While,
    logical_and,
    logical_not,
    logical_or,
    post_decrement,
    post_increment,
    pre_decrement,
    pre_increment,
    ternary,
)
from ombra.compiler._optimizer import fold_expr, fold_stmt
from ombra.glsl import Program, emit
from ombra.glsl._emitter import emit_stmt
from ombra.types import (
    ArrayType,
    AtomicUint,
    Bool,
    BVec2,
    BVec3,
    Double,
    DVec2,
    Float,
    IImage1D,
    IImage2D,
    IImage2DArray,
    IImage3D,
    IImageCube,
    Image1D,
    Image1DArray,
    Image2D,
    Image2DArray,
    Image2DMS,
    Image3D,
    ImageBuffer,
    ImageCube,
    ImageCubeArray,
    Int,
    ISampler1D,
    ISampler1DArray,
    ISampler2D,
    ISampler2DArray,
    ISampler2DMS,
    ISampler3D,
    ISamplerBuffer,
    ISamplerCube,
    ISamplerCubeArray,
    IVec3,
    Mat4,
    Sampler1D,
    Sampler1DArray,
    Sampler1DArrayShadow,
    Sampler1DShadow,
    Sampler2D,
    Sampler2DArray,
    Sampler2DArrayShadow,
    Sampler2DMS,
    Sampler2DMSArray,
    Sampler2DRect,
    Sampler2DRectShadow,
    Sampler2DShadow,
    Sampler3D,
    SamplerBuffer,
    SamplerCube,
    SamplerCubeArray,
    SamplerCubeArrayShadow,
    SamplerCubeShadow,
    UImage1D,
    UImage2D,
    UImage2DArray,
    UImage3D,
    UImageCube,
    UInt,
    USampler1D,
    USampler1DArray,
    USampler2D,
    USampler2DArray,
    USampler2DMS,
    USampler3D,
    USamplerBuffer,
    USamplerCube,
    USamplerCubeArray,
    Vec2,
    Vec3,
    Vec4,
    Void,
    infer_binary_type,
)

# ═══════════════════════════════════════════════════════════════════════
# Expression nodes — construction
# ═══════════════════════════════════════════════════════════════════════


class TestIndexAccess:
    def test_basic(self) -> None:
        arr = Variable("data", ArrayType(Float, 10))  # type: ignore[arg-type]
        i = Variable("i", Int)
        node = IndexAccess(arr, i, Float)
        assert node.obj is arr
        assert node.index is i
        assert node.glsl_type is Float

    def test_via_getitem(self) -> None:
        arr = Variable("data", ArrayType(Float, 10))  # type: ignore[arg-type]
        i = Variable("i", Int)
        node = arr[i]
        assert isinstance(node, IndexAccess)
        assert node.glsl_type is Float

    def test_vector_getitem(self) -> None:
        v = Variable("v", Vec3)
        node = v[Literal(0, Int)]
        assert isinstance(node, IndexAccess)
        assert node.glsl_type is Float

    def test_matrix_getitem(self) -> None:
        m = Variable("m", Mat4)
        node = m[Literal(2, Int)]
        assert isinstance(node, IndexAccess)
        assert node.glsl_type is Vec4


class TestTernary:
    def test_basic(self) -> None:
        cond = Variable("flag", Bool)
        a = Variable("a", Float)
        b = Variable("b", Float)
        node = Ternary(cond, a, b, Float)
        assert node.condition is cond
        assert node.true_expr is a
        assert node.false_expr is b
        assert node.glsl_type is Float

    def test_helper(self) -> None:
        cond = Variable("flag", Bool)
        a = Variable("a", Vec3)
        b = Variable("b", Vec3)
        node = ternary(cond, a, b)
        assert isinstance(node, Ternary)
        assert node.glsl_type is Vec3

    def test_helper_coerces_literals(self) -> None:
        cond = Variable("flag", Bool)
        node = ternary(cond, 1.0, 2.0)
        assert isinstance(node, Ternary)
        assert isinstance(node.true_expr, Literal)
        assert node.true_expr.value == 1.0


class TestPostfixOp:
    def test_construction(self) -> None:
        x = Variable("x", Int)
        node = PostfixOp("++", x, Int)
        assert node.op == "++"
        assert node.operand is x
        assert node.glsl_type is Int

    def test_post_increment_helper(self) -> None:
        x = Variable("x", Int)
        node = post_increment(x)
        assert isinstance(node, PostfixOp)
        assert node.op == "++"

    def test_post_decrement_helper(self) -> None:
        x = Variable("x", Float)
        node = post_decrement(x)
        assert isinstance(node, PostfixOp)
        assert node.op == "--"


class TestPreIncDec:
    def test_pre_increment(self) -> None:
        x = Variable("x", Int)
        node = pre_increment(x)
        assert isinstance(node, UnaryOp)
        assert node.op == "++"

    def test_pre_decrement(self) -> None:
        x = Variable("x", Int)
        node = pre_decrement(x)
        assert isinstance(node, UnaryOp)
        assert node.op == "--"


# ═══════════════════════════════════════════════════════════════════════
# Bitwise and shift operators
# ═══════════════════════════════════════════════════════════════════════


class TestBitwiseOperators:
    def test_and(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        result = a & b
        assert isinstance(result, BinaryOp)
        assert result.op == "&"
        assert result.glsl_type is Int

    def test_or(self) -> None:
        a = Variable("a", UInt)
        b = Variable("b", UInt)
        result = a | b
        assert isinstance(result, BinaryOp)
        assert result.op == "|"
        assert result.glsl_type is UInt

    def test_xor(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        result = a ^ b
        assert isinstance(result, BinaryOp)
        assert result.op == "^"

    def test_lshift(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        result = a << b
        assert isinstance(result, BinaryOp)
        assert result.op == "<<"

    def test_rshift(self) -> None:
        a = Variable("a", UInt)
        b = Variable("b", UInt)
        result = a >> b
        assert isinstance(result, BinaryOp)
        assert result.op == ">>"

    def test_invert(self) -> None:
        a = Variable("a", Int)
        result = ~a
        assert isinstance(result, UnaryOp)
        assert result.op == "~"
        assert result.glsl_type is Int

    def test_rand(self) -> None:
        a = Variable("a", Int)
        result = 3 & a  # type: ignore[operator]
        assert isinstance(result, BinaryOp)
        assert result.op == "&"

    def test_ror(self) -> None:
        a = Variable("a", Int)
        result = 0xFF | a  # type: ignore[operator]
        assert isinstance(result, BinaryOp)
        assert result.op == "|"

    def test_rxor(self) -> None:
        a = Variable("a", Int)
        result = 1 ^ a  # type: ignore[operator]
        assert isinstance(result, BinaryOp)
        assert result.op == "^"


# ═══════════════════════════════════════════════════════════════════════
# Equality helpers
# ═══════════════════════════════════════════════════════════════════════


class TestEqualityHelpers:
    def test_eq(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        result = a.eq(b)
        assert isinstance(result, BinaryOp)
        assert result.op == "=="
        assert result.glsl_type is Bool

    def test_ne(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        result = a.ne(b)
        assert isinstance(result, BinaryOp)
        assert result.op == "!="
        assert result.glsl_type is Bool


# ═══════════════════════════════════════════════════════════════════════
# Auto-coercion
# ═══════════════════════════════════════════════════════════════════════


class TestAutoCoercion:
    def test_float_times_expr(self) -> None:
        """The original crash: float * Expr."""
        x = Variable("x", Float)
        result = 2.0 * x  # type: ignore[operator]
        assert isinstance(result, BinaryOp)
        assert isinstance(result.left, Literal)
        assert result.left.value == 2.0

    def test_int_plus_expr(self) -> None:
        x = Variable("x", Int)
        result = 3 + x  # type: ignore[operator]
        assert isinstance(result, BinaryOp)
        assert isinstance(result.left, Literal)
        assert result.left.value == 3

    def test_expr_minus_float(self) -> None:
        x = Variable("x", Float)
        result = x - 1.5  # type: ignore[operator]
        assert isinstance(result, BinaryOp)
        assert isinstance(result.right, Literal)
        assert result.right.value == 1.5


# ═══════════════════════════════════════════════════════════════════════
# Logical helpers
# ═══════════════════════════════════════════════════════════════════════


class TestLogicalHelpers:
    def test_logical_not(self) -> None:
        x = Variable("flag", Bool)
        result = logical_not(x)
        assert isinstance(result, UnaryOp)
        assert result.op == "!"
        assert result.glsl_type is Bool

    def test_logical_and(self) -> None:
        a = Variable("a", Bool)
        b = Variable("b", Bool)
        result = logical_and(a, b)
        assert isinstance(result, BinaryOp)
        assert result.op == "&&"

    def test_logical_or(self) -> None:
        a = Variable("a", Bool)
        b = Variable("b", Bool)
        result = logical_or(a, b)
        assert isinstance(result, BinaryOp)
        assert result.op == "||"


# ═══════════════════════════════════════════════════════════════════════
# Statement nodes — construction
# ═══════════════════════════════════════════════════════════════════════


class TestNewStatements:
    def test_compound_assignment(self) -> None:
        x = Variable("x", Float)
        node = CompoundAssignment("+=", x, Literal(1.0, Float))
        assert node.op == "+="
        assert node.target is x

    def test_while(self) -> None:
        cond = Variable("running", Bool)
        body = [Break()]
        node = While(cond, body)
        assert node.condition is cond
        assert len(node.body) == 1
        assert isinstance(node.body[0], Break)

    def test_do_while(self) -> None:
        cond = Variable("running", Bool)
        body = [Continue()]
        node = DoWhile(body, cond)
        assert node.condition is cond
        assert len(node.body) == 1

    def test_switch(self) -> None:
        expr = Variable("val", Int)
        cases = [(Literal(0, Int), [Break()]), (Literal(1, Int), [Break()])]
        default = [Break()]
        node = Switch(expr, cases, default)
        assert node.expr is expr
        assert len(node.cases) == 2
        assert node.default_body is not None
        assert len(node.default_body) == 1

    def test_break(self) -> None:
        node = Break()
        assert isinstance(node, Break)

    def test_continue(self) -> None:
        node = Continue()
        assert isinstance(node, Continue)

    def test_const_declaration(self) -> None:
        node = Declaration("x", Float, Literal(3.14, Float), const=True)
        assert node.const is True
        assert node.name == "x"


# ═══════════════════════════════════════════════════════════════════════
# Emitter — new expressions
# ═══════════════════════════════════════════════════════════════════════


class TestEmitNewExpressions:
    def test_index_access(self) -> None:
        arr = Variable("data", ArrayType(Float, 10))  # type: ignore[arg-type]
        i = Variable("i", Int)
        node = IndexAccess(arr, i, Float)
        assert emit(node) == "data[i]"

    def test_ternary(self) -> None:
        cond = Variable("flag", Bool)
        a = Variable("a", Float)
        b = Variable("b", Float)
        node = Ternary(cond, a, b, Float)
        assert emit(node) == "flag ? a : b"

    def test_postfix_increment(self) -> None:
        x = Variable("x", Int)
        node = PostfixOp("++", x, Int)
        assert emit(node) == "x++"

    def test_postfix_decrement(self) -> None:
        x = Variable("x", Int)
        node = PostfixOp("--", x, Int)
        assert emit(node) == "x--"

    def test_prefix_increment(self) -> None:
        x = Variable("x", Int)
        node = pre_increment(x)
        assert emit(node) == "++x"

    def test_prefix_decrement(self) -> None:
        x = Variable("x", Int)
        node = pre_decrement(x)
        assert emit(node) == "--x"

    def test_bitwise_and(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        assert emit(a & b) == "a & b"

    def test_bitwise_or(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        assert emit(a | b) == "a | b"

    def test_bitwise_xor(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        assert emit(a ^ b) == "a ^ b"

    def test_shift_left(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        assert emit(a << b) == "a << b"

    def test_shift_right(self) -> None:
        a = Variable("a", Int)
        b = Variable("b", Int)
        assert emit(a >> b) == "a >> b"

    def test_bitwise_not(self) -> None:
        a = Variable("a", Int)
        assert emit(~a) == "~a"

    def test_logical_not(self) -> None:
        x = Variable("flag", Bool)
        assert emit(logical_not(x)) == "!flag"

    def test_logical_and(self) -> None:
        a = Variable("a", Bool)
        b = Variable("b", Bool)
        assert emit(logical_and(a, b)) == "a && b"

    def test_logical_or(self) -> None:
        a = Variable("a", Bool)
        b = Variable("b", Bool)
        assert emit(logical_or(a, b)) == "a || b"

    def test_eq(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        assert emit(a.eq(b)) == "a == b"

    def test_ne(self) -> None:
        a = Variable("a", Float)
        b = Variable("b", Float)
        assert emit(a.ne(b)) == "a != b"


# ═══════════════════════════════════════════════════════════════════════
# Emitter — new statements
# ═══════════════════════════════════════════════════════════════════════


class TestEmitNewStatements:
    def test_compound_assignment(self) -> None:
        x = Variable("x", Float)
        stmt = CompoundAssignment("+=", x, Literal(1.0, Float))
        result = emit_stmt(stmt, indent=0)
        assert "x += 1.0;" in result

    def test_break(self) -> None:
        result = emit_stmt(Break(), indent=0)
        assert "break;" in result

    def test_continue(self) -> None:
        result = emit_stmt(Continue(), indent=0)
        assert "continue;" in result

    def test_while(self) -> None:
        cond = Variable("running", Bool)
        body = [Break()]
        stmt = While(cond, body)
        result = emit_stmt(stmt, indent=0)
        assert "while (running)" in result
        assert "break;" in result

    def test_do_while(self) -> None:
        cond = Variable("running", Bool)
        body = [Continue()]
        stmt = DoWhile(body, cond)
        result = emit_stmt(stmt, indent=0)
        assert "do {" in result
        assert "} while (running);" in result

    def test_switch(self) -> None:
        expr = Variable("val", Int)
        cases = [
            (Literal(0, Int), [Break()]),
            (
                Literal(1, Int),
                [
                    CompoundAssignment("+=", Variable("x", Float), Literal(1.0, Float)),
                    Break(),
                ],
            ),
        ]
        default = [Break()]
        stmt = Switch(expr, cases, default)
        result = emit_stmt(stmt, indent=0)
        assert "switch (val)" in result
        assert "case 0:" in result
        assert "case 1:" in result
        assert "default:" in result

    def test_const_declaration(self) -> None:
        stmt = Declaration("PI", Float, Literal(3.14159, Float), const=True)
        result = emit_stmt(stmt, indent=0)
        assert "const float PI = 3.14159;" in result

    def test_non_const_declaration(self) -> None:
        stmt = Declaration("x", Float, Literal(0.0, Float), const=False)
        result = emit_stmt(stmt, indent=0)
        assert result.startswith("float x = 0.0;")


# ═══════════════════════════════════════════════════════════════════════
# Optimizer — new nodes
# ═══════════════════════════════════════════════════════════════════════


class TestFoldNewExpressions:
    def test_ternary_literal_true(self) -> None:
        """ternary(True, a, b) → a."""
        a = Variable("a", Float)
        b = Variable("b", Float)
        t = Ternary(Literal(True, Bool), a, b, Float)
        result = fold_expr(t)
        assert isinstance(result, Variable)
        assert result.name == "a"

    def test_ternary_literal_false(self) -> None:
        """ternary(False, a, b) → b."""
        a = Variable("a", Float)
        b = Variable("b", Float)
        t = Ternary(Literal(False, Bool), a, b, Float)
        result = fold_expr(t)
        assert isinstance(result, Variable)
        assert result.name == "b"

    def test_ternary_non_literal_cond_unchanged(self) -> None:
        cond = Variable("flag", Bool)
        a = Variable("a", Float)
        b = Variable("b", Float)
        t = Ternary(cond, a, b, Float)
        result = fold_expr(t)
        assert isinstance(result, Ternary)

    def test_index_access_folds_inner(self) -> None:
        """IndexAccess with foldable inner expressions."""
        arr = Variable("data", ArrayType(Float, 10))  # type: ignore[arg-type]
        idx = BinaryOp("+", Literal(1, Int), Literal(2, Int), Int)
        node = IndexAccess(arr, idx, Float)
        result = fold_expr(node)
        assert isinstance(result, IndexAccess)
        assert isinstance(result.index, Literal)
        assert result.index.value == 3

    def test_postfix_op_folds_inner(self) -> None:
        # PostfixOp operand is a variable — nothing to fold, verify no crash.
        x = Variable("x", Int)
        node = PostfixOp("++", x, Int)
        result = fold_expr(node)
        assert isinstance(result, PostfixOp)


class TestFoldNewStatements:
    def test_compound_assignment_folds_value(self) -> None:
        x = Variable("x", Float)
        val = BinaryOp("+", Literal(1.0, Float), Literal(2.0, Float), Float)
        stmt = CompoundAssignment("+=", x, val)
        result = fold_stmt(stmt)
        assert isinstance(result, CompoundAssignment)
        assert isinstance(result.value, Literal)
        assert result.value.value == 3.0

    def test_while_folds_condition(self) -> None:
        cond = BinaryOp("<", Literal(1.0, Float), Literal(2.0, Float), Bool)
        stmt = While(cond, [Break()])
        result = fold_stmt(stmt)
        assert isinstance(result, While)
        assert isinstance(result.condition, Literal)

    def test_do_while_folds_condition(self) -> None:
        cond = BinaryOp(">", Literal(5.0, Float), Literal(3.0, Float), Bool)
        stmt = DoWhile([Break()], cond)
        result = fold_stmt(stmt)
        assert isinstance(result, DoWhile)
        assert isinstance(result.condition, Literal)

    def test_switch_folds_expr(self) -> None:
        expr = BinaryOp("+", Literal(1, Int), Literal(2, Int), Int)
        stmt = Switch(expr, [(Literal(3, Int), [Break()])], [Break()])
        result = fold_stmt(stmt)
        assert isinstance(result, Switch)
        assert isinstance(result.expr, Literal)
        assert result.expr.value == 3


# ═══════════════════════════════════════════════════════════════════════
# Type inference — bitwise operators
# ═══════════════════════════════════════════════════════════════════════


class TestBitwiseTypeInference:
    def test_int_and_int(self) -> None:
        assert infer_binary_type("&", Int, Int) is Int

    def test_uint_or_uint(self) -> None:
        assert infer_binary_type("|", UInt, UInt) is UInt

    def test_int_xor_int(self) -> None:
        assert infer_binary_type("^", Int, Int) is Int

    def test_shift_left(self) -> None:
        assert infer_binary_type("<<", Int, Int) is Int

    def test_shift_right(self) -> None:
        assert infer_binary_type(">>", UInt, UInt) is UInt

    def test_ivec_and_ivec(self) -> None:
        assert infer_binary_type("&", IVec3, IVec3) is IVec3

    def test_float_bitwise_raises(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            infer_binary_type("&", Float, Float)

    def test_vec_bitwise_raises(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            infer_binary_type("|", Vec3, Vec3)


# ═══════════════════════════════════════════════════════════════════════
# Types — samplers
# ═══════════════════════════════════════════════════════════════════════


class TestSamplerTypes:
    @pytest.mark.parametrize(
        "cls,name",
        [
            (Sampler1D, "sampler1D"),
            (Sampler2D, "sampler2D"),
            (Sampler3D, "sampler3D"),
            (SamplerCube, "samplerCube"),
            (Sampler2DArray, "sampler2DArray"),
            (Sampler1DArray, "sampler1DArray"),
            (SamplerCubeArray, "samplerCubeArray"),
            (SamplerBuffer, "samplerBuffer"),
            (Sampler2DMS, "sampler2DMS"),
            (Sampler2DMSArray, "sampler2DMSArray"),
            (Sampler2DRect, "sampler2DRect"),
        ],
    )
    def test_float_sampler(self, cls, name) -> None:
        assert cls.glsl_name == name

    @pytest.mark.parametrize(
        "cls,name",
        [
            (ISampler1D, "isampler1D"),
            (ISampler2D, "isampler2D"),
            (ISampler3D, "isampler3D"),
            (ISamplerCube, "isamplerCube"),
            (ISampler2DArray, "isampler2DArray"),
            (ISampler1DArray, "isampler1DArray"),
            (ISamplerCubeArray, "isamplerCubeArray"),
            (ISamplerBuffer, "isamplerBuffer"),
            (ISampler2DMS, "isampler2DMS"),
        ],
    )
    def test_int_sampler(self, cls, name) -> None:
        assert cls.glsl_name == name

    @pytest.mark.parametrize(
        "cls,name",
        [
            (USampler1D, "usampler1D"),
            (USampler2D, "usampler2D"),
            (USampler3D, "usampler3D"),
            (USamplerCube, "usamplerCube"),
            (USampler2DArray, "usampler2DArray"),
            (USampler1DArray, "usampler1DArray"),
            (USamplerCubeArray, "usamplerCubeArray"),
            (USamplerBuffer, "usamplerBuffer"),
            (USampler2DMS, "usampler2DMS"),
        ],
    )
    def test_uint_sampler(self, cls, name) -> None:
        assert cls.glsl_name == name

    @pytest.mark.parametrize(
        "cls,name",
        [
            (Sampler1DShadow, "sampler1DShadow"),
            (Sampler2DShadow, "sampler2DShadow"),
            (SamplerCubeShadow, "samplerCubeShadow"),
            (Sampler1DArrayShadow, "sampler1DArrayShadow"),
            (Sampler2DArrayShadow, "sampler2DArrayShadow"),
            (SamplerCubeArrayShadow, "samplerCubeArrayShadow"),
            (Sampler2DRectShadow, "sampler2DRectShadow"),
        ],
    )
    def test_shadow_sampler(self, cls, name) -> None:
        assert cls.glsl_name == name


# ═══════════════════════════════════════════════════════════════════════
# Types — images, atomic_uint, void
# ═══════════════════════════════════════════════════════════════════════


class TestImageAndSpecialTypes:
    @pytest.mark.parametrize(
        "cls,name",
        [
            (Image1D, "image1D"),
            (Image2D, "image2D"),
            (Image3D, "image3D"),
            (ImageCube, "imageCube"),
            (Image2DArray, "image2DArray"),
            (Image1DArray, "image1DArray"),
            (ImageCubeArray, "imageCubeArray"),
            (ImageBuffer, "imageBuffer"),
            (Image2DMS, "image2DMS"),
        ],
    )
    def test_float_image(self, cls, name) -> None:
        assert cls.glsl_name == name

    @pytest.mark.parametrize(
        "cls,name",
        [
            (IImage1D, "iimage1D"),
            (IImage2D, "iimage2D"),
            (IImage3D, "iimage3D"),
            (IImageCube, "iimageCube"),
            (IImage2DArray, "iimage2DArray"),
        ],
    )
    def test_int_image(self, cls, name) -> None:
        assert cls.glsl_name == name

    @pytest.mark.parametrize(
        "cls,name",
        [
            (UImage1D, "uimage1D"),
            (UImage2D, "uimage2D"),
            (UImage3D, "uimage3D"),
            (UImageCube, "uimageCube"),
            (UImage2DArray, "uimage2DArray"),
        ],
    )
    def test_uint_image(self, cls, name) -> None:
        assert cls.glsl_name == name

    def test_atomic_uint(self) -> None:
        assert AtomicUint.glsl_name == "atomic_uint"

    def test_void(self) -> None:
        assert Void.glsl_name == "void"


# ═══════════════════════════════════════════════════════════════════════
# Builtins — new functions
# ═══════════════════════════════════════════════════════════════════════


class TestNewBuiltins:
    """Verify new builtins produce correct FunctionCall nodes."""

    def test_hyperbolic_functions(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import acosh, asinh, atanh, cosh, sinh, tanh

        x = Variable("x", Float)
        for fn in [sinh, cosh, tanh, asinh, acosh, atanh]:
            result = fn(x)
            assert isinstance(result, FunctionCall)
            assert result.glsl_type is Float

    def test_rounding_functions(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import roundEven, trunc

        x = Variable("x", Float)
        for fn in [roundEven, trunc]:
            result = fn(x)
            assert isinstance(result, FunctionCall)
            assert result.glsl_type is Float

    def test_float_queries(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import isinf, isnan

        x = Variable("x", Float)
        for fn in [isnan, isinf]:
            result = fn(x)
            assert isinstance(result, FunctionCall)
            assert result.glsl_type is Bool

    def test_bit_conversion(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import (
            floatBitsToInt,
            floatBitsToUint,
            intBitsToFloat,
            uintBitsToFloat,
        )

        x = Variable("x", Float)
        r1 = floatBitsToInt(x)
        assert isinstance(r1, FunctionCall)
        assert r1.glsl_type is Int

        r2 = floatBitsToUint(x)
        assert isinstance(r2, FunctionCall)
        assert r2.glsl_type is UInt

        xi = Variable("xi", Int)
        r3 = intBitsToFloat(xi)
        assert isinstance(r3, FunctionCall)
        assert r3.glsl_type is Float

        xu = Variable("xu", UInt)
        r4 = uintBitsToFloat(xu)
        assert isinstance(r4, FunctionCall)
        assert r4.glsl_type is Float

    def test_packing(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import packUnorm2x16, unpackUnorm2x16

        v = Variable("v", Vec2)
        result = packUnorm2x16(v)
        assert isinstance(result, FunctionCall)
        assert result.glsl_type is UInt

        u = Variable("u", UInt)
        result2 = unpackUnorm2x16(u)
        assert isinstance(result2, FunctionCall)
        assert result2.glsl_type is Vec2

    def test_integer_bit_manipulation(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import bitCount, findLSB, findMSB

        x = Variable("x", Int)
        for fn in [bitCount, findLSB, findMSB]:
            result = fn(x)
            assert isinstance(result, FunctionCall)
            assert result.glsl_type is Int

    def test_vector_relational(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import lessThan

        a = Variable("a", Vec3)
        b = Variable("b", Vec3)
        result = lessThan(a, b)
        assert isinstance(result, FunctionCall)
        assert result.glsl_type is BVec3

    def test_texture_lookup(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import textureLod

        s = Variable("s", Sampler2D)
        uv = Variable("uv", Vec2)
        lod = Variable("lod", Float)
        result = textureLod(s, uv, lod)
        assert isinstance(result, FunctionCall)
        assert result.glsl_type is Vec4

    def test_fragment_derivatives(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import dFdx, dFdy, fwidth

        x = Variable("x", Float)
        for fn in [dFdx, dFdy, fwidth]:
            result = fn(x)
            assert isinstance(result, FunctionCall)
            assert result.glsl_type is Float

    def test_geometry_shader_builtins(self) -> None:
        from ombra.ast._expressions import FunctionCall
        from ombra.glsl._builtins import EmitVertex, EndPrimitive

        for fn in [EmitVertex, EndPrimitive]:
            result = fn()
            assert isinstance(result, FunctionCall)

    def test_bvec_constructors(self) -> None:
        from ombra.ast._expressions import ConstructorCall
        from ombra.glsl._builtins import bvec2

        r2 = bvec2(Literal(True, Bool), Literal(False, Bool))
        assert isinstance(r2, ConstructorCall)
        assert r2.glsl_type is BVec2

    def test_dvec_constructors(self) -> None:
        from ombra.ast._expressions import ConstructorCall
        from ombra.glsl._builtins import dvec2

        r2 = dvec2(Variable("a", Double), Variable("b", Double))
        assert isinstance(r2, ConstructorCall)
        assert r2.glsl_type is DVec2

    def test_builtin_emits_correct_glsl(self) -> None:
        from ombra.glsl._builtins import sinh

        x = Variable("x", Float)
        assert emit(sinh(x)) == "sinh(x)"


# ═══════════════════════════════════════════════════════════════════════
# Builder — new features
# ═══════════════════════════════════════════════════════════════════════


class TestBuilderNewFeatures:
    def test_extension(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430")
        b.add_extension("GL_ARB_gpu_shader5", "enable")
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        vert, frag = b.build()
        assert "#extension GL_ARB_gpu_shader5 : enable" in vert

    def test_vertex_input(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430")
        b.add_vertex_input(0, "vec3", "aPosition")
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        vert, _ = b.build()
        assert "layout(location = 0) in vec3 aPosition;" in vert

    def test_uniform_with_binding(self) -> None:
        from ombra.glsl._builder import ShaderBuilder, ShaderStage

        b = ShaderBuilder(version="430")
        b.add_uniform("sampler2D", "tex", ShaderStage.FRAGMENT, binding=3)
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        _, frag = b.build()
        assert "layout(binding = 3) uniform sampler2D tex;" in frag

    def test_varying_noperspective(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430")
        b.add_varying("vec2", "v_uv", noperspective=True)
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        vert, frag = b.build()
        assert "noperspective out vec2 v_uv;" in vert
        assert "noperspective in vec2 v_uv;" in frag

    def test_varying_centroid(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430")
        b.add_varying("vec3", "v_color", centroid=True)
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        vert, frag = b.build()
        assert "centroid out vec3 v_color;" in vert
        assert "centroid in vec3 v_color;" in frag

    def test_shared_variable(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430")
        b.add_local_size(256)
        b.add_shared("float", "cache", array_size=256)
        b.add_compute_lines(["void main() { }"])
        src = b.build_compute()
        assert "shared float cache[256];" in src

    def test_profile_core(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430", profile="core")
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        vert, _ = b.build()
        assert "#version 430 core" in vert

    def test_profile_empty(self) -> None:
        from ombra.glsl._builder import ShaderBuilder

        b = ShaderBuilder(version="430", profile="")
        b.add_vertex_lines(["void main() { }"])
        b.add_fragment_lines(["void main() { }"])
        vert, _ = b.build()
        assert "#version 430\n" in vert


# ═══════════════════════════════════════════════════════════════════════
# Assembler — new parameters
# ═══════════════════════════════════════════════════════════════════════


class TestAssemblerNewParams:
    def test_extensions_emitted_after_version(self) -> None:
        from ombra.glsl._assembler import assemble_stage

        src = assemble_stage(
            version="430 core",
            defines=[],
            uniforms=[],
            ssbos=[],
            varyings=[],
            outputs=[],
            struct_blocks=[],
            function_blocks=[],
            glsl_functions=[],
            lines=["void main() { }"],
            ast_stmts=[],
            extensions=["#extension GL_ARB_gpu_shader5 : enable"],
        )
        lines = src.splitlines()
        assert lines[0] == "#version 430 core"
        assert lines[1] == "#extension GL_ARB_gpu_shader5 : enable"

    def test_inputs_emitted(self) -> None:
        from ombra.glsl._assembler import assemble_stage

        src = assemble_stage(
            version="430 core",
            defines=[],
            uniforms=[],
            ssbos=[],
            varyings=[],
            outputs=[],
            struct_blocks=[],
            function_blocks=[],
            glsl_functions=[],
            lines=["void main() { }"],
            ast_stmts=[],
            inputs=["layout(location = 0) in vec3 aPosition;"],
        )
        assert "layout(location = 0) in vec3 aPosition;" in src

    def test_shared_vars_emitted(self) -> None:
        from ombra.glsl._assembler import assemble_stage

        src = assemble_stage(
            version="430 core",
            defines=[],
            uniforms=[],
            ssbos=[],
            varyings=[],
            outputs=[],
            struct_blocks=[],
            function_blocks=[],
            glsl_functions=[],
            lines=["void main() { }"],
            ast_stmts=[],
            shared_vars=["shared float cache[256];"],
        )
        assert "shared float cache[256];" in src


# ═══════════════════════════════════════════════════════════════════════
# Program API — new methods
# ═══════════════════════════════════════════════════════════════════════


class TestProgramNewMethods:
    def test_extension(self) -> None:
        prog = Program()
        prog.extension("GL_ARB_gpu_shader5")
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, frag = prog.build()
        assert "#extension GL_ARB_gpu_shader5 : enable" in vert

    def test_vertex_input_returns_variable(self) -> None:
        prog = Program()
        v = prog.vertex_input(0, "aPos", Vec3)
        assert v.name == "aPos"
        assert v.glsl_type is Vec3

    def test_vertex_input_appears_in_output(self) -> None:
        prog = Program()
        prog.vertex_input(0, "aPos", Vec3)
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, _ = prog.build()
        assert "layout(location = 0) in vec3 aPos;" in vert

    def test_uniform_with_binding(self) -> None:
        prog = Program()
        u = prog.uniform("tex", "sampler2D", stage="fragment", binding=2)
        assert u.name == "tex"
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        _, frag = prog.build()
        assert "layout(binding = 2) uniform sampler2D tex;" in frag

    def test_varying_noperspective(self) -> None:
        prog = Program()
        prog.varying("v_uv", Vec2, noperspective=True)
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, frag = prog.build()
        assert "noperspective out vec2 v_uv;" in vert
        assert "noperspective in vec2 v_uv;" in frag

    def test_varying_centroid(self) -> None:
        prog = Program()
        prog.varying("v_color", Vec3, centroid=True)
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, frag = prog.build()
        assert "centroid out vec3 v_color;" in vert
        assert "centroid in vec3 v_color;" in frag

    def test_shared_returns_variable(self) -> None:
        prog = Program()
        prog.local_size(256)
        v = prog.shared("cache", Float, array_size=256)
        assert v.name == "cache"
        assert v.glsl_type is Float

    def test_shared_in_compute_output(self) -> None:
        prog = Program()
        prog.local_size(256)
        prog.shared("cache", Float, array_size=256)
        prog.compute(lambda: "void main() { }")
        src = prog.build_compute()
        assert "shared float cache[256];" in src

    def test_profile_in_version(self) -> None:
        prog = Program(version="430", profile="core")
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, _ = prog.build()
        assert "#version 430 core" in vert

    def test_profile_compatibility(self) -> None:
        prog = Program(version="430", profile="compatibility")
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, _ = prog.build()
        assert "#version 430 compatibility" in vert

    def test_extension_chaining(self) -> None:
        prog = Program()
        result = prog.extension("GL_ARB_gpu_shader5")
        assert result is prog  # chaining works


# ═══════════════════════════════════════════════════════════════════════
# End-to-end — comprehensive shader with new features
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndNewFeatures:
    def test_compute_with_shared_and_extensions(self) -> None:
        prog = Program()
        prog.extension("GL_ARB_compute_shader", "require")
        prog.local_size(64)
        prog.shared("temp", Float, array_size=64)
        prog.storage(0, "Data", "float data[]", readonly=False, stage="compute")
        prog.compute(
            lambda: (
                "void main() { temp[gl_LocalInvocationIndex] = data[gl_GlobalInvocationID.x]; }"
            )
        )
        src = prog.build_compute()
        assert "#extension GL_ARB_compute_shader : require" in src
        assert "shared float temp[64];" in src
        assert "#version 430 core" in src

    def test_vertex_with_inputs_and_varyings(self) -> None:
        prog = Program()
        _pos = prog.vertex_input(0, "aPosition", Vec3)
        _normal = prog.vertex_input(1, "aNormal", Vec3)
        _v_normal = prog.varying("v_normal", Vec3, noperspective=True)
        _u_mvp = prog.uniform("u_mvp", Mat4)

        @prog.vertex
        def vs():
            return "void main() { v_normal = aNormal; gl_Position = u_mvp * vec4(aPosition, 1.0); }"

        @prog.fragment
        def fs():
            return "void main() { }"

        vert, frag = prog.build()
        assert "layout(location = 0) in vec3 aPosition;" in vert
        assert "layout(location = 1) in vec3 aNormal;" in vert
        assert "noperspective out vec3 v_normal;" in vert
        assert "noperspective in vec3 v_normal;" in frag

    def test_ast_with_new_nodes(self) -> None:
        """Build a program using AST with new expression/statement nodes."""
        prog = Program()
        u_flag = prog.uniform("u_flag", Bool)
        u_a = prog.uniform("u_a", Float)
        u_b = prog.uniform("u_b", Float)
        f_color = prog.output(0, "f_color", Vec4)

        @prog.vertex
        def vs():
            return "void main() { gl_Position = vec4(0.0); }"

        @prog.fragment
        def fs():
            from ombra.glsl._builtins import vec4

            result = Declaration("result", Float, ternary(u_flag, u_a, u_b))
            out = Assignment(f_color, vec4(Variable("result", Float)))
            return [result, out]

        vert, frag = prog.build()
        assert "u_flag ? u_a : u_b" in frag
