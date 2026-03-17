"""Tests for ombra Phase 5 — Optimization & Tooling.

Covers:
- Constant folding (_optimizer.py)
- Dead-code elimination (_dce.py)
- Shader caching (_cache.py)
- Validation (_validation.py)
"""

from __future__ import annotations

from ombra.ast._expressions import (
    BinaryOp,
    ConstructorCall,
    FieldAccess,
    FunctionCall,
    Literal,
    UnaryOp,
    Variable,
)
from ombra.ast._statements import (
    Assignment,
    Declaration,
    ExpressionStatement,
    For,
    If,
    Return,
)
from ombra.compiler._cache import ShaderCache, hash_sources
from ombra.compiler._dce import eliminate_dead_functions, find_referenced_names
from ombra.compiler._optimizer import fold_constants, fold_expr, fold_stmt
from ombra.decorators import ShaderFunction
from ombra.glsl import (
    ShaderBuilder,
    ShaderStage,
)
from ombra.glsl._validation import (
    Severity,
    ValidationError,
    validate_builder,
    validate_source,
)
from ombra.types import Bool, Float, Int, Vec3

# ═════════════════════════════════════════════════════════════════════
# Constant Folding
# ═════════════════════════════════════════════════════════════════════


class TestFoldExpr:
    """Constant folding on individual expressions."""

    def test_literal_unchanged(self) -> None:
        lit = Literal(1.0, Float)
        assert fold_expr(lit) is lit

    def test_variable_unchanged(self) -> None:
        v = Variable("x", Float)
        assert fold_expr(v) is v

    def test_literal_addition(self) -> None:
        expr = BinaryOp("+", Literal(1.0, Float), Literal(2.0, Float), Float)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value == 3.0

    def test_literal_subtraction(self) -> None:
        expr = BinaryOp("-", Literal(5.0, Float), Literal(3.0, Float), Float)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value == 2.0

    def test_literal_multiplication(self) -> None:
        expr = BinaryOp("*", Literal(3.0, Float), Literal(4.0, Float), Float)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value == 12.0

    def test_literal_division(self) -> None:
        expr = BinaryOp("/", Literal(10.0, Float), Literal(2.0, Float), Float)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value == 5.0

    def test_literal_modulo(self) -> None:
        expr = BinaryOp("%", Literal(7, Int), Literal(3, Int), Int)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value == 1

    def test_comparison_folding(self) -> None:
        expr = BinaryOp("<", Literal(1.0, Float), Literal(2.0, Float), Bool)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value is True

    def test_comparison_false(self) -> None:
        expr = BinaryOp(">", Literal(1.0, Float), Literal(2.0, Float), Bool)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value is False

    def test_negate_literal(self) -> None:
        expr = UnaryOp("-", Literal(3.0, Float), Float)
        result = fold_expr(expr)
        assert isinstance(result, Literal)
        assert result.value == -3.0

    def test_add_zero_identity(self) -> None:
        x = Variable("x", Float)
        expr = BinaryOp("+", x, Literal(0.0, Float), Float)
        result = fold_expr(expr)
        assert result is x

    def test_zero_plus_x_identity(self) -> None:
        x = Variable("x", Float)
        expr = BinaryOp("+", Literal(0.0, Float), x, Float)
        result = fold_expr(expr)
        assert result is x

    def test_subtract_zero_identity(self) -> None:
        x = Variable("x", Float)
        expr = BinaryOp("-", x, Literal(0.0, Float), Float)
        result = fold_expr(expr)
        assert result is x

    def test_multiply_one_identity(self) -> None:
        x = Variable("x", Float)
        expr = BinaryOp("*", x, Literal(1.0, Float), Float)
        result = fold_expr(expr)
        assert result is x

    def test_one_times_x_identity(self) -> None:
        x = Variable("x", Float)
        expr = BinaryOp("*", Literal(1.0, Float), x, Float)
        result = fold_expr(expr)
        assert result is x

    def test_multiply_zero(self) -> None:
        x = Variable("x", Float)
        zero = Literal(0.0, Float)
        expr = BinaryOp("*", x, zero, Float)
        result = fold_expr(expr)
        assert result is zero

    def test_zero_times_x(self) -> None:
        x = Variable("x", Float)
        zero = Literal(0.0, Float)
        expr = BinaryOp("*", zero, x, Float)
        result = fold_expr(expr)
        assert result is zero

    def test_divide_by_one_identity(self) -> None:
        x = Variable("x", Float)
        expr = BinaryOp("/", x, Literal(1.0, Float), Float)
        result = fold_expr(expr)
        assert result is x

    def test_nested_fold(self) -> None:
        # (2.0 + 3.0) * x → 5.0 * x
        inner = BinaryOp("+", Literal(2.0, Float), Literal(3.0, Float), Float)
        x = Variable("x", Float)
        outer = BinaryOp("*", inner, x, Float)
        result = fold_expr(outer)
        assert isinstance(result, BinaryOp)
        assert isinstance(result.left, Literal)
        assert result.left.value == 5.0
        assert result.right is x

    def test_division_by_zero_no_fold(self) -> None:
        expr = BinaryOp("/", Literal(1.0, Float), Literal(0.0, Float), Float)
        result = fold_expr(expr)
        # Should remain a BinaryOp (no fold on divide by zero).
        assert isinstance(result, BinaryOp)

    def test_function_call_args_folded(self) -> None:
        arg = BinaryOp("+", Literal(1.0, Float), Literal(2.0, Float), Float)
        call = FunctionCall("sin", [arg], Float)
        result = fold_expr(call)
        assert isinstance(result, FunctionCall)
        assert isinstance(result.args[0], Literal)
        assert result.args[0].value == 3.0

    def test_constructor_args_folded(self) -> None:
        arg = BinaryOp("*", Literal(2.0, Float), Literal(3.0, Float), Float)
        ctor = ConstructorCall(
            Vec3, [arg, Literal(0.0, Float), Literal(1.0, Float)], Vec3
        )
        result = fold_expr(ctor)
        assert isinstance(result, ConstructorCall)
        assert isinstance(result.args[0], Literal)
        assert result.args[0].value == 6.0

    def test_field_access_inner_folded(self) -> None:
        inner = BinaryOp("+", Variable("v", Vec3), Variable("v", Vec3), Vec3)
        fa = FieldAccess(inner, "xyz")
        result = fold_expr(fa)
        # FieldAccess with non-foldable inner — shouldn't crash.
        assert isinstance(result, FieldAccess)


class TestFoldStmt:
    """Constant folding across statements."""

    def test_return_folded(self) -> None:
        stmt = Return(BinaryOp("+", Literal(1.0, Float), Literal(2.0, Float), Float))
        result = fold_stmt(stmt)
        assert isinstance(result, Return)
        assert isinstance(result.value, Literal)
        assert result.value.value == 3.0

    def test_declaration_folded(self) -> None:
        stmt = Declaration(
            "x",
            Float,
            BinaryOp("*", Literal(3.0, Float), Literal(1.0, Float), Float),
        )
        result = fold_stmt(stmt)
        assert isinstance(result, Declaration)
        assert isinstance(result.initializer, Literal)
        assert result.initializer.value == 3.0

    def test_assignment_folded(self) -> None:
        target = Variable("x", Float)
        value = BinaryOp("+", Literal(0.0, Float), Variable("y", Float), Float)
        stmt = Assignment(target, value)
        result = fold_stmt(stmt)
        assert isinstance(result, Assignment)
        # 0.0 + y → y
        assert isinstance(result.value, Variable)
        assert result.value.name == "y"

    def test_if_condition_folded(self) -> None:
        cond = BinaryOp("<", Literal(1.0, Float), Literal(2.0, Float), Bool)
        stmt = If(cond, [Return(Literal(1.0, Float))])
        result = fold_stmt(stmt)
        assert isinstance(result, If)
        assert isinstance(result.condition, Literal)
        assert result.condition.value is True

    def test_for_body_folded(self) -> None:
        init = Declaration("i", Int, Literal(0, Int))
        cond = BinaryOp("<", Variable("i", Int), Literal(10, Int), Bool)
        update = BinaryOp("+", Variable("i", Int), Literal(1, Int), Int)
        body = [Return(BinaryOp("+", Literal(1.0, Float), Literal(2.0, Float), Float))]
        stmt = For(init, cond, update, body)
        result = fold_stmt(stmt)
        assert isinstance(result, For)
        # Body's return should be folded.
        ret = result.body[0]
        assert isinstance(ret, Return)
        assert isinstance(ret.value, Literal)
        assert ret.value.value == 3.0

    def test_fold_constants_list(self) -> None:
        stmts = [
            Declaration(
                "a",
                Float,
                BinaryOp("*", Literal(1.0, Float), Variable("x", Float), Float),
            ),
            Return(BinaryOp("+", Literal(0.0, Float), Variable("a", Float), Float)),
        ]
        results = fold_constants(stmts)
        # 1.0 * x → x
        assert isinstance(results[0], Declaration)
        assert isinstance(results[0].initializer, Variable)
        # 0.0 + a → a
        assert isinstance(results[1], Return)
        assert isinstance(results[1].value, Variable)

    def test_expression_statement_folded(self) -> None:
        stmt = ExpressionStatement(
            BinaryOp("+", Literal(1.0, Float), Literal(2.0, Float), Float)
        )
        result = fold_stmt(stmt)
        assert isinstance(result, ExpressionStatement)
        assert isinstance(result.expr, Literal)
        assert result.expr.value == 3.0


# ═════════════════════════════════════════════════════════════════════
# Dead-Code Elimination
# ═════════════════════════════════════════════════════════════════════


class TestDeadCodeElimination:
    """Tests for eliminate_dead_functions."""

    def _make_fn(self, name: str, calls: list[str]) -> ShaderFunction:
        """Create a ShaderFunction that calls the given function names."""
        body = []
        for c in calls:
            body.append(Return(FunctionCall(c, [Literal(1.0, Float)], Float)))
        if not body:
            body.append(Return(Literal(0.0, Float)))
        return ShaderFunction(name, [("x", Float)], Float, body)

    def test_single_entry(self) -> None:
        fns = [self._make_fn("main", [])]
        result = eliminate_dead_functions(fns, {"main"})
        assert len(result) == 1
        assert result[0].name == "main"

    def test_unused_function_removed(self) -> None:
        fns = [
            self._make_fn("main", []),
            self._make_fn("unused", []),
        ]
        result = eliminate_dead_functions(fns, {"main"})
        assert len(result) == 1
        assert result[0].name == "main"

    def test_transitive_dependency_kept(self) -> None:
        fns = [
            self._make_fn("main", ["helper"]),
            self._make_fn("helper", ["leaf"]),
            self._make_fn("leaf", []),
            self._make_fn("unused", []),
        ]
        result = eliminate_dead_functions(fns, {"main"})
        names = [fn.name for fn in result]
        assert "main" in names
        assert "helper" in names
        assert "leaf" in names
        assert "unused" not in names

    def test_preserves_order(self) -> None:
        fns = [
            self._make_fn("leaf", []),
            self._make_fn("helper", ["leaf"]),
            self._make_fn("main", ["helper"]),
        ]
        result = eliminate_dead_functions(fns, {"main"})
        names = [fn.name for fn in result]
        assert names == ["leaf", "helper", "main"]

    def test_empty_entry_names(self) -> None:
        fns = [self._make_fn("foo", [])]
        result = eliminate_dead_functions(fns, set())
        assert result == []

    def test_multiple_entry_points(self) -> None:
        fns = [
            self._make_fn("a", []),
            self._make_fn("b", []),
            self._make_fn("c", []),
        ]
        result = eliminate_dead_functions(fns, {"a", "c"})
        names = [fn.name for fn in result]
        assert "a" in names
        assert "c" in names
        assert "b" not in names


class TestFindReferencedNames:
    """Tests for find_referenced_names."""

    def test_from_ast(self) -> None:
        fn = ShaderFunction(
            "foo",
            [("x", Float)],
            Float,
            [Return(FunctionCall("bar", [Variable("x", Float)], Float))],
        )
        names = find_referenced_names([fn], [])
        assert "bar" in names
        assert "x" in names

    def test_from_raw_lines(self) -> None:
        names = find_referenced_names([], ["float y = u_mvp * pos;"])
        assert "u_mvp" in names
        assert "pos" in names
        assert "y" in names


# ═════════════════════════════════════════════════════════════════════
# Shader Caching
# ═════════════════════════════════════════════════════════════════════


class TestShaderCache:
    """Tests for ShaderCache."""

    def test_cache_miss_then_hit(self) -> None:
        cache = ShaderCache()
        b = ShaderBuilder()
        b.add_vertex_lines(["void main() { gl_Position = vec4(0); }"])
        b.add_fragment_lines(["void main() { f_color = vec4(1); }"])

        key1, (v1, f1) = cache.get_or_build(b)
        assert len(cache) == 1

        # Same builder → same key.
        key2, (v2, f2) = cache.get_or_build(b)
        assert key1 == key2
        assert v1 == v2
        assert f1 == f2
        assert len(cache) == 1

    def test_different_builders_different_keys(self) -> None:
        cache = ShaderCache()
        b1 = ShaderBuilder()
        b1.add_vertex_lines(["void main() { gl_Position = vec4(0); }"])
        b1.add_fragment_lines(["void main() { f_color = vec4(1); }"])

        b2 = ShaderBuilder()
        b2.add_vertex_lines(["void main() { gl_Position = vec4(1); }"])
        b2.add_fragment_lines(["void main() { f_color = vec4(0); }"])

        key1, _ = cache.get_or_build(b1)
        key2, _ = cache.get_or_build(b2)
        assert key1 != key2
        assert len(cache) == 2

    def test_invalidate(self) -> None:
        cache = ShaderCache()
        b = ShaderBuilder()
        b.add_vertex_lines(["void main() {}"])
        b.add_fragment_lines(["void main() {}"])
        key, _ = cache.get_or_build(b)
        assert cache.contains(key)
        cache.invalidate(key)
        assert not cache.contains(key)

    def test_clear(self) -> None:
        cache = ShaderCache()
        b = ShaderBuilder()
        b.add_vertex_lines(["void main() {}"])
        b.add_fragment_lines(["void main() {}"])
        cache.get_or_build(b)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_hash_sources_deterministic(self) -> None:
        h1 = hash_sources("vert", "frag")
        h2 = hash_sources("vert", "frag")
        assert h1 == h2

    def test_hash_sources_order_matters(self) -> None:
        h1 = hash_sources("a", "b")
        h2 = hash_sources("b", "a")
        assert h1 != h2


# ═════════════════════════════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════════════════════════════


class TestValidation:
    """Tests for shader validation."""

    def test_missing_main_warning(self) -> None:
        errors = validate_source("uniform float x;", "vertex")
        assert any(
            e.severity == Severity.WARNING and "main()" in e.message for e in errors
        )

    def test_main_present_no_warning(self) -> None:
        src = "#version 430\nvoid main() { }"
        errors = validate_source(src, "vertex")
        assert not any("main()" in e.message for e in errors)

    def test_unused_uniform_warning(self) -> None:
        src = "#version 430\nuniform float u_scale;\nvoid main() { gl_Position = vec4(0); }"
        errors = validate_source(src, "vertex")
        assert any("u_scale" in e.message for e in errors)

    def test_used_uniform_no_warning(self) -> None:
        src = "#version 430\nuniform float u_scale;\nvoid main() { gl_Position = vec4(u_scale); }"
        errors = validate_source(src, "vertex")
        assert not any("u_scale" in e.message for e in errors)

    def test_gl_point_coord_in_vertex(self) -> None:
        b = ShaderBuilder()
        b.add_vertex_lines(["void main() { vec2 p = gl_PointCoord; }"])
        b.add_fragment_lines(["void main() { }"])
        errors = validate_builder(b)
        assert any(
            e.severity == Severity.ERROR and "gl_PointCoord" in e.message
            for e in errors
        )

    def test_gl_frag_coord_in_vertex(self) -> None:
        b = ShaderBuilder()
        b.add_vertex_lines(["void main() { vec4 f = gl_FragCoord; }"])
        b.add_fragment_lines(["void main() { }"])
        errors = validate_builder(b)
        assert any(
            e.severity == Severity.ERROR and "gl_FragCoord" in e.message for e in errors
        )

    def test_layout_below_330(self) -> None:
        src = "#version 120\nlayout(location = 0) out vec4 f_color;\nvoid main() { }"
        errors = validate_source(src, "fragment")
        assert any(
            e.severity == Severity.ERROR and "layout" in e.message for e in errors
        )

    def test_layout_at_430_ok(self) -> None:
        src = "#version 430\nlayout(location = 0) out vec4 f_color;\nvoid main() { }"
        errors = validate_source(src, "fragment")
        assert not any("layout" in e.message for e in errors)

    def test_validation_error_str(self) -> None:
        err = ValidationError(Severity.ERROR, "vertex", "test message")
        assert "[error]" in str(err)
        assert "(vertex)" in str(err)
        assert "test message" in str(err)

    def test_clean_builder_no_errors(self) -> None:
        b = ShaderBuilder()
        b.add_uniform("mat4", "u_mvp", ShaderStage.VERTEX)
        b.add_vertex_lines(
            [
                "void main() {",
                "    gl_Position = u_mvp * vec4(0, 0, 0, 1);",
                "}",
            ]
        )
        b.add_fragment_lines(
            [
                "void main() {",
                "    f_color = vec4(1.0);",
                "}",
            ]
        )
        errors = validate_builder(b)
        # Only expected: no errors beyond potential missing uniform warnings in frag stage.
        error_level = [e for e in errors if e.severity == Severity.ERROR]
        assert len(error_level) == 0
