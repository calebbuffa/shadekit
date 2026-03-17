"""Tests for ombra Phase 4 — @shader_function, DependencyGraph, Builder integration."""

from __future__ import annotations

import pytest

from ombra.ast import (
    Declaration,
    FunctionCall,
    Literal,
    Return,
    Variable,
)
from ombra.compiler import (
    CircularDependencyError,
    DependencyGraph,
)
from ombra.decorators import ShaderFunction, shader_function
from ombra.glsl import emit, emit_function
from ombra.glsl._builtins import clamp, cross, dot, mix, normalize, vec3
from ombra.types import Float, Vec3


class TestGlslFunction:
    def test_simple_function(self) -> None:
        @shader_function
        def luminance(c: Vec3) -> Float:
            return dot(c, vec3(0.299, 0.587, 0.114))

        assert isinstance(luminance, ShaderFunction)
        assert luminance.name == "luminance"
        assert len(luminance.params) == 1
        assert luminance.params[0] == ("c", Vec3)
        assert luminance.return_type is Float
        assert len(luminance.body) == 1
        assert isinstance(luminance.body[0], Return)

    def test_signature(self) -> None:
        @shader_function
        def add(a: Float, b: Float) -> Float:
            return a + b

        assert add.signature() == "float add(float a, float b)"

    def test_void_return(self) -> None:
        @shader_function
        def noop(x: Float) -> None:
            pass

        assert noop.return_type is None
        assert noop.signature() == "void noop(float x)"
        assert len(noop.body) == 0  # no return captured

    def test_calling_creates_function_call(self) -> None:
        @shader_function
        def double(x: Float) -> Float:
            return x + x

        v = Variable("val", Float)
        call = double(v)
        assert isinstance(call, FunctionCall)
        assert call.func_name == "double"
        assert call.glsl_type is Float
        assert len(call.args) == 1

    def test_wrong_arg_count(self) -> None:
        @shader_function
        def foo(a: Float, b: Float) -> Float:
            return a + b

        with pytest.raises(TypeError, match="takes 2"):
            foo(Variable("x", Float))

    def test_missing_annotation(self) -> None:
        with pytest.raises(TypeError, match="must have a shader type annotation"):

            @shader_function
            def bad(x) -> Float:  # type: ignore[return-value]
                return x

    def test_non_expr_return(self) -> None:
        with pytest.raises(TypeError, match="must return an Expr or None"):

            @shader_function
            def bad(x: Float) -> Float:
                return 42  # type: ignore[return-value]


class TestEmitFunction:
    def test_simple(self) -> None:
        @shader_function
        def luminance(c: Vec3) -> Float:
            return dot(c, vec3(0.299, 0.587, 0.114))

        src = emit_function(luminance)
        assert "float luminance(vec3 c) {" in src
        assert "return dot(c, vec3(0.299, 0.587, 0.114));" in src
        assert src.strip().endswith("}")

    def test_binary_ops(self) -> None:
        @shader_function
        def add_scaled(a: Vec3, b: Vec3, t: Float) -> Vec3:
            return a + b * t

        src = emit_function(add_scaled)
        assert "vec3 add_scaled(vec3 a, vec3 b, float t) {" in src
        assert "return a + b * t;" in src or "return a + (b * t);" in src

    def test_chained_builtins(self) -> None:
        @shader_function
        def safe_normalize(v: Vec3) -> Vec3:
            return normalize(v)

        src = emit_function(safe_normalize)
        assert "return normalize(v);" in src


class TestBuiltins:
    def test_dot(self) -> None:
        a = Variable("a", Vec3)
        b = Variable("b", Vec3)
        result = dot(a, b)
        assert isinstance(result, FunctionCall)
        assert result.func_name == "dot"
        assert result.glsl_type is Float

    def test_vec3_constructor(self) -> None:
        result = vec3(1.0, 2.0, 3.0)
        assert emit(result) == "vec3(1.0, 2.0, 3.0)"

    def test_mix(self) -> None:
        a = Variable("a", Vec3)
        b = Variable("b", Vec3)
        t = Variable("t", Float)
        result = mix(a, b, t)
        assert result.func_name == "mix"
        assert result.glsl_type is Vec3

    def test_clamp(self) -> None:
        x = Variable("x", Float)
        lo = Literal(0.0, Float)
        hi = Literal(1.0, Float)
        result = clamp(x, lo, hi)
        assert emit(result) == "clamp(x, 0.0, 1.0)"

    def test_normalize(self) -> None:
        v = Variable("v", Vec3)
        result = normalize(v)
        assert emit(result) == "normalize(v)"

    def test_cross(self) -> None:
        a = Variable("a", Vec3)
        b = Variable("b", Vec3)
        result = cross(a, b)
        assert emit(result) == "cross(a, b)"


class TestDependencyGraph:
    def test_single_function(self) -> None:
        @shader_function
        def foo(x: Float) -> Float:
            return x + x

        graph = DependencyGraph()
        graph.add(foo)
        order = graph.resolve()
        assert order == [foo]

    def test_dependency_order(self) -> None:
        @shader_function
        def helper(x: Float) -> Float:
            return x * x

        @shader_function
        def caller(a: Float) -> Float:
            return helper(a) + a

        graph = DependencyGraph()
        graph.add(caller)
        graph.add(helper)
        order = graph.resolve()

        # helper must come before caller
        assert order.index(helper) < order.index(caller)

    def test_diamond_dependency(self) -> None:
        @shader_function
        def base(x: Float) -> Float:
            return x

        @shader_function
        def left(x: Float) -> Float:
            return base(x) + x

        @shader_function
        def right(x: Float) -> Float:
            return base(x) * x

        @shader_function
        def top(x: Float) -> Float:
            return left(x) + right(x)

        graph = DependencyGraph()
        graph.add(top)
        graph.add(left)
        graph.add(right)
        graph.add(base)
        order = graph.resolve()

        assert order.index(base) < order.index(left)
        assert order.index(base) < order.index(right)
        assert order.index(left) < order.index(top)
        assert order.index(right) < order.index(top)

    def test_circular_dependency_detected(self) -> None:
        # Manually create circular functions (can't do via decorator).
        fn_a = ShaderFunction("a", [("x", Float)], Float, [])
        fn_b = ShaderFunction("b", [("x", Float)], Float, [])

        # a calls b, b calls a.
        fn_a.body.append(Return(FunctionCall("b", [Variable("x", Float)], Float)))
        fn_b.body.append(Return(FunctionCall("a", [Variable("x", Float)], Float)))

        graph = DependencyGraph()
        graph.add(fn_a)
        graph.add(fn_b)

        with pytest.raises(CircularDependencyError):
            graph.resolve()

    def test_unknown_function_calls_ignored(self) -> None:
        """Calls to built-in functions (not in graph) are ignored."""

        @shader_function
        def shade(n: Vec3) -> Vec3:
            return normalize(n)

        graph = DependencyGraph()
        graph.add(shade)
        order = graph.resolve()
        assert order == [shade]

    def test_duplicate_add(self) -> None:
        @shader_function
        def foo(x: Float) -> Float:
            return x

        graph = DependencyGraph()
        graph.add(foo)
        graph.add(foo)  # should not duplicate
        assert len(graph.resolve()) == 1


class TestBuilderIntegration:
    def test_add_glsl_function(self) -> None:
        from ombra.glsl import ShaderBuilder, ShaderStage

        @shader_function
        def double(x: Float) -> Float:
            return x + x

        b = ShaderBuilder()
        b.add_glsl_function(double, ShaderStage.FRAGMENT)
        b.add_fragment_lines(["void main() {", "  float v = double(1.0);", "}"])
        _, frag = b.build()

        assert "float double(float x) {" in frag
        assert "return x + x;" in frag
        assert "void main() {" in frag

    def test_add_glsl_function_with_deps(self) -> None:
        from ombra.glsl import ShaderBuilder, ShaderStage

        @shader_function
        def helper(x: Float) -> Float:
            return x * x

        @shader_function
        def caller(a: Float) -> Float:
            return helper(a) + a

        b = ShaderBuilder()
        b.add_glsl_function(caller, ShaderStage.FRAGMENT)
        b.add_glsl_function(helper, ShaderStage.FRAGMENT)
        b.add_fragment_lines(["void main() {", "}"])
        _, frag = b.build()

        # helper must appear before caller in the output
        helper_pos = frag.index("float helper(float x)")
        caller_pos = frag.index("float caller(float a)")
        assert helper_pos < caller_pos

    def test_add_vertex_stmts(self) -> None:
        from ombra.glsl import ShaderBuilder

        b = ShaderBuilder()

        decl = Declaration("x", Float, Literal(1.0, Float))
        b.add_vertex_stmts(decl)

        vert, _ = b.build()

        assert "void main() {" in vert
        assert "    float x = 1.0;" in vert
        assert "}" in vert

    def test_add_fragment_stmts_list(self) -> None:
        from ombra.glsl import ShaderBuilder

        b = ShaderBuilder()
        stmts = [
            Declaration("v", Float, Literal(2.0, Float)),
        ]
        b.add_fragment_stmts(stmts)
        _, frag = b.build()

        assert "float v = 2.0;" in frag

    def test_backwards_compatible(self) -> None:
        """String-based API still works unchanged."""
        from ombra.glsl import ShaderBuilder

        b = ShaderBuilder()
        b.add_vertex_lines(["void main() {", "  gl_Position = vec4(0);", "}"])
        b.add_fragment_lines(["void main() {", "  f_color = vec4(1);", "}"])
        vert, frag = b.build()

        assert "gl_Position = vec4(0);" in vert
        assert "f_color = vec4(1);" in frag

    def test_clone_preserves_glsl_functions(self) -> None:
        from ombra.glsl import ShaderBuilder, ShaderStage

        @shader_function
        def foo(x: Float) -> Float:
            return x

        b = ShaderBuilder()
        b.add_glsl_function(foo, ShaderStage.FRAGMENT)
        b2 = b.clone()
        b2.add_fragment_lines(["void main() { }"])
        _, frag = b2.build()
        assert "float foo(float x) {" in frag


class TestEndToEnd:
    def test_height_color(self) -> None:
        """The motivating example from the ombra design plan."""

        @shader_function
        def height_color(pos: Vec3) -> Vec3:
            t = clamp(pos.z, Literal(0.0, Float), Literal(1.0, Float))
            return mix(vec3(0.2, 0.4, 0.1), vec3(0.8, 0.9, 1.0), t)

        src = emit_function(height_color)
        assert "vec3 height_color(vec3 pos)" in src
        assert "mix(" in src
        assert "clamp(" in src

    def test_full_pipeline(self) -> None:
        """Decorator → DependencyGraph → Builder → GLSL output."""
        from ombra.glsl import ShaderBuilder, ShaderStage

        @shader_function
        def square(x: Float) -> Float:
            return x * x

        @shader_function
        def dist_squared(a: Vec3, b: Vec3) -> Float:
            d = a - b
            return dot(d, d)

        b = ShaderBuilder()
        b.add_glsl_function(dist_squared, ShaderStage.FRAGMENT)
        b.add_glsl_function(square, ShaderStage.FRAGMENT)
        b.add_fragment_lines(["void main() {", "  // use functions", "}"])
        _, frag = b.build()

        assert "float square(float x) {" in frag
        assert "float dist_squared(vec3 a, vec3 b) {" in frag
        assert "void main() {" in frag
