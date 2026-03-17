"""Tests for shadekit.glsl.Program — high-level Program API.

Covers:
- Program construction and build
- Resource declarations (uniform, storage, varying, output, struct, define)
- Stage decorators (@prog.vertex, @prog.fragment, @prog.compute)
- from_glsl / from_compute_glsl factories
- GlslFunction inclusion
- Clone independence
- Error handling
"""

from __future__ import annotations

import pytest

from shadekit.ast import (
    Assignment,
    Block,
    Declaration,
    ExpressionStatement,
    If,
    Return,
    Variable,
)
from shadekit.decorators import shader_function
from shadekit.glsl import Program
from shadekit.glsl._builtins import dot, mat3, vec3, vec4
from shadekit.types import (
    Float,
    Int,
    Mat4,
    StructType,
    UInt,
    Vec3,
    Vec4,
)

# ═══════════════════════════════════════════════════════════════════════
# Basic construction and build
# ═══════════════════════════════════════════════════════════════════════


class TestProgramBasic:
    def test_empty_program_builds(self):
        prog = Program()
        vert, frag = prog.build()
        assert "#version 430" in vert
        assert "#version 430" in frag

    def test_custom_version(self):
        prog = Program(version="450")
        vert, frag = prog.build()
        assert "#version 450" in vert

    def test_repr(self):
        prog = Program()
        assert "Program" in repr(prog)
        assert "430" in repr(prog)


# ═══════════════════════════════════════════════════════════════════════
# Resource declarations
# ═══════════════════════════════════════════════════════════════════════


class TestResourceDeclarations:
    def test_uniform_returns_variable(self):
        prog = Program()
        u_mvp = prog.uniform("u_mvp", Mat4)
        assert u_mvp.name == "u_mvp"
        assert u_mvp.glsl_type is Mat4

    def test_uniform_appears_in_output(self):
        prog = Program()
        prog.uniform("u_time", Float)
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, frag = prog.build()
        assert "uniform float u_time;" in vert
        assert "uniform float u_time;" in frag

    def test_uniform_vertex_only(self):
        prog = Program()
        prog.uniform("u_mvp", Mat4, stage="vertex")
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")
        vert, frag = prog.build()
        assert "uniform mat4 u_mvp;" in vert
        assert "u_mvp" not in frag

    def test_uniform_string_type(self):
        prog = Program()
        v = prog.uniform("u_tex", "sampler2D", stage="fragment")
        assert v.name == "u_tex"
        # String types return Variable with None glsl_type
        vert, frag = prog.build()
        assert "uniform sampler2D u_tex;" in frag

    def test_varying_returns_variable(self):
        prog = Program()
        v_pos = prog.varying("v_pos", Vec3)
        assert v_pos.name == "v_pos"
        assert v_pos.glsl_type is Vec3

    def test_varying_in_output(self):
        prog = Program()
        prog.varying("v_normal", Vec3)
        vert, frag = prog.build()
        assert "out vec3 v_normal;" in vert
        assert "in vec3 v_normal;" in frag

    def test_flat_varying(self):
        prog = Program()
        prog.varying("v_id", UInt, flat=True)
        vert, frag = prog.build()
        assert "flat out uint v_id;" in vert
        assert "flat in uint v_id;" in frag

    def test_output_returns_variable(self):
        prog = Program()
        f_color = prog.output(0, "f_color", Vec4)
        assert f_color.name == "f_color"
        assert f_color.glsl_type is Vec4

    def test_output_in_frag(self):
        prog = Program()
        prog.output(0, "f_color", Vec4)
        _, frag = prog.build()
        assert "layout(location = 0) out vec4 f_color;" in frag

    def test_storage_declaration(self):
        prog = Program()
        prog.storage(0, "Pos", "float pos_data[];", stage="vertex")
        vert, _ = prog.build()
        assert "buffer Pos" in vert
        assert "pos_data" in vert

    def test_storage_readwrite(self):
        prog = Program()
        prog.storage(0, "Buf", "float data[];", readonly=False, stage="compute")
        prog.local_size(64)
        prog.compute(lambda: "void main() { }")
        src = prog.build_compute()
        assert "buffer Buf" in src
        assert "readonly" not in src

    def test_define(self):
        prog = Program()
        prog.define("HAS_NORMALS")
        vert, _ = prog.build()
        assert "#define HAS_NORMALS" in vert

    def test_define_with_value(self):
        prog = Program()
        prog.define("MAX_LIGHTS", 8, stage="fragment")
        _, frag = prog.build()
        assert "#define MAX_LIGHTS 8" in frag

    def test_struct(self):
        prog = Program()
        st = StructType("Material", {"diffuse": Vec3, "alpha": Float})
        prog.struct(st, stage="fragment")
        _, frag = prog.build()
        assert "struct Material" in frag
        assert "vec3 diffuse;" in frag

    def test_chaining(self):
        prog = Program()
        result = (
            prog.define("A")
            .define("B")
            .storage(0, "Buf", "float d[];", stage="compute")
            .local_size(64)
        )
        assert result is prog


# ═══════════════════════════════════════════════════════════════════════
# Stage decorators
# ═══════════════════════════════════════════════════════════════════════


class TestStageDecorators:
    def test_vertex_string(self):
        prog = Program()

        @prog.vertex
        def vs():
            return """
            void main() {
                gl_Position = vec4(0.0);
            }
            """

        vert, _ = prog.build()
        assert "gl_Position = vec4(0.0);" in vert

    def test_fragment_string(self):
        prog = Program()

        @prog.fragment
        def fs():
            return """
            void main() {
                f_color = vec4(1.0);
            }
            """

        _, frag = prog.build()
        assert "f_color = vec4(1.0);" in frag

    def test_compute_string(self):
        prog = Program()
        prog.local_size(64)

        @prog.compute
        def cs():
            return """
            void main() {
                uint idx = gl_GlobalInvocationID.x;
            }
            """

        src = prog.build_compute()
        assert "gl_GlobalInvocationID" in src

    def test_decorator_none_return(self):
        """Returning None from decorator is a no-op (manual builder usage)."""
        prog = Program()

        @prog.vertex
        def vs():
            prog.builder.add_vertex_lines(["void main() { }"])
            # returns None implicitly

        vert, _ = prog.build()
        assert "void main()" in vert

    def test_decorator_returns_function(self):
        """The decorator returns the original function."""
        prog = Program()

        @prog.vertex
        def my_vs():
            return "void main() { }"

        assert callable(my_vs)

    def test_lambda_stage(self):
        """Stages can be set via direct call with lambdas."""
        prog = Program()
        prog.vertex(lambda: "void main() { gl_Position = vec4(0); }")
        prog.fragment(lambda: "void main() { }")
        vert, frag = prog.build()
        assert "gl_Position" in vert

    def test_invalid_return_type_raises(self):
        prog = Program()
        with pytest.raises(TypeError, match="expected str"):

            @prog.vertex
            def bad():
                return 42


# ═══════════════════════════════════════════════════════════════════════
# GlslFunction integration
# ═══════════════════════════════════════════════════════════════════════


@shader_function
def _luminance(c: Vec3) -> Float:
    return dot(c, vec3(0.299, 0.587, 0.114))


@shader_function
def _saturate(x: Float) -> Float:
    from shadekit.ast._expressions import Literal
    from shadekit.glsl._builtins import clamp
    from shadekit.types._scalars import Float as F

    return clamp(x, Literal(0.0, F), Literal(1.0, F))


class TestGlslFunctionInclusion:
    def test_include_function(self):
        prog = Program()
        prog.include(_luminance, stage="fragment")
        prog.fragment(lambda: "void main() { float l = luminance(vec3(1)); }")
        _, frag = prog.build()
        assert "float _luminance(vec3 c)" in frag
        assert "dot(" in frag

    def test_include_multiple(self):
        prog = Program()
        prog.include(_luminance, stage="fragment")
        prog.include(_saturate, stage="fragment")
        prog.fragment(lambda: "void main() { }")
        _, frag = prog.build()
        assert "_luminance" in frag
        assert "_saturate" in frag

    def test_include_in_compute(self):
        prog = Program()
        prog.local_size(64)
        prog.include(_luminance, stage="compute")
        prog.compute(lambda: "void main() { }")
        src = prog.build_compute()
        assert "_luminance" in src


# ═══════════════════════════════════════════════════════════════════════
# from_glsl factories
# ═══════════════════════════════════════════════════════════════════════


class TestFromGlsl:
    def test_from_glsl_preserves_code(self):
        vert = "#version 430\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430\nvoid main() { }"
        prog = Program.from_glsl(vert, frag)
        v, f = prog.build()
        assert "gl_Position" in v

    def test_from_glsl_inject_define(self):
        vert = "#version 430\nvoid main() { }"
        frag = "#version 430\nvoid main() { }"
        prog = Program.from_glsl(vert, frag)
        prog.define("EXTRA_FEATURE")
        v, _ = prog.build()
        assert "#define EXTRA_FEATURE" in v

    def test_from_compute_glsl(self):
        src = "#version 430\nlayout(local_size_x=64) in;\nvoid main() { }"
        prog = Program.from_compute_glsl(src)
        prog.define("EXTRA", stage="compute")
        out = prog.build_compute()
        assert "#define EXTRA" in out
        assert "local_size_x = 64" in out


# ═══════════════════════════════════════════════════════════════════════
# Clone
# ═══════════════════════════════════════════════════════════════════════


class TestProgramClone:
    def test_clone_produces_equal_output(self):
        prog = Program()
        prog.uniform("u_time", Float)
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")

        clone = prog.clone()
        assert prog.build() == clone.build()

    def test_clone_independence(self):
        prog = Program()
        prog.vertex(lambda: "void main() { }")
        prog.fragment(lambda: "void main() { }")

        clone = prog.clone()
        clone.define("ONLY_IN_CLONE")

        v_orig, _ = prog.build()
        v_clone, _ = clone.build()
        assert "ONLY_IN_CLONE" not in v_orig
        assert "ONLY_IN_CLONE" in v_clone


# ═══════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════


class TestErrors:
    def test_invalid_stage_string(self):
        prog = Program()
        with pytest.raises(ValueError, match="Unknown stage"):
            prog.define("X", stage="geometry")

    def test_invalid_type_raises(self):
        prog = Program()
        with pytest.raises(TypeError, match="Expected ShaderMeta"):
            prog.uniform("u_bad", 42)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# End-to-end: realistic shader
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def test_complete_raster_program(self):
        """Build a realistic mesh shader with the high-level API using Block."""
        prog = Program()

        # Resources — capture Variable nodes for use in Block
        u_mvp = prog.uniform("u_mvp", Mat4, stage="vertex")
        u_mv = prog.uniform("u_mv", Mat4, stage="vertex")
        u_albedo = prog.uniform("u_albedo", Vec3, stage="fragment")
        prog.storage(0, "Pos", "float pos_data[];", stage="vertex")
        prog.storage(1, "Nrm", "float nrm_data[];", stage="vertex")
        prog.storage(3, "Idx", "uint idx_data[];", stage="vertex")
        v_normal = prog.varying("v_normal", Vec3)
        f_color = prog.output(0, "f_color", Vec4)

        # Manual Variables for SSBO members and built-in globals
        pos_data = Variable("pos_data", Float)
        nrm_data = Variable("nrm_data", Float)
        idx_data = Variable("idx_data", UInt)
        gl_VertexID = Variable("gl_VertexID", Int)
        gl_Position = Variable("gl_Position", Vec4)

        @prog.vertex
        def vs():
            b = Block()
            vid = b.var("vid", UInt, idx_data[gl_VertexID])
            pos = b.var(
                "pos",
                Vec3,
                vec3(pos_data[vid * 3], pos_data[vid * 3 + 1], pos_data[vid * 3 + 2]),  # type: ignore[operator]
            )
            nrm = b.var(
                "nrm",
                Vec3,
                vec3(nrm_data[vid * 3], nrm_data[vid * 3 + 1], nrm_data[vid * 3 + 2]),  # type: ignore[operator]
            )
            b.set(v_normal, mat3(u_mv) * nrm)
            b.set(gl_Position, u_mvp * vec4(pos, 1.0))
            return b

        @prog.fragment
        def fs():
            b = Block()
            b.set(f_color, vec4(u_albedo, 1.0))
            return b

        vert, frag = prog.build()

        # Verify structure
        assert "#version 430" in vert
        assert "uniform mat4 u_mvp;" in vert
        assert "uniform mat4 u_mv;" in vert
        assert "buffer Pos" in vert
        assert "out vec3 v_normal;" in vert
        assert "void main()" in vert

        # Verify AST-generated code is in the vertex shader
        assert "uint vid = idx_data[gl_VertexID];" in vert
        assert "v_normal = mat3(u_mv) * nrm;" in vert
        assert "gl_Position = u_mvp * vec4(pos, 1.0);" in vert

        assert "#version 430" in frag
        assert "uniform vec3 u_albedo;" in frag
        assert "in vec3 v_normal;" in frag
        assert "layout(location = 0) out vec4 f_color;" in frag
        assert "void main()" in frag
        assert "f_color = vec4(u_albedo, 1.0);" in frag

        # Vertex-only resources not in fragment
        assert "u_mvp" not in frag
        assert "Pos" not in frag

    def test_complete_raster_program_raw_glsl(self):
        """Original raw-GLSL version still works."""
        prog = Program()
        prog.uniform("u_mvp", Mat4, stage="vertex")
        prog.uniform("u_mv", Mat4, stage="vertex")
        prog.uniform("u_albedo", Vec3, stage="fragment")
        prog.storage(0, "Pos", "float pos_data[];", stage="vertex")
        prog.storage(1, "Nrm", "float nrm_data[];", stage="vertex")
        prog.storage(3, "Idx", "uint idx_data[];", stage="vertex")
        prog.varying("v_normal", Vec3)
        prog.output(0, "f_color", Vec4)

        @prog.vertex
        def vs():
            return """
            void main() {
                uint vid = idx_data[gl_VertexID];
                vec3 pos = vec3(pos_data[vid*3u], pos_data[vid*3u+1u], pos_data[vid*3u+2u]);
                vec3 nrm = vec3(nrm_data[vid*3u], nrm_data[vid*3u+1u], nrm_data[vid*3u+2u]);
                v_normal = mat3(u_mv) * nrm;
                gl_Position = u_mvp * vec4(pos, 1.0);
            }
            """

        @prog.fragment
        def fs():
            return """
            void main() {
                f_color = vec4(u_albedo, 1.0);
            }
            """

        vert, frag = prog.build()
        assert "void main()" in vert
        assert "void main()" in frag

    def test_complete_compute_program(self):
        """Build a realistic compute shader with the high-level API using Block."""
        prog = Program()
        prog.local_size(256)
        u_count = prog.uniform("u_count", UInt, stage="compute")
        prog.storage(0, "Input", "float in_data[];", stage="compute")
        prog.storage(1, "Output", "float out_data[];", readonly=False, stage="compute")

        # Manual Variables for SSBO members and built-in globals
        in_data = Variable("in_data", Float)
        out_data = Variable("out_data", Float)
        gl_GlobalInvocationID = Variable("gl_GlobalInvocationID", UInt)

        @prog.compute
        def cs():
            b = Block()
            idx = b.var("idx", UInt, gl_GlobalInvocationID.x)
            b += If(idx >= u_count, [Return()])
            b.set(out_data[idx], in_data[idx] * 2.0)  # type: ignore[operator]
            return b

        src = prog.build_compute()
        assert "#version 430" in src
        assert "local_size_x = 256" in src
        assert "uniform uint u_count;" in src
        assert "buffer Input" in src
        assert "buffer Output" in src
        assert "gl_GlobalInvocationID" in src
        assert "uint idx = gl_GlobalInvocationID.x;" in src
        assert "out_data[idx] = in_data[idx] * 2.0;" in src


# ═══════════════════════════════════════════════════════════════════════
# Block statement collector
# ═══════════════════════════════════════════════════════════════════════


class TestBlock:
    """Tests for the Block statement collector."""

    def test_var_returns_variable_and_appends_declaration(self):
        b = Block()
        x = b.var("x", Float, 1.0)
        assert isinstance(x, Variable)
        assert x.name == "x"
        assert len(b) == 1
        assert isinstance(b.stmts[0], Declaration)
        assert b.stmts[0].name == "x"

    def test_const_declaration(self):
        b = Block()
        c = b.const("PI", Float, 3.14159)
        assert isinstance(c, Variable)
        assert c.name == "PI"
        assert b.stmts[0].const is True  # type: ignore[union-attr]

    def test_set_appends_assignment(self):
        b = Block()
        x = b.var("x", Float)
        b.set(x, 2.0)
        assert len(b) == 2
        assert isinstance(b.stmts[1], Assignment)

    def test_return_with_value(self):
        b = Block()
        x = b.var("x", Float, 1.0)
        b.return_(x)
        assert len(b) == 2
        assert isinstance(b.stmts[1], Return)
        assert b.stmts[1].value is not None

    def test_return_void(self):
        b = Block()
        b.return_()
        assert isinstance(b.stmts[0], Return)
        assert b.stmts[0].value is None

    def test_iadd_raw_stmt(self):
        b = Block()
        x = Variable("x", Float)
        b += Assignment(x, Variable("y", Float))
        assert len(b) == 1
        assert isinstance(b.stmts[0], Assignment)

    def test_iterable(self):
        b = Block()
        b.var("a", Float, 0.0)
        b.var("b", Float, 1.0)
        stmts = list(b)
        assert len(stmts) == 2
        assert all(isinstance(s, Declaration) for s in stmts)

    def test_block_in_vertex_decorator(self):
        """Block works with @prog.vertex."""
        prog = Program()
        gl_Position = Variable("gl_Position", Vec4)

        @prog.vertex
        def vs():
            b = Block()
            b.set(gl_Position, vec4(0.0, 0.0, 0.0, 1.0))
            return b

        vert, _frag = prog.build()
        assert "gl_Position = vec4(0.0, 0.0, 0.0, 1.0);" in vert

    def test_block_in_fragment_decorator(self):
        """Block works with @prog.fragment."""
        prog = Program()
        f_color = prog.output(0, "f_color", Vec4)

        @prog.fragment
        def fs():
            b = Block()
            b.set(f_color, vec4(1.0, 0.0, 0.0, 1.0))
            return b

        _vert, frag = prog.build()
        assert "f_color = vec4(1.0, 0.0, 0.0, 1.0);" in frag

    def test_block_repr(self):
        b = Block()
        b.var("x", Float)
        b.var("y", Float)
        assert repr(b) == "Block(2 stmts)"


# ═══════════════════════════════════════════════════════════════════════
# Auto-inclusion of @shader_function in @prog.* decorators
# ═══════════════════════════════════════════════════════════════════════


class TestAutoInclusion:
    """@shader_function calls in Block/list are auto-included in the shader."""

    def test_glsl_function_auto_included_in_fragment(self):
        """A @shader_function used in @prog.fragment is emitted without prog.include()."""
        from shadekit.decorators import shader_function
        from shadekit.glsl._builtins import dot

        @shader_function
        def luminance(c: Vec3) -> Float:
            return dot(c, vec3(0.299, 0.587, 0.114))

        prog = Program()
        u_albedo = prog.uniform("u_albedo", Vec3, stage="fragment")
        f_color = prog.output(0, "f_color", Vec4)

        @prog.fragment
        def fs():
            b = Block()
            lum = b.var("lum", Float, luminance(u_albedo))
            b.set(f_color, vec4(lum, lum, lum, 1.0))
            return b

        _vert, frag = prog.build()
        # Function definition should be present
        assert "float luminance(vec3 c)" in frag
        # main() should call it
        assert "float lum = luminance(u_albedo);" in frag

    def test_glsl_function_auto_included_in_vertex(self):
        """A @shader_function used in @prog.vertex Block is auto-included."""
        from shadekit.decorators import shader_function

        @shader_function
        def custom_transform(p: Vec3) -> Vec4:
            return vec4(p, 1.0)

        prog = Program()
        gl_Position = Variable("gl_Position", Vec4)

        @prog.vertex
        def vs():
            b = Block()
            pos = b.var("pos", Vec3, vec3(0.0, 0.0, 0.0))
            b.set(gl_Position, custom_transform(pos))
            return b

        vert, _frag = prog.build()
        assert "vec4 custom_transform(vec3 p)" in vert
        assert "gl_Position = custom_transform(pos);" in vert

    def test_glsl_function_auto_included_in_compute(self):
        """A @shader_function used in @prog.compute Block is auto-included."""
        from shadekit.decorators import shader_function

        @shader_function
        def double_it(x: Float) -> Float:
            return x * 2.0  # type: ignore[operator]

        prog = Program()
        prog.local_size(64)

        @prog.compute
        def cs():
            b = Block()
            _val = b.var("val", Float, double_it(Variable("in_data", Float)))
            return b

        src = prog.build_compute()
        assert "float double_it(float x)" in src
        assert "float val = double_it(in_data);" in src

    def test_transitive_dependency_auto_included(self):
        """If func A calls func B, both are auto-included."""
        from shadekit.decorators import shader_function
        from shadekit.glsl._builtins import dot

        @shader_function
        def luminance(c: Vec3) -> Float:
            return dot(c, vec3(0.299, 0.587, 0.114))

        @shader_function
        def is_bright(c: Vec3) -> Float:
            return luminance(c)

        prog = Program()
        u_color = prog.uniform("u_color", Vec3, stage="fragment")
        f_out = prog.output(0, "f_out", Vec4)

        @prog.fragment
        def fs():
            b = Block()
            brightness = b.var("brightness", Float, is_bright(u_color))
            b.set(f_out, vec4(brightness, brightness, brightness, 1.0))
            return b

        _vert, frag = prog.build()
        # Both functions emitted
        assert "float luminance(vec3 c)" in frag
        assert "float is_bright(vec3 c)" in frag
        # luminance emitted before is_bright (dependency order)
        lum_pos = frag.index("float luminance(vec3 c)")
        bright_pos = frag.index("float is_bright(vec3 c)")
        assert lum_pos < bright_pos

    def test_auto_include_with_list_return(self):
        """Auto-inclusion works when returning list[Stmt] (not just Block)."""
        from shadekit.decorators import shader_function

        @shader_function
        def side_effect(x: Float) -> None:
            return None  # void function

        prog = Program()

        @prog.fragment
        def fs():
            call = side_effect(Variable("val", Float))
            return [ExpressionStatement(call)]

        _vert, frag = prog.build()
        assert "void side_effect(float x)" in frag

    def test_auto_include_with_single_stmt(self):
        """Auto-inclusion works when returning a single Stmt."""
        from shadekit.decorators import shader_function

        @shader_function
        def my_scale(x: Float) -> Float:
            return x * 3.0  # type: ignore[operator]

        prog = Program()

        @prog.fragment
        def fs():
            return Declaration("v", Float, my_scale(Variable("u", Float)))

        _vert, frag = prog.build()
        assert "float my_scale(float x)" in frag
        assert "float v = my_scale(u);" in frag

    def test_no_duplicate_inclusion(self):
        """Same function used multiple times is only emitted once."""
        from shadekit.decorators import shader_function
        from shadekit.glsl._builtins import dot

        @shader_function
        def luminance(c: Vec3) -> Float:
            return dot(c, vec3(0.299, 0.587, 0.114))

        prog = Program()
        u_a = prog.uniform("u_a", Vec3, stage="fragment")
        u_b = prog.uniform("u_b", Vec3, stage="fragment")
        f_out = prog.output(0, "f_out", Vec4)

        @prog.fragment
        def fs():
            b = Block()
            la = b.var("la", Float, luminance(u_a))
            lb = b.var("lb", Float, luminance(u_b))
            b.set(f_out, vec4(la, lb, 0.0, 1.0))
            return b

        _vert, frag = prog.build()
        # Count occurrences of the function signature
        count = frag.count("float luminance(vec3 c)")
        assert count == 1


# ═══════════════════════════════════════════════════════════════════════
# Stage — str subclass for mix-and-match
# ═══════════════════════════════════════════════════════════════════════


class TestStage:
    """Stage is a str subclass carrying stage metadata."""

    def test_build_returns_stages(self):
        prog = Program()
        vert, frag = prog.build()
        from shadekit.glsl import Stage

        assert isinstance(vert, Stage)
        assert isinstance(frag, Stage)
        assert isinstance(vert, str)
        assert isinstance(frag, str)

    def test_stage_kind(self):
        prog = Program()
        vert, frag = prog.build()
        assert vert.kind == "vertex"
        assert frag.kind == "fragment"

    def test_compute_stage_kind(self):
        prog = Program()
        prog.local_size(64)

        @prog.compute
        def cs():
            return "// noop"

        stage = prog.build_compute()
        from shadekit.glsl import Stage

        assert isinstance(stage, Stage)
        assert stage.kind == "compute"

    def test_stage_is_string(self):
        """Stage works in all string contexts."""
        prog = Program()
        vert, frag = prog.build()
        assert "#version 430" in vert
        assert "#version 430" in frag
        assert vert.startswith("#version")

    def test_stage_repr(self):
        prog = Program()
        vert, _frag = prog.build()
        r = repr(vert)
        assert "vertex" in r
        assert "lines=" in r

    def test_mix_and_match(self):
        """Stages from different Programs can be freely combined."""
        prog_a = Program()

        @prog_a.vertex
        def vs_a():
            return "// vertex A"

        @prog_a.fragment
        def fs_a():
            return "// fragment A"

        prog_b = Program()

        @prog_b.vertex
        def vs_b():
            return "// vertex B"

        @prog_b.fragment
        def fs_b():
            return "// fragment B"

        va, fa = prog_a.build()
        vb, fb = prog_b.build()

        # Cross-combine: vertex A + fragment B
        assert "vertex A" in va
        assert "fragment B" in fb
        # Both are just strings — can be passed to any GL compiler
        combined = (va, fb)
        assert len(combined) == 2

    def test_stage_save(self, tmp_path):
        prog = Program()

        @prog.vertex
        def vs():
            return "// test vertex"

        vert, _frag = prog.build()
        out = tmp_path / "test.vert.glsl"
        vert.save(out)
        assert out.read_text(encoding="utf-8") == str(vert)

    def test_program_save_raster(self, tmp_path):
        prog = Program()
        prog.save(tmp_path / "shader.glsl")
        assert (tmp_path / "shader.vert.glsl").exists()
        assert (tmp_path / "shader.frag.glsl").exists()

    def test_program_save_compute(self, tmp_path):
        prog = Program()
        prog.local_size(128)

        @prog.compute
        def cs():
            return "// compute"

        prog.save(tmp_path / "compute.glsl")
        assert (tmp_path / "compute.glsl").exists()

    def test_program_str_raster(self):
        prog = Program()
        s = str(prog)
        assert "// --- fragment ---" in s
        assert "#version 430" in s

    def test_program_str_compute(self):
        prog = Program()
        prog.local_size(64)

        @prog.compute
        def cs():
            return "// work"

        s = str(prog)
        assert "local_size" in s
        assert "// --- fragment ---" not in s

    def test_stage_hash_and_equality(self):
        """Stage hashing/equality uses string content (inherited from str)."""
        prog = Program()
        prog.uniform("u_time", Float, stage="fragment")
        v1, f1 = prog.build()
        v2, f2 = prog.build()
        assert v1 == v2
        assert hash(v1) == hash(v2)
        # Fragment has a uniform that vertex doesn't → different content
        assert v1 != f1
