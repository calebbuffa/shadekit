"""Tests for transitive dependency resolution.

Ensures that collect_transitive_deps (core) and Program.include (GLSL)
always discover and emit all transitively-referenced @shader_function
instances in correct dependency order.
"""

from __future__ import annotations

from shadekit.compiler import collect_transitive_deps
from shadekit.decorators import shader_function
from shadekit.glsl import Program, clamp, dot, mix, vec3
from shadekit.types import Float, Vec3

# ── Fixtures ─────────────────────────────────────────────────────────


@shader_function
def luminance(c: Vec3) -> Float:
    return dot(c, vec3(0.299, 0.587, 0.114))


@shader_function
def desaturate(color: Vec3, amount: Float) -> Vec3:
    grey = luminance(color)
    return mix(color, vec3(grey, grey, grey), amount)


@shader_function
def tone_map(color: Vec3) -> Vec3:
    return clamp(desaturate(color, luminance(color)), 0.0, 1.0)


@shader_function
def leaf(x: Float) -> Float:
    return x * x


@shader_function
def mid_a(x: Float) -> Float:
    return leaf(x) + x


@shader_function
def mid_b(x: Float) -> Float:
    return leaf(x) * x


@shader_function
def diamond_top(x: Float) -> Float:
    return mid_a(x) + mid_b(x)


@shader_function
def standalone(x: Float) -> Float:
    return x + x


# ═════════════════════════════════════════════════════════════════════
# collect_transitive_deps (core utility)
# ═════════════════════════════════════════════════════════════════════


class TestCollectTransitiveDeps:
    """Tests for the core collect_transitive_deps function."""

    def test_single_function_no_deps(self) -> None:
        deps = collect_transitive_deps(standalone)
        assert len(deps) == 1
        assert deps[0] is standalone

    def test_direct_dependency(self) -> None:
        deps = collect_transitive_deps(desaturate)
        names = [fn.name for fn in deps]
        assert "luminance" in names
        assert "desaturate" in names
        assert len(deps) == 2

    def test_dependency_order(self) -> None:
        """Dependencies must precede their dependents."""
        deps = collect_transitive_deps(desaturate)
        names = [fn.name for fn in deps]
        assert names.index("luminance") < names.index("desaturate")

    def test_transitive_chain(self) -> None:
        """tone_map -> desaturate -> luminance: all three resolved."""
        deps = collect_transitive_deps(tone_map)
        names = [fn.name for fn in deps]
        assert "luminance" in names
        assert "desaturate" in names
        assert "tone_map" in names
        assert len(deps) == 3
        # Order: luminance before desaturate before tone_map.
        assert names.index("luminance") < names.index("desaturate")
        assert names.index("desaturate") < names.index("tone_map")

    def test_diamond_dependency(self) -> None:
        """Diamond: top -> mid_a -> leaf, top -> mid_b -> leaf."""
        deps = collect_transitive_deps(diamond_top)
        names = [fn.name for fn in deps]
        assert "leaf" in names
        assert "mid_a" in names
        assert "mid_b" in names
        assert "diamond_top" in names
        # Leaf shared by both mids — only emitted once.
        assert names.count("leaf") == 1
        assert len(deps) == 4
        # leaf before both mids, both mids before top.
        assert names.index("leaf") < names.index("mid_a")
        assert names.index("leaf") < names.index("mid_b")
        assert names.index("mid_a") < names.index("diamond_top")
        assert names.index("mid_b") < names.index("diamond_top")

    def test_no_duplicates(self) -> None:
        """Even when a function is referenced multiple times, it appears once."""
        deps = collect_transitive_deps(tone_map)
        names = [fn.name for fn in deps]
        # luminance is referenced both directly by tone_map and via desaturate.
        assert names.count("luminance") == 1


# ═════════════════════════════════════════════════════════════════════
# Program.include() — GLSL backend integration
# ═════════════════════════════════════════════════════════════════════


class TestProgramIncludeTransitiveDeps:
    """Program.include() must resolve transitive deps and emit correct GLSL."""

    def test_include_single_no_deps(self) -> None:
        prog = Program()
        prog.include(standalone, stage="fragment")
        _, frag = prog.build()
        assert "float standalone(float x)" in frag

    def test_include_emits_transitive_dep(self) -> None:
        """Including desaturate must also emit luminance."""
        prog = Program()
        prog.include(desaturate, stage="fragment")
        _, frag = prog.build()
        assert "float luminance(vec3 c)" in frag
        assert "vec3 desaturate(vec3 color, float amount)" in frag

    def test_include_transitive_chain_order(self) -> None:
        """luminance must appear before desaturate in emitted source."""
        prog = Program()
        prog.include(desaturate, stage="fragment")
        _, frag = prog.build()
        lum_idx = frag.index("float luminance(")
        desat_idx = frag.index("vec3 desaturate(")
        assert lum_idx < desat_idx

    def test_include_deep_chain(self) -> None:
        """tone_map -> desaturate -> luminance: all three in correct order."""
        prog = Program()
        prog.include(tone_map, stage="fragment")
        _, frag = prog.build()
        assert "float luminance(" in frag
        assert "vec3 desaturate(" in frag
        assert "vec3 tone_map(" in frag
        lum_idx = frag.index("float luminance(")
        desat_idx = frag.index("vec3 desaturate(")
        tone_idx = frag.index("vec3 tone_map(")
        assert lum_idx < desat_idx < tone_idx

    def test_include_diamond(self) -> None:
        """Diamond deps: leaf emitted once, before both mids, before top."""
        prog = Program()
        prog.include(diamond_top, stage="fragment")
        _, frag = prog.build()
        assert frag.count("float leaf(float x)") == 1
        assert "float mid_a(" in frag
        assert "float mid_b(" in frag
        assert "float diamond_top(" in frag
        leaf_idx = frag.index("float leaf(")
        top_idx = frag.index("float diamond_top(")
        assert leaf_idx < top_idx

    def test_include_multiple_functions_deduplicates(self) -> None:
        """Including two functions that share a dep doesn't duplicate it."""
        prog = Program()
        prog.include(mid_a, stage="fragment")
        prog.include(mid_b, stage="fragment")
        _, frag = prog.build()
        # leaf should appear exactly once even though both mid_a and mid_b
        # depend on it.
        assert frag.count("float leaf(float x)") == 1

    def test_include_in_vertex_stage(self) -> None:
        """Transitive deps also work for vertex stage."""
        prog = Program()
        prog.include(desaturate, stage="vertex")
        vert, _ = prog.build()
        assert "float luminance(" in vert
        assert "vec3 desaturate(" in vert

    def test_include_preserves_function_body(self) -> None:
        """Emitted GLSL should contain correct function body code."""
        prog = Program()
        prog.include(luminance, stage="fragment")
        _, frag = prog.build()
        # luminance body should contain a dot() call
        assert "dot(c, vec3(" in frag


# ═════════════════════════════════════════════════════════════════════
# Stage decorator auto-include transitive deps
# ═════════════════════════════════════════════════════════════════════


class TestStageDecoratorAutoInclude:
    """@prog.fragment auto-includes referenced shader_functions transitively."""

    def test_auto_include_via_ast_stmts(self) -> None:
        from shadekit.ast import Return, Variable

        prog = Program()
        c_var = Variable("c", Vec3)

        @prog.fragment
        def fs():
            return Return(desaturate(c_var, luminance(c_var)))

        _, frag = prog.build()
        assert "float luminance(" in frag
        assert "vec3 desaturate(" in frag
