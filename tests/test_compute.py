"""Tests for shadekit compute shader support.

Covers:
- ShaderStage.COMPUTE enum value
- Compute-stage storage routing in ShaderBuilder
- build_compute() assembly with local_size layout
- from_compute_glsl() factory
- Compute builtins (barriers, atomics, image ops)
- Compute validation (local_size, graphics-only builtins)
- ShaderCache compute path
"""

from __future__ import annotations

from shadekit.compiler import ShaderCache, hash_sources

# Module-level @shader_function for test_compute_with_glsl_function
# (get_type_hints requires types to be resolvable in the defining scope).
from shadekit.decorators import shader_function
from shadekit.glsl import (
    ShaderBuilder,
    ShaderStage,
    validate_compute_builder,
    validate_source,
)
from shadekit.glsl._builtins import sqrt as _sqrt
from shadekit.glsl._validation import (
    Severity,
    _check_compute_builtins,
    _check_compute_local_size,
)
from shadekit.types import Float as _Float


@shader_function
def _magnitude(x: _Float, y: _Float) -> _Float:
    return _sqrt(x * x + y * y)


# ═══════════════════════════════════════════════════════════════════════
# ShaderStage.COMPUTE
# ═══════════════════════════════════════════════════════════════════════


class TestShaderStageCompute:
    def test_compute_value(self):
        assert ShaderStage.COMPUTE == 4

    def test_compute_not_in_both(self):
        assert ShaderStage.COMPUTE not in ShaderStage.BOTH

    def test_compute_or_vertex(self):
        combined = ShaderStage.COMPUTE | ShaderStage.VERTEX
        assert ShaderStage.COMPUTE in combined
        assert ShaderStage.VERTEX in combined
        assert ShaderStage.FRAGMENT not in combined


# ═══════════════════════════════════════════════════════════════════════
# Compute-stage builder routing
# ═══════════════════════════════════════════════════════════════════════


class TestComputeBuilderRouting:
    def test_add_define_routes_to_compute(self):
        b = ShaderBuilder()
        b.add_define("WORK", stage=ShaderStage.COMPUTE)
        assert any("WORK" in d for d in b._compute_defines)
        assert not any("WORK" in d for d in b._vert_defines)
        assert not any("WORK" in d for d in b._frag_defines)

    def test_add_uniform_routes_to_compute(self):
        b = ShaderBuilder()
        b.add_uniform("float", "u_time", stage=ShaderStage.COMPUTE)
        assert any("u_time" in u for u in b._compute_uniforms)
        assert not any("u_time" in u for u in b._vert_uniforms)

    def test_add_ssbo_routes_to_compute(self):
        b = ShaderBuilder()
        b.add_ssbo(
            0, "Data", "float data[];", stage=ShaderStage.COMPUTE, readonly=False
        )
        assert any("Data" in s for s in b._compute_ssbos)
        assert not any("Data" in s for s in b._vert_ssbos)

    def test_add_struct_routes_to_compute(self):
        b = ShaderBuilder()
        b.add_struct("particle", "Particle", stage=ShaderStage.COMPUTE)
        b.add_struct_field("particle", "vec3", "position")
        assert "particle" in b._compute_struct_ids
        assert "particle" not in b._vert_struct_ids

    def test_add_function_routes_to_compute(self):
        b = ShaderBuilder()
        b.add_function("helper", "float helper(float x)", stage=ShaderStage.COMPUTE)
        b.add_function_lines("helper", "return x * 2.0;")
        assert "helper" in b._compute_function_ids
        assert "helper" not in b._vert_function_ids

    def test_add_compute_lines(self):
        b = ShaderBuilder()
        b.add_compute_lines("void main() {")
        b.add_compute_lines(["  // work", "}"])
        assert len(b._compute_lines) == 3

    def test_add_local_size(self):
        b = ShaderBuilder()
        b.add_local_size(256, 1, 1)
        assert b._compute_local_size == (256, 1, 1)

    def test_add_local_size_defaults(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        assert b._compute_local_size == (64, 1, 1)


# ═══════════════════════════════════════════════════════════════════════
# build_compute()
# ═══════════════════════════════════════════════════════════════════════


class TestBuildCompute:
    def test_basic_compute_shader(self):
        b = ShaderBuilder()
        b.add_local_size(256)
        b.add_ssbo(
            0, "Data", "float data[];", stage=ShaderStage.COMPUTE, readonly=False
        )
        b.add_compute_lines(
            [
                "void main() {",
                "  uint idx = gl_GlobalInvocationID.x;",
                "  data[idx] *= 2.0;",
                "}",
            ]
        )
        src = b.build_compute()

        assert "#version 430" in src
        assert "local_size_x = 256" in src
        assert "local_size_y = 1" in src
        assert "buffer Data" in src
        assert "gl_GlobalInvocationID" in src

    def test_compute_with_defines_and_uniforms(self):
        b = ShaderBuilder()
        b.add_local_size(64, 4)
        b.add_define("BLOCK_SIZE", 256, stage=ShaderStage.COMPUTE)
        b.add_uniform("float", "u_dt", stage=ShaderStage.COMPUTE)
        b.add_compute_lines(["void main() { }"])
        src = b.build_compute()

        assert "#define BLOCK_SIZE 256" in src
        assert "uniform float u_dt;" in src
        assert "local_size_x = 64" in src
        assert "local_size_y = 4" in src

    def test_compute_isolation_from_vertex_fragment(self):
        """Compute storage is independent — vertex/fragment data should NOT leak."""
        b = ShaderBuilder()
        b.add_define("VERT_ONLY", stage=ShaderStage.VERTEX)
        b.add_define("FRAG_ONLY", stage=ShaderStage.FRAGMENT)
        b.add_define("COMP_ONLY", stage=ShaderStage.COMPUTE)
        b.add_local_size(128)
        b.add_compute_lines(["void main() { }"])
        src = b.build_compute()

        assert "COMP_ONLY" in src
        assert "VERT_ONLY" not in src
        assert "FRAG_ONLY" not in src

    def test_compute_with_structs(self):
        b = ShaderBuilder()
        b.add_local_size(32)
        b.add_struct("particle", "Particle", stage=ShaderStage.COMPUTE)
        b.add_struct_field("particle", "vec4", "pos")
        b.add_struct_field("particle", "vec4", "vel")
        b.add_compute_lines(["void main() { }"])
        src = b.build_compute()

        assert "struct Particle {" in src
        assert "vec4 pos;" in src
        assert "vec4 vel;" in src

    def test_compute_with_functions(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_function("helper", "float helper(float x)", stage=ShaderStage.COMPUTE)
        b.add_function_lines("helper", "return x * 2.0;")
        b.add_compute_lines(["void main() {", "  float v = helper(1.0);", "}"])
        src = b.build_compute()

        assert "float helper(float x) {" in src
        assert "return x * 2.0;" in src

    def test_compute_with_glsl_function(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_glsl_function(_magnitude, stage=ShaderStage.COMPUTE)
        b.add_compute_lines(["void main() { }"])
        src = b.build_compute()

        assert "float _magnitude(float x, float y)" in src

    def test_no_local_size_emits_no_layout(self):
        b = ShaderBuilder()
        b.add_compute_lines(["void main() { }"])
        src = b.build_compute()
        assert "local_size" not in src


# ═══════════════════════════════════════════════════════════════════════
# from_compute_glsl()
# ═══════════════════════════════════════════════════════════════════════


class TestFromComputeGlsl:
    def test_parses_local_size(self):
        src = """\
#version 430
layout(local_size_x = 128, local_size_y = 2, local_size_z = 1) in;

void main() {
  uint idx = gl_GlobalInvocationID.x;
}
"""
        b = ShaderBuilder.from_compute_glsl(src)
        assert b._compute_local_size == (128, 2, 1)
        rebuilt = b.build_compute()
        assert "local_size_x = 128" in rebuilt
        assert "gl_GlobalInvocationID" in rebuilt

    def test_parses_partial_local_size(self):
        src = "#version 430\nlayout(local_size_x = 64) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        assert b._compute_local_size == (64, 1, 1)

    def test_injects_define_after_parse(self):
        src = "#version 430\nlayout(local_size_x = 32) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        b.add_define("EXTRA", stage=ShaderStage.COMPUTE)
        rebuilt = b.build_compute()
        assert "#define EXTRA" in rebuilt
        assert "void main()" in rebuilt


# ═══════════════════════════════════════════════════════════════════════
# Compute builtins
# ═══════════════════════════════════════════════════════════════════════


class TestComputeBuiltins:
    def test_barrier(self):
        from shadekit.glsl._builtins import barrier

        node = barrier()
        assert node.func_name == "barrier"
        assert node.glsl_type is None  # void

    def test_memory_barrier(self):
        from shadekit.glsl._builtins import memoryBarrier

        node = memoryBarrier()
        assert node.func_name == "memoryBarrier"

    def test_group_memory_barrier(self):
        from shadekit.glsl._builtins import groupMemoryBarrier

        node = groupMemoryBarrier()
        assert node.func_name == "groupMemoryBarrier"

    def test_memory_barrier_buffer(self):
        from shadekit.glsl._builtins import memoryBarrierBuffer

        node = memoryBarrierBuffer()
        assert node.func_name == "memoryBarrierBuffer"

    def test_memory_barrier_shared(self):
        from shadekit.glsl._builtins import memoryBarrierShared

        node = memoryBarrierShared()
        assert node.func_name == "memoryBarrierShared"

    def test_memory_barrier_image(self):
        from shadekit.glsl._builtins import memoryBarrierImage

        node = memoryBarrierImage()
        assert node.func_name == "memoryBarrierImage"

    def test_atomic_add(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import atomicAdd
        from shadekit.types import UInt

        mem = Variable("counter", UInt)
        data = Variable("val", UInt)
        node = atomicAdd(mem, data)
        assert node.func_name == "atomicAdd"
        assert node.glsl_type is UInt

    def test_atomic_min(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import atomicMin
        from shadekit.types import UInt

        node = atomicMin(Variable("a", UInt), Variable("b", UInt))
        assert node.func_name == "atomicMin"

    def test_atomic_max(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import atomicMax
        from shadekit.types import UInt

        node = atomicMax(Variable("a", UInt), Variable("b", UInt))
        assert node.func_name == "atomicMax"

    def test_atomic_exchange(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import atomicExchange
        from shadekit.types import UInt

        node = atomicExchange(Variable("m", UInt), Variable("v", UInt))
        assert node.func_name == "atomicExchange"

    def test_atomic_comp_swap(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import atomicCompSwap
        from shadekit.types import UInt

        node = atomicCompSwap(
            Variable("m", UInt), Variable("c", UInt), Variable("d", UInt)
        )
        assert node.func_name == "atomicCompSwap"

    def test_image_load(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import imageLoad
        from shadekit.types import Vec4
        from shadekit.types._samplers import Sampler2D

        img = Variable("img", Sampler2D)
        coord = Variable("c", Vec4)
        node = imageLoad(img, coord)
        assert node.func_name == "imageLoad"
        assert node.glsl_type is Vec4

    def test_image_store(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import imageStore
        from shadekit.types import Vec4
        from shadekit.types._samplers import Sampler2D

        img = Variable("img", Sampler2D)
        coord = Variable("c", Vec4)
        data = Variable("d", Vec4)
        node = imageStore(img, coord, data)
        assert node.func_name == "imageStore"
        assert node.glsl_type is None  # void

    def test_image_atomic_add(self):
        from shadekit.ast import Variable
        from shadekit.glsl._builtins import imageAtomicAdd
        from shadekit.types import UInt
        from shadekit.types._samplers import Sampler2D

        node = imageAtomicAdd(
            Variable("img", Sampler2D), Variable("c", UInt), Variable("d", UInt)
        )
        assert node.func_name == "imageAtomicAdd"
        assert node.glsl_type is UInt


# ═══════════════════════════════════════════════════════════════════════
# Compute validation
# ═══════════════════════════════════════════════════════════════════════


class TestComputeValidation:
    def test_missing_local_size(self):
        errs = _check_compute_local_size("void main() { }")
        assert len(errs) == 1
        assert errs[0].severity == Severity.ERROR
        assert "local_size" in errs[0].message

    def test_has_local_size(self):
        errs = _check_compute_local_size(
            "layout(local_size_x = 64) in;\nvoid main() { }"
        )
        assert len(errs) == 0

    def test_graphics_builtin_gl_Position(self):
        errs = _check_compute_builtins("gl_Position = vec4(0);")
        assert any("gl_Position" in e.message for e in errs)

    def test_graphics_builtin_gl_FragCoord(self):
        errs = _check_compute_builtins("vec4 fc = gl_FragCoord;")
        assert any("gl_FragCoord" in e.message for e in errs)

    def test_graphics_builtin_gl_FragDepth(self):
        errs = _check_compute_builtins("gl_FragDepth = 0.5;")
        assert any("gl_FragDepth" in e.message for e in errs)

    def test_clean_compute(self):
        errs = _check_compute_builtins(
            "uint idx = gl_GlobalInvocationID.x;\nbarrier();"
        )
        assert len(errs) == 0

    def test_validate_compute_builder_clean(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_ssbo(0, "Buf", "float data[];", stage=ShaderStage.COMPUTE, readonly=False)
        b.add_compute_lines(
            [
                "void main() {",
                "  uint idx = gl_GlobalInvocationID.x;",
                "  data[idx] = 0.0;",
                "}",
            ]
        )
        errs = validate_compute_builder(b)
        assert len(errs) == 0

    def test_validate_compute_builder_missing_main(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_compute_lines(["// no main"])
        errs = validate_compute_builder(b)
        assert any("main()" in e.message for e in errs)

    def test_validate_compute_builder_missing_local_size(self):
        b = ShaderBuilder()
        b.add_compute_lines(["void main() { }"])
        errs = validate_compute_builder(b)
        assert any("local_size" in e.message for e in errs)

    def test_validate_compute_builder_graphics_builtin(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_compute_lines(["void main() { gl_Position = vec4(0); }"])
        errs = validate_compute_builder(b)
        assert any("gl_Position" in e.message for e in errs)

    def test_validate_source_compute_stage(self):
        src = "#version 430\nlayout(local_size_x=64) in;\nvoid main() { }"
        errs = validate_source(src, "compute")
        assert len(errs) == 0


# ═══════════════════════════════════════════════════════════════════════
# ShaderCache — compute path
# ═══════════════════════════════════════════════════════════════════════


class TestShaderCacheCompute:
    def test_compute_cache_miss_then_hit(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_compute_lines(["void main() { }"])
        cache = ShaderCache()

        key1, src1 = cache.get_or_build_compute(b)
        key2, src2 = cache.get_or_build_compute(b)
        assert key1 == key2
        assert src1 == src2
        assert cache.contains(key1)

    def test_compute_and_raster_independent(self):
        cache = ShaderCache()

        cb = ShaderBuilder()
        cb.add_local_size(64)
        cb.add_compute_lines(["void main() { }"])
        ckey, _ = cache.get_or_build_compute(cb)

        rb = ShaderBuilder()
        rb.add_vertex_lines(["void main() { gl_Position = vec4(0); }"])
        rb.add_fragment_lines(["void main() { }"])
        rkey, _ = cache.get_or_build(rb)

        assert ckey != rkey
        assert len(cache) == 2

    def test_compute_invalidate(self):
        cache = ShaderCache()
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_compute_lines(["void main() { }"])
        key, _ = cache.get_or_build_compute(b)
        assert cache.contains(key)
        cache.invalidate(key)
        assert not cache.contains(key)

    def test_compute_clear(self):
        cache = ShaderCache()
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_compute_lines(["void main() { }"])
        cache.get_or_build_compute(b)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_hash_sources_single_arg(self):
        """hash_sources should work with a single source (compute)."""
        h = hash_sources("some compute shader")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


# ═══════════════════════════════════════════════════════════════════════
# Clone preserves compute state
# ═══════════════════════════════════════════════════════════════════════


class TestCloneCompute:
    def test_clone_preserves_compute_state(self):
        b = ShaderBuilder()
        b.add_local_size(128, 2)
        b.add_define("TEST", stage=ShaderStage.COMPUTE)
        b.add_ssbo(0, "Buf", "float d[];", stage=ShaderStage.COMPUTE)
        b.add_compute_lines(["void main() { }"])

        c = b.clone()
        src_b = b.build_compute()
        src_c = c.build_compute()
        assert src_b == src_c

    def test_clone_independence(self):
        b = ShaderBuilder()
        b.add_local_size(64)
        b.add_compute_lines(["void main() { }"])

        c = b.clone()
        c.add_define("EXTRA", stage=ShaderStage.COMPUTE)

        assert "EXTRA" not in b.build_compute()
        assert "EXTRA" in c.build_compute()
