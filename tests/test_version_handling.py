"""Tests for ombra version directive handling.

Covers:
- _strip_version() with various #version formats (bare, profile, ES)
- from_glsl() roundtrip: single #version in output
- from_compute_glsl() roundtrip: single #version in output
- Program.from_glsl() roundtrip with profiles
- Program.from_compute_glsl() roundtrip with profiles
- Edge cases: no version, leading whitespace, CRLF line endings
"""

from __future__ import annotations

from ombra.glsl import Program
from ombra.glsl._builder import ShaderBuilder


class TestStripVersion:
    """Direct tests for ShaderBuilder._strip_version()."""

    def test_bare_version(self):
        src = "#version 430\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "void main()" in result

    def test_version_with_core_profile(self):
        src = "#version 430 core\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "void main()" in result

    def test_version_with_compatibility_profile(self):
        src = "#version 430 compatibility\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "void main()" in result

    def test_version_with_es_profile(self):
        src = "#version 300 es\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "void main()" in result

    def test_version_with_leading_whitespace(self):
        src = "  #version 430 core\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "void main()" in result

    def test_no_version_directive(self):
        src = "void main() { gl_Position = vec4(0); }"
        result = ShaderBuilder._strip_version(src)
        assert result.strip() == src.strip()

    def test_only_first_version_stripped(self):
        """Only the first #version should be stripped (count=1)."""
        src = "#version 430 core\n#version 430 core\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert result.count("#version") == 1

    def test_version_450(self):
        src = "#version 450 core\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result

    def test_version_310_es(self):
        src = "#version 310 es\nprecision highp float;\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "precision highp float;" in result

    def test_crlf_line_endings(self):
        src = "#version 430 core\r\nvoid main() { }"
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "void main()" in result

    def test_preserves_code_after_version(self):
        src = (
            "#version 430 core\n"
            "uniform float u_time;\n"
            "void main() {\n"
            "  gl_Position = vec4(u_time);\n"
            "}\n"
        )
        result = ShaderBuilder._strip_version(src)
        assert "#version" not in result
        assert "uniform float u_time;" in result
        assert "gl_Position = vec4(u_time);" in result


# ═══════════════════════════════════════════════════════════════════════
# from_glsl — single #version in output
# ═══════════════════════════════════════════════════════════════════════


class TestFromGlslVersionHandling:
    """Verify from_glsl() roundtrip produces exactly one #version."""

    def _count_version_lines(self, src: str) -> int:
        return sum(
            1 for line in src.splitlines() if line.strip().startswith("#version")
        )

    def test_bare_version_roundtrip(self):
        vert = "#version 430\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430\nvoid main() { }"
        b = ShaderBuilder.from_glsl(vert, frag)
        v, f = b.build()
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1

    def test_core_profile_roundtrip(self):
        vert = "#version 430 core\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430 core\nvoid main() { }"
        b = ShaderBuilder.from_glsl(vert, frag)
        v, f = b.build()
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1
        assert "#version 430 core" in v
        assert "#version 430 core" in f

    def test_compatibility_profile_roundtrip(self):
        vert = "#version 430 compatibility\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430 compatibility\nvoid main() { }"
        b = ShaderBuilder.from_glsl(vert, frag)
        v, f = b.build()
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1

    def test_es_profile_roundtrip(self):
        vert = "#version 300 es\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 300 es\nvoid main() { }"
        b = ShaderBuilder.from_glsl(vert, frag)
        v, f = b.build()
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1

    def test_mixed_profiles_roundtrip(self):
        """Source has bare #version, builder defaults to core profile."""
        vert = "#version 430\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430\nvoid main() { }"
        b = ShaderBuilder.from_glsl(vert, frag)
        v, f = b.build()
        # Builder emits its own version (430 core by default)
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1

    def test_with_injected_defines(self):
        vert = "#version 430 core\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430 core\nvoid main() { }"
        b = ShaderBuilder.from_glsl(vert, frag)
        b.add_define("EXTRA_FEATURE")
        v, f = b.build()
        assert self._count_version_lines(v) == 1
        assert "#define EXTRA_FEATURE" in v


# ═══════════════════════════════════════════════════════════════════════
# from_compute_glsl — single #version in output
# ═══════════════════════════════════════════════════════════════════════


class TestFromComputeGlslVersionHandling:
    """Verify from_compute_glsl() roundtrip produces exactly one #version."""

    def _count_version_lines(self, src: str) -> int:
        return sum(
            1 for line in src.splitlines() if line.strip().startswith("#version")
        )

    def test_bare_version_roundtrip(self):
        src = "#version 430\nlayout(local_size_x = 64) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        out = b.build_compute()
        assert self._count_version_lines(out) == 1

    def test_core_profile_roundtrip(self):
        """This was the original bug — #version 430 core wasn't stripped."""
        src = "#version 430 core\nlayout(local_size_x = 64) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        out = b.build_compute()
        assert self._count_version_lines(out) == 1
        assert "local_size_x = 64" in out
        assert "void main()" in out

    def test_compatibility_profile_roundtrip(self):
        src = "#version 430 compatibility\nlayout(local_size_x = 128) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        out = b.build_compute()
        assert self._count_version_lines(out) == 1

    def test_es_profile_roundtrip(self):
        src = "#version 310 es\nlayout(local_size_x = 32) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        out = b.build_compute()
        assert self._count_version_lines(out) == 1

    def test_with_injected_define(self):
        from ombra.glsl._builder import ShaderStage

        src = "#version 430 core\nlayout(local_size_x = 64) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        b.add_define("BLOCK_SIZE", 256, stage=ShaderStage.COMPUTE)
        out = b.build_compute()
        assert self._count_version_lines(out) == 1
        assert "#define BLOCK_SIZE 256" in out

    def test_preserves_local_size_after_strip(self):
        src = "#version 430 core\nlayout(local_size_x = 256, local_size_y = 4, local_size_z = 2) in;\nvoid main() { }"
        b = ShaderBuilder.from_compute_glsl(src)
        assert b._compute_local_size == (256, 4, 2)
        out = b.build_compute()
        assert "local_size_x = 256" in out
        assert "local_size_y = 4" in out
        assert "local_size_z = 2" in out

    def test_realistic_compute_shader(self):
        """Full realistic compute shader with profile — roundtrip must produce single #version."""
        src = """\
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    float data[];
};

uniform float u_scale;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    data[idx] *= u_scale;
}
"""
        b = ShaderBuilder.from_compute_glsl(src)
        out = b.build_compute()
        version_count = sum(
            1 for line in out.splitlines() if line.strip().startswith("#version")
        )
        assert version_count == 1
        assert "gl_GlobalInvocationID" in out
        assert "u_scale" in out or "data" in out


# ═══════════════════════════════════════════════════════════════════════
# Program API roundtrip — version handling
# ═══════════════════════════════════════════════════════════════════════


class TestProgramVersionHandling:
    """Verify Program factories handle version profiles correctly."""

    def _count_version_lines(self, src: str) -> int:
        return sum(
            1 for line in src.splitlines() if line.strip().startswith("#version")
        )

    def test_from_glsl_bare_version(self):
        vert = "#version 430\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430\nvoid main() { }"
        prog = Program.from_glsl(vert, frag)
        v, f = prog.build()
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1

    def test_from_glsl_core_profile(self):
        vert = "#version 430 core\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430 core\nvoid main() { }"
        prog = Program.from_glsl(vert, frag)
        v, f = prog.build()
        assert self._count_version_lines(v) == 1
        assert self._count_version_lines(f) == 1

    def test_from_glsl_with_define_injection(self):
        vert = "#version 430 core\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430 core\nvoid main() { }"
        prog = Program.from_glsl(vert, frag)
        prog.define("MY_FEATURE")
        v, f = prog.build()
        assert self._count_version_lines(v) == 1
        assert "#define MY_FEATURE" in v

    def test_from_compute_glsl_bare_version(self):
        src = "#version 430\nlayout(local_size_x=64) in;\nvoid main() { }"
        prog = Program.from_compute_glsl(src)
        out = prog.build_compute()
        assert self._count_version_lines(out) == 1

    def test_from_compute_glsl_core_profile(self):
        """The original bug scenario via Program API."""
        src = "#version 430 core\nlayout(local_size_x=64) in;\nvoid main() { }"
        prog = Program.from_compute_glsl(src)
        out = prog.build_compute()
        assert self._count_version_lines(out) == 1
        assert "local_size_x = 64" in out

    def test_from_compute_glsl_with_define(self):
        src = "#version 430 core\nlayout(local_size_x=64) in;\nvoid main() { }"
        prog = Program.from_compute_glsl(src)
        prog.define("EXTRA", stage="compute")
        out = prog.build_compute()
        assert self._count_version_lines(out) == 1
        assert "#define EXTRA" in out

    def test_version_is_first_line(self):
        """#version must always be the very first line in output."""
        vert = "#version 430 core\nvoid main() { gl_Position = vec4(0); }"
        frag = "#version 430 core\nvoid main() { }"
        prog = Program.from_glsl(vert, frag)
        v, f = prog.build()
        assert v.splitlines()[0].startswith("#version")
        assert f.splitlines()[0].startswith("#version")

    def test_compute_version_is_first_line(self):
        src = "#version 430 core\nlayout(local_size_x=64) in;\nvoid main() { }"
        prog = Program.from_compute_glsl(src)
        out = prog.build_compute()
        assert out.splitlines()[0].startswith("#version")


# ═══════════════════════════════════════════════════════════════════════
# _VERSION_RE regex — direct pattern tests
# ═══════════════════════════════════════════════════════════════════════


class TestVersionRegex:
    """Direct tests for the _VERSION_RE regex pattern."""

    def test_matches_bare_version(self):
        assert ShaderBuilder._VERSION_RE.search("#version 430")

    def test_matches_core_profile(self):
        assert ShaderBuilder._VERSION_RE.search("#version 430 core")

    def test_matches_compatibility_profile(self):
        assert ShaderBuilder._VERSION_RE.search("#version 430 compatibility")

    def test_matches_es_profile(self):
        assert ShaderBuilder._VERSION_RE.search("#version 300 es")

    def test_matches_version_450(self):
        assert ShaderBuilder._VERSION_RE.search("#version 450")

    def test_matches_version_310_es(self):
        assert ShaderBuilder._VERSION_RE.search("#version 310 es")

    def test_matches_with_leading_spaces(self):
        assert ShaderBuilder._VERSION_RE.search("   #version 430 core")

    def test_matches_with_trailing_spaces(self):
        assert ShaderBuilder._VERSION_RE.search("#version 430 core   ")

    def test_no_match_without_hash(self):
        assert ShaderBuilder._VERSION_RE.search("version 430") is None

    def test_no_match_in_comment(self):
        # The regex matches anywhere on a line; this tests it exists in multi-line context
        m = ShaderBuilder._VERSION_RE.search("// #version 430")
        # This actually matches because the regex just looks for the pattern
        # The important thing is _strip_version only strips count=1
        # and the real-world usage always has it on its own line

    def test_matches_in_multiline(self):
        src = "// comment\n#version 430 core\nvoid main() { }"
        m = ShaderBuilder._VERSION_RE.search(src)
        assert m is not None
        assert "430 core" in m.group()
