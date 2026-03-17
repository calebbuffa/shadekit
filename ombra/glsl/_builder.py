"""Programmatic GLSL shader assembly.

:class:`ShaderBuilder` composes GLSL 430 vertex + fragment (or compute)
shaders from dynamically-added defines, uniforms, SSBOs, varyings,
structs, functions, and main-body code lines.

Inspired by CesiumJS ``ShaderBuilder``, adapted for OpenGL 4.3 / SSBO
vertex pulling.

Usage::

    from ombra import ShaderBuilder, ShaderStage

    b = ShaderBuilder()
    b.add_define("HAS_NORMALS")
    b.add_uniform("mat4", "u_mvp", ShaderStage.VERTEX)
    b.add_ssbo(0, "Pos", "float pos_data[];", ShaderStage.VERTEX)
    b.add_varying("vec3", "v_normal")
    b.add_vertex_lines(["void main() {", "  gl_Position = ...;", "}"])
    b.add_fragment_lines(["void main() {", "  f_color = ...;", "}"])
    vert_src, frag_src = b.build()

To wrap existing ``.glsl`` files for composition::

    vert = Path("MeshVS.glsl").read_text()
    frag = Path("MeshFS.glsl").read_text()
    builder = ShaderBuilder.from_glsl(vert, frag)
    builder.add_define("CUSTOM_FEATURE")
    vert_src, frag_src = builder.build()
"""

from __future__ import annotations

import enum
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ombra.decorators._function import ShaderFunction

from ombra.ast._statements import Stmt
from ombra.types._structs import StructType

_LOCAL_SIZE_RE = re.compile(
    r"^\s*layout\s*\("
    r"\s*local_size_x\s*=\s*(\d+)"
    r"(?:\s*,\s*local_size_y\s*=\s*(\d+))?"
    r"(?:\s*,\s*local_size_z\s*=\s*(\d+))?"
    r"\s*\)\s*in\s*;",
    re.MULTILINE,
)


class ShaderStage(enum.IntFlag):
    """Which shader stage(s) a declaration targets."""

    VERTEX = 1
    FRAGMENT = 2
    BOTH = VERTEX | FRAGMENT
    COMPUTE = 4


class ShaderBuilder:
    """Composable GLSL shader assembly.

    Collects defines, uniforms, SSBOs, varyings, structs, functions,
    and code lines for vertex and fragment shaders independently.
    Call :meth:`build` to produce the final GLSL source strings.
    """

    __slots__ = (
        "_version",
        "_profile",
        "_vert_defines",
        "_frag_defines",
        "_vert_uniforms",
        "_frag_uniforms",
        "_vert_ssbos",
        "_frag_ssbos",
        "_vert_varyings",
        "_frag_varyings",
        "_vert_outputs",
        "_frag_outputs",
        "_structs",
        "_functions",
        "_vert_struct_ids",
        "_frag_struct_ids",
        "_vert_function_ids",
        "_frag_function_ids",
        "_vert_lines",
        "_frag_lines",
        "_vert_glsl_functions",
        "_frag_glsl_functions",
        "_vert_ast_stmts",
        "_frag_ast_stmts",
        "_compute_defines",
        "_compute_uniforms",
        "_compute_ssbos",
        "_compute_struct_ids",
        "_compute_function_ids",
        "_compute_lines",
        "_compute_glsl_functions",
        "_compute_ast_stmts",
        "_compute_local_size",
        "_extensions",
        "_compute_shared",
        "_vert_inputs",
    )

    def __init__(self, *, version: str = "430", profile: str = "core") -> None:
        self._version = version
        self._profile = profile

        # Per-stage line accumulators
        self._vert_defines: list[str] = []
        self._frag_defines: list[str] = []
        self._vert_uniforms: list[str] = []
        self._frag_uniforms: list[str] = []
        self._vert_ssbos: list[str] = []
        self._frag_ssbos: list[str] = []
        self._vert_varyings: list[str] = []
        self._frag_varyings: list[str] = []
        self._vert_outputs: list[str] = []
        self._frag_outputs: list[str] = []

        # Named structs and functions (shared across stages)
        self._structs: dict[str, _ShaderStruct] = {}
        self._functions: dict[str, _ShaderFunction] = {}
        self._vert_struct_ids: list[str] = []
        self._frag_struct_ids: list[str] = []
        self._vert_function_ids: list[str] = []
        self._frag_function_ids: list[str] = []

        # Main body code (appended last)
        self._vert_lines: list[str] = []
        self._frag_lines: list[str] = []

        # ShaderFunction objects (Phase 4 — resolved via DependencyGraph)
        self._vert_glsl_functions: list[ShaderFunction] = []
        self._frag_glsl_functions: list[ShaderFunction] = []

        # AST statements for main body (emitted after string lines)
        self._vert_ast_stmts: list[Stmt] = []
        self._frag_ast_stmts: list[Stmt] = []

        # Compute stage (independent of vertex/fragment)
        self._compute_defines: list[str] = []
        self._compute_uniforms: list[str] = []
        self._compute_ssbos: list[str] = []
        self._compute_struct_ids: list[str] = []
        self._compute_function_ids: list[str] = []
        self._compute_lines: list[str] = []
        self._compute_glsl_functions: list[ShaderFunction] = []
        self._compute_ast_stmts: list[Stmt] = []
        self._compute_local_size: tuple[int, int, int] | None = None

        # Extensions (emitted right after #version)
        self._extensions: list[str] = []
        # Compute shared variables
        self._compute_shared: list[str] = []
        # Vertex attribute inputs
        self._vert_inputs: list[str] = []

    _VERSION_RE = re.compile(r"^\s*#version\s+\d+(?:\s+\w+)?\s*$", re.MULTILINE)

    @classmethod
    def from_glsl(cls, vert_src: str, frag_src: str) -> "ShaderBuilder":
        """Create a builder pre-loaded with existing GLSL source.

        The ``#version`` directive is stripped from both sources (the
        builder emits its own).  Everything else is stored as code lines
        so that :meth:`build` can prepend additional declarations
        (custom uniforms, structs, etc.) injected via the builder API.
        """
        b = cls()
        b._vert_lines = cls._strip_version(vert_src).splitlines()
        b._frag_lines = cls._strip_version(frag_src).splitlines()
        return b

    @classmethod
    def _strip_version(cls, src: str) -> str:
        return cls._VERSION_RE.sub("", src, count=1).lstrip("\n")

    def add_define(
        self,
        name: str,
        value: str | int | float | None = None,
        stage: ShaderStage = ShaderStage.BOTH,
    ) -> "ShaderBuilder":
        """Add a ``#define`` directive."""
        line = f"#define {name}" if value is None else f"#define {name} {value}"
        if ShaderStage.VERTEX in stage:
            self._vert_defines.append(line)
        if ShaderStage.FRAGMENT in stage:
            self._frag_defines.append(line)
        if ShaderStage.COMPUTE in stage:
            self._compute_defines.append(line)
        return self

    def add_extension(
        self,
        name: str,
        behavior: str = "enable",
    ) -> "ShaderBuilder":
        """Add a ``#extension`` directive (emitted right after ``#version``).

        Parameters
        ----------
        name : str
            Extension name (e.g. ``"GL_ARB_gpu_shader5"``).
        behavior : str
            One of ``"enable"``, ``"require"``, ``"warn"``, ``"disable"``.
        """
        self._extensions.append(f"#extension {name} : {behavior}")
        return self

    def add_vertex_input(
        self,
        location: int,
        glsl_type: str,
        name: str,
    ) -> "ShaderBuilder":
        """Add a vertex attribute input (``layout(location=N) in type name;``).

        Parameters
        ----------
        location : int
            Attribute location.
        glsl_type : str
            GLSL type (e.g. ``"vec3"``).
        name : str
            Attribute name.
        """
        self._vert_inputs.append(
            f"layout(location = {location}) in {glsl_type} {name};"
        )
        return self

    def add_uniform(
        self,
        glsl_type: str,
        name: str,
        stage: ShaderStage = ShaderStage.BOTH,
        *,
        binding: int | None = None,
    ) -> "ShaderBuilder":
        """Add a uniform declaration.

        Parameters
        ----------
        glsl_type : str
            GLSL type (e.g. ``"mat4"``, ``"sampler2D"``).
        name : str
            Uniform name.
        stage : ShaderStage
            Target stage(s).
        binding : int | None
            Optional ``layout(binding=N)`` qualifier (useful for samplers).
        """
        if binding is not None:
            line = f"layout(binding = {binding}) uniform {glsl_type} {name};"
        else:
            line = f"uniform {glsl_type} {name};"
        if ShaderStage.VERTEX in stage:
            self._vert_uniforms.append(line)
        if ShaderStage.FRAGMENT in stage:
            self._frag_uniforms.append(line)
        if ShaderStage.COMPUTE in stage:
            self._compute_uniforms.append(line)
        return self

    def add_ssbo(
        self,
        binding: int,
        block_name: str,
        body: str,
        stage: ShaderStage = ShaderStage.VERTEX,
        *,
        readonly: bool = True,
    ) -> "ShaderBuilder":
        """Add an SSBO (shader storage buffer) declaration."""
        qual = "readonly " if readonly else ""
        line = f"layout(std430, binding = {binding}) {qual}buffer {block_name} {{ {body} }};"
        if ShaderStage.VERTEX in stage:
            self._vert_ssbos.append(line)
        if ShaderStage.FRAGMENT in stage:
            self._frag_ssbos.append(line)
        if ShaderStage.COMPUTE in stage:
            self._compute_ssbos.append(line)
        return self

    def add_varying(
        self,
        glsl_type: str,
        name: str,
        *,
        flat: bool = False,
        noperspective: bool = False,
        smooth: bool = False,
        centroid: bool = False,
    ) -> "ShaderBuilder":
        """Add a varying between vertex and fragment shaders.

        Parameters
        ----------
        glsl_type : str
            GLSL type.
        name : str
            Varying name.
        flat : bool
            ``flat`` interpolation (no interpolation).
        noperspective : bool
            ``noperspective`` interpolation (screen-space linear).
        smooth : bool
            ``smooth`` interpolation (the default, rarely needs explicit).
        centroid : bool
            ``centroid`` sample location qualifier.
        """
        qualifiers = []
        if flat:
            qualifiers.append("flat")
        elif noperspective:
            qualifiers.append("noperspective")
        elif smooth:
            qualifiers.append("smooth")
        if centroid:
            qualifiers.append("centroid")
        prefix = " ".join(qualifiers) + " " if qualifiers else ""
        self._vert_varyings.append(f"{prefix}out {glsl_type} {name};")
        self._frag_varyings.append(f"{prefix}in {glsl_type} {name};")
        return self

    def add_fragment_output(
        self,
        location: int,
        glsl_type: str,
        name: str,
    ) -> "ShaderBuilder":
        """Add a fragment output declaration."""
        self._frag_outputs.append(
            f"layout(location = {location}) out {glsl_type} {name};"
        )
        return self

    def add_struct(
        self,
        struct_id: str,
        struct_name: str,
        stage: ShaderStage,
    ) -> "ShaderBuilder":
        """Declare a named struct."""
        self._structs[struct_id] = _ShaderStruct(struct_name)
        if ShaderStage.VERTEX in stage:
            self._vert_struct_ids.append(struct_id)
        if ShaderStage.FRAGMENT in stage:
            self._frag_struct_ids.append(struct_id)
        if ShaderStage.COMPUTE in stage:
            self._compute_struct_ids.append(struct_id)
        return self

    def add_struct_field(
        self,
        struct_id: str,
        glsl_type: str,
        name: str,
    ) -> "ShaderBuilder":
        """Add a field to a previously declared struct."""
        self._structs[struct_id].fields.append(f"    {glsl_type} {name};")
        return self

    def add_struct_type(
        self,
        struct_type: StructType,
        stage: ShaderStage,
    ) -> "ShaderBuilder":
        """Add a :class:`~ombra.types.StructType` declaration.

        Generates the ``struct … { … };`` block from the type's
        :meth:`declaration` and injects it into the specified stage(s).
        """

        if not isinstance(struct_type, StructType):
            raise TypeError(f"Expected StructType, got {type(struct_type).__name__}")
        sid = struct_type.glsl_name
        s = _ShaderStruct(struct_type.glsl_name)
        s._declaration_lines = struct_type.declaration().splitlines()
        self._structs[sid] = s
        if ShaderStage.VERTEX in stage:
            self._vert_struct_ids.append(sid)
        if ShaderStage.FRAGMENT in stage:
            self._frag_struct_ids.append(sid)
        if ShaderStage.COMPUTE in stage:
            self._compute_struct_ids.append(sid)
        return self

    def add_function(
        self,
        func_id: str,
        signature: str,
        stage: ShaderStage,
    ) -> "ShaderBuilder":
        """Declare a named function."""
        self._functions[func_id] = _ShaderFunction(signature)
        if ShaderStage.VERTEX in stage:
            self._vert_function_ids.append(func_id)
        if ShaderStage.FRAGMENT in stage:
            self._frag_function_ids.append(func_id)
        if ShaderStage.COMPUTE in stage:
            self._compute_function_ids.append(func_id)
        return self

    def add_function_lines(
        self,
        func_id: str,
        lines: list[str] | str,
    ) -> "ShaderBuilder":
        """Append body lines to a previously declared function."""
        fn = self._functions[func_id]
        if isinstance(lines, str):
            fn.body.append(lines)
        else:
            fn.body.extend(lines)
        return self

    def add_glsl_function(
        self,
        fn: ShaderFunction,
        stage: ShaderStage = ShaderStage.BOTH,
    ) -> "ShaderBuilder":
        """Register a :class:`ShaderFunction` for emission.

        The function (and its transitive dependencies) will be emitted
        in dependency order before the main body.
        """
        if ShaderStage.VERTEX in stage:
            self._vert_glsl_functions.append(fn)
        if ShaderStage.FRAGMENT in stage:
            self._frag_glsl_functions.append(fn)
        if ShaderStage.COMPUTE in stage:
            self._compute_glsl_functions.append(fn)
        return self

    def add_vertex_stmts(self, stmts: Sequence[Stmt] | Stmt) -> "ShaderBuilder":
        """Append AST statement(s) to the vertex shader main body."""
        if isinstance(stmts, Stmt):
            self._vert_ast_stmts.append(stmts)
        else:
            self._vert_ast_stmts.extend(stmts)
        return self

    def add_fragment_stmts(self, stmts: Sequence[Stmt] | Stmt) -> "ShaderBuilder":
        """Append AST statement(s) to the fragment shader main body."""
        if isinstance(stmts, Stmt):
            self._frag_ast_stmts.append(stmts)
        else:
            self._frag_ast_stmts.extend(stmts)
        return self

    def add_vertex_lines(self, lines: list[str] | str) -> "ShaderBuilder":
        """Append raw GLSL lines to the vertex shader."""
        if isinstance(lines, str):
            self._vert_lines.append(lines)
        else:
            self._vert_lines.extend(lines)
        return self

    def add_fragment_lines(self, lines: list[str] | str) -> "ShaderBuilder":
        """Append raw GLSL lines to the fragment shader."""
        if isinstance(lines, str):
            self._frag_lines.append(lines)
        else:
            self._frag_lines.extend(lines)
        return self

    def add_local_size(self, x: int = 1, y: int = 1, z: int = 1) -> "ShaderBuilder":
        """Set the compute shader work-group size.

        Emits ``layout(local_size_x=X, local_size_y=Y, local_size_z=Z) in;``
        at the top of the compute shader.
        """
        self._compute_local_size = (x, y, z)
        return self

    def add_compute_lines(self, lines: list[str] | str) -> "ShaderBuilder":
        """Append raw GLSL lines to the compute shader."""
        if isinstance(lines, str):
            self._compute_lines.append(lines)
        else:
            self._compute_lines.extend(lines)
        return self

    def add_compute_stmts(self, stmts: Sequence[Stmt] | Stmt) -> "ShaderBuilder":
        """Append AST statement(s) to the compute shader main body."""
        if isinstance(stmts, Stmt):
            self._compute_ast_stmts.append(stmts)
        else:
            self._compute_ast_stmts.extend(stmts)
        return self

    def add_shared(
        self, glsl_type: str, name: str, *, array_size: int | None = None
    ) -> "ShaderBuilder":
        """Declare a ``shared`` variable for compute shaders.

        Parameters
        ----------
        glsl_type : str
            GLSL type (e.g. ``"float"``, ``"vec4"``).
        name : str
            Variable name.
        array_size : int | None
            If set, declares ``shared type name[size];``.
        """
        if array_size is not None:
            self._compute_shared.append(f"shared {glsl_type} {name}[{array_size}];")
        else:
            self._compute_shared.append(f"shared {glsl_type} {name};")
        return self

    @classmethod
    def from_compute_glsl(cls, source: str) -> "ShaderBuilder":
        """Create a builder pre-loaded with existing compute GLSL source.

        The ``#version`` and ``layout(local_size_*)`` directives are
        stripped (the builder emits its own).
        """
        b = cls()
        cleaned = cls._strip_version(source)
        # Extract local_size if present.
        ls_match = _LOCAL_SIZE_RE.search(cleaned)
        if ls_match:
            x = int(ls_match.group(1)) if ls_match.group(1) else 1
            y = int(ls_match.group(2)) if ls_match.group(2) else 1
            z = int(ls_match.group(3)) if ls_match.group(3) else 1
            b._compute_local_size = (x, y, z)
            cleaned = cleaned[: ls_match.start()] + cleaned[ls_match.end() :]
            cleaned = cleaned.lstrip("\n")
        b._compute_lines = cleaned.splitlines()
        return b

    def build(self) -> tuple[str, str]:
        """Assemble and return ``(vertex_source, fragment_source)``."""
        from ombra.glsl._assembler import assemble_stage

        version_str = (
            f"{self._version} {self._profile}" if self._profile else self._version
        )

        vert = assemble_stage(
            version=version_str,
            extensions=self._extensions,
            defines=self._vert_defines,
            uniforms=self._vert_uniforms,
            ssbos=self._vert_ssbos,
            varyings=self._vert_varyings,
            outputs=self._vert_outputs,
            struct_blocks=[
                self._structs[sid].generate() for sid in self._vert_struct_ids
            ],
            function_blocks=[
                self._functions[fid].generate() for fid in self._vert_function_ids
            ],
            glsl_functions=self._vert_glsl_functions,
            lines=self._vert_lines,
            ast_stmts=self._vert_ast_stmts,
            inputs=self._vert_inputs,
        )
        frag = assemble_stage(
            version=version_str,
            extensions=self._extensions,
            defines=self._frag_defines,
            uniforms=self._frag_uniforms,
            ssbos=self._frag_ssbos,
            varyings=self._frag_varyings,
            outputs=self._frag_outputs,
            struct_blocks=[
                self._structs[sid].generate() for sid in self._frag_struct_ids
            ],
            function_blocks=[
                self._functions[fid].generate() for fid in self._frag_function_ids
            ],
            glsl_functions=self._frag_glsl_functions,
            lines=self._frag_lines,
            ast_stmts=self._frag_ast_stmts,
        )
        return vert, frag

    def build_compute(self) -> str:
        """Assemble and return a compute shader source string.

        Uses the dedicated compute storage (populated via
        ``add_*(stage=ShaderStage.COMPUTE)``, :meth:`add_compute_lines`,
        :meth:`add_compute_stmts`, and :meth:`add_local_size`).
        """
        from ombra.glsl._assembler import assemble_stage

        version_str = (
            f"{self._version} {self._profile}" if self._profile else self._version
        )

        layout: list[str] = []
        if self._compute_local_size is not None:
            x, y, z = self._compute_local_size
            layout.append(
                f"layout(local_size_x = {x}, local_size_y = {y}, "
                f"local_size_z = {z}) in;"
            )

        return assemble_stage(
            version=version_str,
            extensions=self._extensions,
            layout_qualifiers=layout,
            defines=self._compute_defines,
            uniforms=self._compute_uniforms,
            ssbos=self._compute_ssbos,
            varyings=[],
            outputs=[],
            struct_blocks=[
                self._structs[sid].generate() for sid in self._compute_struct_ids
            ],
            function_blocks=[
                self._functions[fid].generate() for fid in self._compute_function_ids
            ],
            glsl_functions=self._compute_glsl_functions,
            lines=self._compute_lines,
            ast_stmts=self._compute_ast_stmts,
            shared_vars=self._compute_shared,
        )

    def clone(self) -> "ShaderBuilder":
        """Return a deep copy of this builder."""
        import copy

        return copy.deepcopy(self)


class _ShaderStruct:
    """Accumulates fields for a GLSL struct."""

    __slots__ = ("name", "fields", "_declaration_lines")

    def __init__(self, name: str) -> None:
        self.name = name
        self.fields: list[str] = []
        self._declaration_lines: list[str] | None = None

    def generate(self) -> list[str]:
        if self._declaration_lines is not None:
            return list(self._declaration_lines)
        lines = [f"struct {self.name} {{"]
        lines.extend(self.fields)
        lines.append("};")
        return lines


class _ShaderFunction:
    """Accumulates body lines for a GLSL function."""

    __slots__ = ("signature", "body")

    def __init__(self, signature: str) -> None:
        self.signature = signature
        self.body: list[str] = []

    def generate(self) -> list[str]:
        lines = [f"{self.signature} {{"]
        for line in self.body:
            lines.append(f"    {line}")
        lines.append("}")
        return lines
