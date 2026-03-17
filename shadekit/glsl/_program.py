"""High-level shader program composition.

:class:`Program` provides a declarative, Pythonic API for composing
complete GLSL shader programs.  It wraps :class:`ShaderBuilder` and
exposes a cleaner interface for the common case: define some structs,
uniforms, and SSBOs, write ``main()`` in Python, and build.

Usage — vertex + fragment raster program::

    from shadekit.glsl import Program, vec4, normalize
    from shadekit.types import Float, Vec3, Vec4, Mat4, UInt

    prog = Program()

    # Declare resources — each returns a Variable node you can use in code.
    u_mvp = prog.uniform("u_mvp", Mat4)
    u_albedo = prog.uniform("u_albedo", Vec3, stage="fragment")

    pos = prog.input(0, "Pos", "float pos_data[]")
    nrm = prog.input(1, "Nrm", "float nrm_data[]")
    idx = prog.input(3, "Idx", "uint idx_data[]")

    v_normal = prog.varying("v_normal", Vec3)
    f_color = prog.output(0, "f_color", Vec4)

    @prog.vertex
    def vs():
        # ...raw GLSL or AST code...
        pass

    @prog.fragment
    def fs():
        pass

    vert_src, frag_src = prog.build()

Usage — compute shader::

    from shadekit.glsl import Program

    prog = Program()
    prog.local_size(256)
    buf = prog.storage(0, "Data", "float data[]", readonly=False)

    @prog.compute
    def cs():
        pass

    compute_src = prog.build_compute()

For loading existing GLSL files with composition hooks::

    prog = Program.from_glsl(vert_src, frag_src)
    prog.define("HAS_FEATURE_IDS")
    vert, frag = prog.build()
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shadekit.ast._block import Block
from shadekit.ast._expressions import Variable
from shadekit.ast._statements import Stmt
from shadekit.compiler._ast_walk import (
    collect_shader_functions,
    collect_transitive_deps,
)
from shadekit.glsl._builder import ShaderBuilder, ShaderStage
from shadekit.types._base import ShaderMeta
from shadekit.types._structs import StructType

if TYPE_CHECKING:
    from shadekit.decorators._function import ShaderFunction


class Stage(str):
    """A single shader stage source — vertex, fragment, or compute.

    Subclasses :class:`str` so it works anywhere a plain string does
    (destructuring, hashing, passing to GL compilers) while carrying
    stage metadata and convenience methods.

    Mix-and-match usage::

        mesh_vs, mesh_fs = mesh_program.build()
        wire_vs, wire_fs = wireframe_program.build()

        # Combine freely — Stage IS a string
        registry.get_program(ctx, mesh_vs, wire_fs)

        # Save individual stages
        mesh_vs.save("mesh.vert.glsl")
        wire_fs.save("wire.frag.glsl")
    """

    __slots__ = ("_kind",)

    def __new__(cls, source: str, *, kind: str = "") -> "Stage":
        instance = super().__new__(cls, source)
        instance._kind = kind
        return instance

    @property
    def kind(self) -> str:
        """Stage kind: ``"vertex"``, ``"fragment"``, or ``"compute"``."""
        return self._kind

    def save(self, path: "str | Path") -> None:
        """Write this stage's GLSL source to *path*."""
        Path(path).write_text(str(self), encoding="utf-8")

    def __repr__(self) -> str:
        lines = str(self).count("\n") + 1
        return f"Stage(kind={self._kind!r}, lines={lines})"


class Program:
    """Declarative GLSL shader program.

    A higher-level wrapper around :class:`ShaderBuilder` that groups
    resource declarations and shader stage code in a single, coherent
    object.

    Resources (uniforms, SSBOs, varyings, outputs) are declared once
    and return :class:`Variable` AST nodes that can be used directly
    inside ``@glsl_function`` bodies or AST statement lists.

    Parameters
    ----------
    version : str
        GLSL version string (default ``"430"``).
    profile : str
        GLSL profile (default ``"core"``; also ``"compatibility"`` or ``""``).
    """

    __slots__ = ("_builder",)

    def __init__(self, *, version: str = "430", profile: str = "core") -> None:
        self._builder = ShaderBuilder(version=version, profile=profile)

    @classmethod
    def from_glsl(cls, vert_src: str, frag_src: str) -> "Program":
        """Create a program pre-loaded with existing GLSL source.

        The ``#version`` directive is stripped from each source.
        Everything else is stored as code lines so additional
        declarations can be prepended via the program API.
        """
        p = cls.__new__(cls)
        p._builder = ShaderBuilder.from_glsl(vert_src, frag_src)
        return p

    @classmethod
    def from_compute_glsl(cls, source: str) -> "Program":
        """Create a compute program pre-loaded with existing GLSL source."""
        p = cls.__new__(cls)
        p._builder = ShaderBuilder.from_compute_glsl(source)
        return p

    def extension(
        self,
        name: str,
        behavior: str = "enable",
    ) -> "Program":
        """Add a ``#extension`` directive.

        Parameters
        ----------
        name : str
            Extension name (e.g. ``"GL_ARB_gpu_shader5"``).
        behavior : str
            ``"enable"``, ``"require"``, ``"warn"``, or ``"disable"``.

        Returns
        -------
        Program
            Self, for chaining.
        """
        self._builder.add_extension(name, behavior)
        return self

    def vertex_input(
        self,
        location: int,
        name: str,
        glsl_type: ShaderMeta | str,
    ) -> Variable:
        """Declare a vertex attribute input and return a :class:`Variable`.

        Parameters
        ----------
        location : int
            Attribute location.
        name : str
            Attribute name.
        glsl_type : ShaderMeta | str
            GLSL type (e.g. ``Vec3``, ``"vec3"``).

        Returns
        -------
        Variable
            An AST node referencing this input.
        """
        type_str, meta = _resolve_type(glsl_type)
        self._builder.add_vertex_input(location, type_str, name)
        return Variable(name, meta)

    def define(
        self,
        name: str,
        value: str | int | float | None = None,
        *,
        stage: str = "both",
    ) -> "Program":
        """Add a ``#define`` directive.

        Parameters
        ----------
        name : str
            Macro name.
        value : str | int | float | None
            Optional macro value.
        stage : str
            ``"vertex"``, ``"fragment"``, ``"compute"``, or ``"both"``
            (default).
        """
        self._builder.add_define(name, value, _resolve_stage(stage))
        return self

    def uniform(
        self,
        name: str,
        glsl_type: ShaderMeta | str,
        *,
        stage: str = "both",
        binding: int | None = None,
    ) -> Variable:
        """Declare a uniform and return a :class:`Variable` for AST use.

        Parameters
        ----------
        name : str
            Uniform name.
        glsl_type : ShaderMeta | str
            GLSL type (e.g. ``Mat4``, ``Float``, or ``"sampler2D"``).
        stage : str
            ``"vertex"``, ``"fragment"``, ``"compute"``, or ``"both"``.
        binding : int | None
            Optional ``layout(binding=N)`` qualifier (useful for samplers).

        Returns
        -------
        Variable
            An AST node referencing this uniform.
        """
        type_str, meta = _resolve_type(glsl_type)
        self._builder.add_uniform(
            type_str, name, _resolve_stage(stage), binding=binding
        )
        return Variable(name, meta)

    def storage(
        self,
        binding: int,
        block_name: str,
        body: str,
        *,
        readonly: bool = True,
        stage: str = "vertex",
    ) -> "Program":
        """Declare an SSBO (shader storage buffer).

        Parameters
        ----------
        binding : int
            SSBO binding point.
        block_name : str
            Buffer block name in GLSL.
        body : str
            Inner body of the buffer (e.g. ``"float data[]"``).
        readonly : bool
            Whether the buffer is read-only (default ``True``).
        stage : str
            Target stage.

        Returns
        -------
        Program
            Self, for chaining.
        """
        self._builder.add_ssbo(
            binding, block_name, body, _resolve_stage(stage), readonly=readonly
        )
        return self

    def varying(
        self,
        name: str,
        glsl_type: ShaderMeta | str,
        *,
        flat: bool = False,
        noperspective: bool = False,
        smooth: bool = False,
        centroid: bool = False,
    ) -> Variable:
        """Declare a varying (vertex → fragment) and return a :class:`Variable`.

        Parameters
        ----------
        name : str
            Varying name.
        glsl_type : ShaderMeta | str
            GLSL type.
        flat : bool
            ``flat`` interpolation (no interpolation).
        noperspective : bool
            ``noperspective`` interpolation (screen-space linear).
        smooth : bool
            ``smooth`` interpolation (explicit, rarely needed).
        centroid : bool
            ``centroid`` sample location qualifier.

        Returns
        -------
        Variable
            An AST node referencing this varying.
        """
        type_str, meta = _resolve_type(glsl_type)
        self._builder.add_varying(
            type_str,
            name,
            flat=flat,
            noperspective=noperspective,
            smooth=smooth,
            centroid=centroid,
        )
        return Variable(name, meta)

    def output(
        self,
        location: int,
        name: str,
        glsl_type: ShaderMeta | str,
    ) -> Variable:
        """Declare a fragment output and return a :class:`Variable`.

        Parameters
        ----------
        location : int
            Output location.
        name : str
            Output variable name.
        glsl_type : ShaderMeta | str
            GLSL type.

        Returns
        -------
        Variable
            An AST node referencing this output.
        """
        type_str, meta = _resolve_type(glsl_type)
        self._builder.add_fragment_output(location, type_str, name)
        return Variable(name, meta)

    def struct(
        self,
        struct_type: StructType,
        *,
        stage: str = "fragment",
    ) -> "Program":
        """Declare a struct type.

        Parameters
        ----------
        struct_type : StructType
            The struct definition.
        stage : str
            Target stage(s).

        Returns
        -------
        Program
            Self, for chaining.
        """
        self._builder.add_struct_type(struct_type, _resolve_stage(stage))
        return self

    def shared(
        self,
        name: str,
        glsl_type: ShaderMeta | str,
        *,
        array_size: int | None = None,
    ) -> Variable:
        """Declare a ``shared`` variable for compute shaders.

        Parameters
        ----------
        name : str
            Variable name.
        glsl_type : ShaderMeta | str
            GLSL type (e.g. ``Float``, ``Vec4``).
        array_size : int | None
            If set, declares ``shared type name[size];``.

        Returns
        -------
        Variable
            An AST node referencing this shared variable.
        """
        type_str, meta = _resolve_type(glsl_type)
        self._builder.add_shared(type_str, name, array_size=array_size)
        return Variable(name, meta)

    def include(self, fn: ShaderFunction, *, stage: str = "both") -> "Program":
        """Include a :class:`ShaderFunction` in the program.

        The function and all its transitive dependencies are discovered
        automatically and emitted in dependency order before ``main()``.

        Parameters
        ----------
        fn : ShaderFunction
            The function to include.
        stage : str
            Target stage(s).

        Returns
        -------
        Program
            Self, for chaining.
        """
        resolved_stage = _resolve_stage(stage)
        for dep in collect_transitive_deps(fn):
            self._builder.add_glsl_function(dep, resolved_stage)
        return self

    def local_size(self, x: int = 1, y: int = 1, z: int = 1) -> "Program":
        """Set the compute shader work-group size.

        Parameters
        ----------
        x, y, z : int
            Work-group dimensions.

        Returns
        -------
        Program
            Self, for chaining.
        """
        self._builder.add_local_size(x, y, z)
        return self

    @property
    def vertex(self) -> _StageDecorator:
        """Decorator that captures raw GLSL lines for the vertex stage.

        Usage::

            @prog.vertex
            def vs():
                return '''
                void main() {
                    gl_Position = u_mvp * vec4(pos, 1.0);
                }
                '''
        """
        return _StageDecorator(self._builder, "vertex")

    @property
    def fragment(self) -> _StageDecorator:
        """Decorator that captures raw GLSL lines for the fragment stage."""
        return _StageDecorator(self._builder, "fragment")

    @property
    def compute(self) -> _StageDecorator:
        """Decorator that captures raw GLSL lines for the compute stage."""
        return _StageDecorator(self._builder, "compute")

    def build(self) -> tuple[Stage, Stage]:
        """Assemble and return ``(vertex_stage, fragment_stage)``.

        Each returned :class:`Stage` is a ``str`` subclass, so existing
        code that destructures into ``(vert_src, frag_src)`` works
        unchanged.  Stages can be saved individually and mixed with
        stages from other programs.
        """
        vert_src, frag_src = self._builder.build()
        return Stage(vert_src, kind="vertex"), Stage(frag_src, kind="fragment")

    def build_compute(self) -> Stage:
        """Assemble and return a compute :class:`Stage`."""
        return Stage(self._builder.build_compute(), kind="compute")

    def clone(self) -> "Program":
        """Return a deep copy of this program."""
        p = Program.__new__(Program)
        p._builder = self._builder.clone()
        return p

    @property
    def builder(self) -> ShaderBuilder:
        """Access the underlying :class:`ShaderBuilder` directly.

        Use this when the high-level API doesn't cover your use case.
        """
        return self._builder

    def __repr__(self) -> str:
        return f"Program(version={self._builder._version!r})"

    def __str__(self) -> str:
        """Return the assembled GLSL source.

        For raster programs (vertex + fragment), returns both stages
        separated by a ``// --- fragment ---`` marker.  For compute
        programs, returns the single compute source.
        """
        b = self._builder
        if b._compute_local_size is not None or b._compute_lines:
            return str(self.build_compute())
        vert, frag = self.build()
        return f"{vert}\n// --- fragment ---\n{frag}"

    def save(self, path: "str | Any") -> None:
        """Write the assembled GLSL source to *path*.

        Accepts a string path or a :class:`~pathlib.Path`.  For raster
        programs the vertex and fragment stages are written to
        ``<stem>.vert.glsl`` and ``<stem>.frag.glsl`` respectively.
        For compute programs a single file is written.
        """

        p = Path(path)
        b = self._builder
        if b._compute_local_size is not None or b._compute_lines:
            self.build_compute().save(p)
        else:
            vert, frag = self.build()
            stem = p.with_suffix("")  # strip any suffix
            vert.save(stem.with_suffix(".vert.glsl"))
            frag.save(stem.with_suffix(".frag.glsl"))


class _StageDecorator:
    """Captures a function's return value as GLSL code for a specific stage."""

    __slots__ = ("_builder", "_stage")

    def __init__(self, builder: ShaderBuilder, stage: str) -> None:
        self._builder = builder
        self._stage = stage

    def __call__(self, fn: Any) -> Any:
        """Decorate a function that returns GLSL source or AST statements.

        The function is called immediately.  Its return value is
        interpreted as:

        - ``str``: raw GLSL lines, added via ``add_*_lines``
        - ``list[Stmt]``, ``Block``, or ``Stmt``: AST statements, added via ``add_*_stmts``
        - ``None``: no-op (code was added via side effects)

        Any :func:`@glsl_function <shadekit.decorators.glsl_function>`
        calls found in the AST are automatically included in the shader
        (no manual ``prog.include()`` required).
        """
        result = fn()
        if result is None:
            return fn

        if isinstance(result, str):
            source = _dedent_glsl(result)
            if self._stage == "vertex":
                self._builder.add_vertex_lines(source.splitlines())
            elif self._stage == "fragment":
                self._builder.add_fragment_lines(source.splitlines())
            elif self._stage == "compute":
                self._builder.add_compute_lines(source.splitlines())
        elif isinstance(result, Block):
            stmts = list(result)
            self._add_stmts(stmts)
            self._auto_include(stmts)
        elif isinstance(result, Stmt):
            self._add_stmts(result)
            self._auto_include([result])
        elif isinstance(result, (list, tuple)):
            stmts = list(result)
            self._add_stmts(stmts)
            self._auto_include(stmts)
        else:
            raise TypeError(
                f"@prog.{self._stage}: expected str, Stmt, list[Stmt], Block, or None, "
                f"got {type(result).__name__}"
            )
        return fn

    def _add_stmts(self, stmts: list[Stmt] | Stmt) -> None:
        """Add statements to the correct stage."""
        if self._stage == "vertex":
            self._builder.add_vertex_stmts(stmts)
        elif self._stage == "fragment":
            self._builder.add_fragment_stmts(stmts)
        elif self._stage == "compute":
            self._builder.add_compute_stmts(stmts)

    def _auto_include(self, stmts: list[Stmt]) -> None:
        """Walk *stmts* and auto-include any referenced ``@shader_function``s."""
        stage = _resolve_stage(self._stage)
        for fn in collect_shader_functions(stmts):
            self._builder.add_glsl_function(fn, stage)


_STAGE_MAP = {
    "vertex": ShaderStage.VERTEX,
    "fragment": ShaderStage.FRAGMENT,
    "both": ShaderStage.BOTH,
    "compute": ShaderStage.COMPUTE,
}


def _resolve_stage(name: str) -> ShaderStage:
    """Convert a stage name string to a :class:`ShaderStage` flag."""
    try:
        return _STAGE_MAP[name]
    except KeyError:
        valid = ", ".join(sorted(_STAGE_MAP))
        raise ValueError(f"Unknown stage {name!r}; expected one of: {valid}") from None


def _resolve_type(
    glsl_type: Any,
) -> tuple[str, Any]:
    """Convert a ShaderMeta class or string to ``(glsl_name_str, meta_or_None)``.

    If *glsl_type* is a ShaderMeta subclass, returns ``(glsl_name, type_class)``.
    If *glsl_type* is a plain string (e.g. ``"sampler2D"``), returns ``(string, None)``.
    """

    if isinstance(glsl_type, str):
        return glsl_type, None
    if isinstance(glsl_type, ShaderMeta):
        return glsl_type.glsl_name, glsl_type
    raise TypeError(f"Expected ShaderMeta type or str, got {type(glsl_type).__name__}")


def _dedent_glsl(text: str) -> str:
    """Remove common leading whitespace from a GLSL text block.

    Similar to :func:`textwrap.dedent` but handles the common pattern of
    triple-quoted strings where the first line is blank.
    """

    return textwrap.dedent(text).strip("\n")
