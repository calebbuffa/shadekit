"""Stage assembly — compiles ShaderBuilder state into GLSL source.

The builder collects declarations; this module formats them into a
complete GLSL shader stage string.  Keeps emission logic (dependency
resolution, AST emission) in the compiler package where it belongs.
"""

from __future__ import annotations

from shadekit.ast._statements import Stmt
from shadekit.compiler._dependency_graph import DependencyGraph
from shadekit.decorators._function import ShaderFunction
from shadekit.glsl._emitter import emit_stmt
from shadekit.glsl._function_emitter import emit_function


def assemble_stage(
    *,
    version: str,
    defines: list[str],
    uniforms: list[str],
    ssbos: list[str],
    varyings: list[str],
    outputs: list[str],
    struct_blocks: list[list[str]],
    function_blocks: list[list[str]],
    glsl_functions: list[ShaderFunction],
    lines: list[str],
    ast_stmts: list[Stmt],
    layout_qualifiers: list[str] | None = None,
    extensions: list[str] | None = None,
    inputs: list[str] | None = None,
    shared_vars: list[str] | None = None,
) -> str:
    """Format a single shader stage into a GLSL source string.

    Parameters are the raw declaration lists accumulated by
    :class:`~shadekit.builder.ShaderBuilder`.  This function owns the
    dependency-graph resolution and AST emission.
    """
    parts: list[str] = [f"#version {version}"]

    if extensions:
        for ext in extensions:
            parts.append(ext)

    if layout_qualifiers:
        parts.append("")
        parts.extend(layout_qualifiers)

    if defines:
        parts.append("")
        parts.extend(defines)

    if inputs:
        parts.append("")
        parts.extend(inputs)

    for section in (uniforms, ssbos, varyings, outputs):
        if section:
            parts.append("")
            parts.extend(section)

    if shared_vars:
        parts.append("")
        parts.extend(shared_vars)

    for block in struct_blocks:
        parts.append("")
        parts.extend(block)

    for block in function_blocks:
        parts.append("")
        parts.extend(block)

    # Emit GlslFunction definitions in dependency order.
    if glsl_functions:
        graph = DependencyGraph()
        for fn in glsl_functions:
            graph.add(fn)
        for fn in graph.resolve():
            parts.append("")
            parts.extend(emit_function(fn).rstrip("\n").splitlines())

    if lines:
        parts.append("")
        parts.extend(lines)

    # Emit AST statements wrapped in void main().
    if ast_stmts:
        parts.append("")
        parts.append("void main() {")
        for stmt in ast_stmts:
            parts.append(emit_stmt(stmt, indent=1).rstrip("\n"))
        parts.append("}")

    parts.append("")  # trailing newline
    return "\n".join(parts)
