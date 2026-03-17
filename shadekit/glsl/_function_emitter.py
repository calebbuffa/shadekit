"""Emit shader function definitions from :class:`ShaderFunction` AST."""

from __future__ import annotations

from shadekit.decorators._function import ShaderFunction
from shadekit.glsl._emitter import emit_stmt


def emit_function(fn: ShaderFunction) -> str:
    """Emit a complete GLSL function definition.

    Returns a string like::

        float luminance(vec3 c) {
            return dot(c, vec3(0.299, 0.587, 0.114));
        }
    """
    lines: list[str] = [f"{fn.signature()} {{"]
    for stmt in fn.body:
        lines.append(emit_stmt(stmt, indent=1).rstrip("\n"))
    lines.append("}")
    return "\n".join(lines) + "\n"
