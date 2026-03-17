"""shadekit — Python shader composition package.

Core package providing a language-agnostic AST, type system, decorators,
and compiler utilities.  GLSL-specific code (emitter, builder, builtins,
program) lives in :mod:`shadekit.glsl`.

::

    # Core (language-agnostic):
    from shadekit.types import Vec3, Float, ShaderMeta
    from shadekit.ast import Variable, BinaryOp
    from shadekit.decorators import shader_function

    # GLSL backend:
    from shadekit.glsl import Program, ShaderBuilder, emit
"""
