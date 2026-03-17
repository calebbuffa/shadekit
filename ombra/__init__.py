"""Ombra — Python shader composition package.

Core package providing a language-agnostic AST, type system, decorators,
and compiler utilities.  GLSL-specific code (emitter, builder, builtins,
program) lives in :mod:`ombra.glsl`.

::

    # Core (language-agnostic):
    from ombra.types import Vec3, Float, ShaderMeta
    from ombra.ast import Variable, BinaryOp
    from ombra.decorators import shader_function

    # GLSL backend:
    from ombra.glsl import Program, ShaderBuilder, emit
"""
