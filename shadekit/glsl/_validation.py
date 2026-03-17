"""GLSL shader validation.

Checks a :class:`ShaderBuilder` or raw GLSL source for common errors
before the host application attempts GPU compilation.

Usage::

    from shadekit.compiler import validate_shader
    errors = validate_shader(builder)
    for err in errors:
        print(err)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shadekit.glsl._builder import ShaderBuilder


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A single validation diagnostic."""

    severity: Severity
    stage: str  # "vertex", "fragment", or "both"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] ({self.stage}) {self.message}"


def validate_builder(builder: ShaderBuilder) -> list[ValidationError]:
    """Validate a :class:`ShaderBuilder` and return a list of diagnostics.

    Checks performed:
    - Missing ``main()`` entry point in generated source
    - Uniforms declared but never referenced in the stage body
    - ``gl_PointCoord`` used in the vertex stage
    - ``layout`` qualifiers require version ≥ 330
    """
    vert, frag = builder.build()
    errors: list[ValidationError] = []
    errors.extend(_check_main(vert, "vertex"))
    errors.extend(_check_main(frag, "fragment"))
    errors.extend(_check_unused_uniforms(vert, "vertex"))
    errors.extend(_check_unused_uniforms(frag, "fragment"))
    errors.extend(_check_stage_builtins(vert, frag))
    errors.extend(_check_version_features(vert, "vertex"))
    errors.extend(_check_version_features(frag, "fragment"))
    return errors


def validate_compute_builder(builder: ShaderBuilder) -> list[ValidationError]:
    """Validate a compute :class:`ShaderBuilder` and return diagnostics.

    Checks performed:
    - Missing ``main()`` entry point
    - Missing ``local_size`` layout qualifier
    - Graphics-only builtins used in compute stage
    - Uniforms declared but never referenced
    """
    source = builder.build_compute()
    errors: list[ValidationError] = []
    errors.extend(_check_main(source, "compute"))
    errors.extend(_check_unused_uniforms(source, "compute"))
    errors.extend(_check_compute_local_size(source))
    errors.extend(_check_compute_builtins(source))
    errors.extend(_check_version_features(source, "compute"))
    return errors


def validate_source(source: str, stage: str = "unknown") -> list[ValidationError]:
    """Validate a raw GLSL source string."""
    errors: list[ValidationError] = []
    errors.extend(_check_main(source, stage))
    errors.extend(_check_unused_uniforms(source, stage))
    errors.extend(_check_version_features(source, stage))
    return errors


_MAIN_RE = re.compile(r"\bvoid\s+main\s*\(")
_UNIFORM_RE = re.compile(r"^\s*uniform\s+\S+\s+(\w+)\s*;", re.MULTILINE)
_LAYOUT_RE = re.compile(r"\blayout\s*\(")
_VERSION_RE = re.compile(r"#version\s+(\d+)")
_LOCAL_SIZE_RE = re.compile(r"\blocal_size_x\b")


def _check_main(source: str, stage: str) -> list[ValidationError]:
    if not _MAIN_RE.search(source):
        return [ValidationError(Severity.WARNING, stage, "No main() entry point found")]
    return []


def _check_unused_uniforms(source: str, stage: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    uniforms = _UNIFORM_RE.findall(source)
    for uname in uniforms:
        # Count occurrences beyond the declaration line itself.
        pattern = re.compile(rf"\b{re.escape(uname)}\b")
        occurrences = pattern.findall(source)
        if len(occurrences) <= 1:
            errors.append(
                ValidationError(
                    Severity.WARNING,
                    stage,
                    f"Uniform '{uname}' declared but never referenced",
                )
            )
    return errors


def _check_stage_builtins(vert: str, frag: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    # gl_PointCoord is fragment-only.
    if "gl_PointCoord" in vert:
        errors.append(
            ValidationError(
                Severity.ERROR,
                "vertex",
                "gl_PointCoord is only available in the fragment shader",
            )
        )
    # gl_FragCoord is fragment-only.
    if "gl_FragCoord" in vert:
        errors.append(
            ValidationError(
                Severity.ERROR,
                "vertex",
                "gl_FragCoord is only available in the fragment shader",
            )
        )
    return errors


def _check_version_features(source: str, stage: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    m = _VERSION_RE.search(source)
    if m is None:
        return errors
    version = int(m.group(1))

    if version < 330 and _LAYOUT_RE.search(source):
        errors.append(
            ValidationError(
                Severity.ERROR,
                stage,
                f"layout() qualifiers require GLSL version >= 330 (found {version})",
            )
        )
    return errors


# Graphics-only builtins that must not appear in compute shaders.
_GRAPHICS_ONLY_BUILTINS = (
    "gl_Position",
    "gl_FragCoord",
    "gl_PointCoord",
    "gl_FragDepth",
    "gl_VertexID",
    "gl_InstanceID",
    "gl_FrontFacing",
)


def _check_compute_local_size(source: str) -> list[ValidationError]:
    if not _LOCAL_SIZE_RE.search(source):
        return [
            ValidationError(
                Severity.ERROR,
                "compute",
                "Compute shader missing layout(local_size_x=...) qualifier",
            )
        ]
    return []


def _check_compute_builtins(source: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for builtin in _GRAPHICS_ONLY_BUILTINS:
        if builtin in source:
            errors.append(
                ValidationError(
                    Severity.ERROR,
                    "compute",
                    f"{builtin} is not available in compute shaders",
                )
            )
    return errors
