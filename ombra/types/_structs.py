"""GLSL struct type and ``@glsl_struct`` decorator."""

from __future__ import annotations

import dataclasses
import typing

from ._base import ShaderMeta


class StructType:
    """A named GLSL struct type.

    ``StructType("Material", {"albedo": Vec3, "roughness": Float})``
    represents::

        struct Material {
            vec3 albedo;
            float roughness;
        };

    Attributes:
        glsl_name: The struct name.
        fields: Ordered mapping of field name → GLSL type class.
    """

    __slots__ = ("glsl_name", "fields")

    def __init__(self, name: str, fields: dict[str, ShaderMeta]) -> None:
        self.glsl_name: str = name
        self.fields = fields

    def declaration(self) -> str:
        """Return the GLSL ``struct`` declaration."""
        lines = [f"struct {self.glsl_name} {{"]
        for field_name, field_type in self.fields.items():
            lines.append(f"    {field_type.glsl_name} {field_name};")
        lines.append("};")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.glsl_name

    def __str__(self) -> str:
        return self.glsl_name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StructType):
            return self.glsl_name == other.glsl_name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.glsl_name)


def shader_struct(cls: type) -> type:
    """Attach shader struct metadata to a :func:`dataclasses.dataclass`.

    Usage::

        from dataclasses import dataclass
        from ombra.types import Vec3, Float, shader_struct

        @shader_struct
        @dataclass
        class Material:
            albedo: Vec3
            roughness: Float
            metallic: Float

    The decorator adds two attributes to **cls**:

    ``__glsl_type__``
        A :class:`StructType` instance.
    ``__glsl_declaration__``
        The GLSL ``struct`` source string.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"@glsl_struct requires a dataclass, got {cls!r}")

    # Resolve annotations (handles ``from __future__ import annotations``).
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}

    fields: dict[str, ShaderMeta] = {}
    for f in dataclasses.fields(cls):
        ann = hints.get(f.name, f.type)
        if not isinstance(ann, ShaderMeta):
            raise TypeError(
                f"Field {cls.__name__}.{f.name} has type {ann!r}, "
                f"expected a ShaderType class (e.g. Float, Vec3)"
            )
        fields[f.name] = ann

    st = StructType(cls.__name__, fields)
    cls.__glsl_type__ = st  # type: ignore[attr-defined]
    cls.__glsl_declaration__ = st.declaration()  # type: ignore[attr-defined]
    return cls
