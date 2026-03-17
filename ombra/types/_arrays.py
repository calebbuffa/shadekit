"""GLSL array type."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import ShaderMeta


class ArrayType:
    """A fixed-size GLSL array type.

    ``ArrayType(Vec3, 10)`` represents ``vec3[10]``.

    Attributes:
        glsl_name: The GLSL type string (e.g. ``"vec3[10]"``).
        base: Element type (a :class:`ShaderMeta` class or another ``ArrayType``).
        size: Number of elements.
    """

    __slots__ = ("glsl_name", "base", "size")

    def __init__(self, base: ShaderMeta | ArrayType, size: int) -> None:
        self.glsl_name: str = f"{base.glsl_name}[{size}]"
        self.base = base
        self.size = size

    def __repr__(self) -> str:
        return self.glsl_name

    def __str__(self) -> str:
        return self.glsl_name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArrayType):
            return self.glsl_name == other.glsl_name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.glsl_name)
