"""Metaclass-based GLSL type system.

Each GLSL type (``Float``, ``Vec3``, ``Mat4``, etc.) is a Python **class**
created by :class:`GlslMeta`.  This means every pre-defined type is a valid
type annotation::

    @dataclass
    class Material:
        albedo: Vec3      # valid — Vec3 is a class
        roughness: Float  # valid — Float is a class
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class ShaderMeta(type):
    """Metaclass for shader type descriptors.

    Classes whose metaclass is ``ShaderMeta`` carry a ``glsl_name`` class
    attribute and compare / hash by that name.  This operates on the
    **class** level — ``repr(Float)`` returns ``'float'``, not
    ``'<class Float>'``.
    """

    glsl_name: str

    def __repr__(cls) -> str:
        return cls.glsl_name

    def __str__(cls) -> str:
        return cls.glsl_name

    def __eq__(cls, other: object) -> bool:
        if isinstance(other, ShaderMeta):
            return cls.glsl_name == other.glsl_name
        return NotImplemented

    def __hash__(cls) -> int:
        return hash(cls.glsl_name)


class ShaderType(metaclass=ShaderMeta):
    """Base class for all shader types.

    Subclass this to define concrete types.  The **class itself** is the
    type descriptor — do not instantiate it.

    Under ``TYPE_CHECKING``, instances also expose arithmetic and
    attribute-access stubs so that ``@shader_function`` bodies — where
    parameters are annotated with shader type classes but receive
    :class:`~ombra.ast.Expr` nodes at runtime — type-check correctly.
    """

    glsl_name = ""

    if TYPE_CHECKING:

        def __add__(self, other: Any) -> Any: ...
        def __radd__(self, other: Any) -> Any: ...
        def __sub__(self, other: Any) -> Any: ...
        def __rsub__(self, other: Any) -> Any: ...
        def __mul__(self, other: Any) -> Any: ...
        def __rmul__(self, other: Any) -> Any: ...
        def __truediv__(self, other: Any) -> Any: ...
        def __rtruediv__(self, other: Any) -> Any: ...
        def __mod__(self, other: Any) -> Any: ...
        def __neg__(self) -> Any: ...
        def __lt__(self, other: Any) -> Any: ...
        def __le__(self, other: Any) -> Any: ...
        def __gt__(self, other: Any) -> Any: ...
        def __ge__(self, other: Any) -> Any: ...
        def __getattr__(self, name: str) -> Any: ...
