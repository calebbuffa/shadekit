"""GLSL vector type classes."""

from ._base import ShaderType
from ._scalars import Bool, Double, Float, Int, ScalarType, UInt


class VectorType(ShaderType):
    """Base for GLSL vector types (``vec3``, ``ivec2``, etc.).

    Class attributes:
        component_type: The scalar type of each component.
        size: Number of components (2, 3, or 4).
    """

    component_type: type[ScalarType]
    size: int


# float vectors
class Vec2(VectorType):
    glsl_name = "vec2"
    component_type = Float
    size = 2


class Vec3(VectorType):
    glsl_name = "vec3"
    component_type = Float
    size = 3


class Vec4(VectorType):
    glsl_name = "vec4"
    component_type = Float
    size = 4


# double vectors
class DVec2(VectorType):
    glsl_name = "dvec2"
    component_type = Double
    size = 2


class DVec3(VectorType):
    glsl_name = "dvec3"
    component_type = Double
    size = 3


class DVec4(VectorType):
    glsl_name = "dvec4"
    component_type = Double
    size = 4


# int vectors
class IVec2(VectorType):
    glsl_name = "ivec2"
    component_type = Int
    size = 2


class IVec3(VectorType):
    glsl_name = "ivec3"
    component_type = Int
    size = 3


class IVec4(VectorType):
    glsl_name = "ivec4"
    component_type = Int
    size = 4


# uint vectors
class UVec2(VectorType):
    glsl_name = "uvec2"
    component_type = UInt
    size = 2


class UVec3(VectorType):
    glsl_name = "uvec3"
    component_type = UInt
    size = 3


class UVec4(VectorType):
    glsl_name = "uvec4"
    component_type = UInt
    size = 4


# bool vectors
class BVec2(VectorType):
    glsl_name = "bvec2"
    component_type = Bool
    size = 2


class BVec3(VectorType):
    glsl_name = "bvec3"
    component_type = Bool
    size = 3


class BVec4(VectorType):
    glsl_name = "bvec4"
    component_type = Bool
    size = 4
