"""GLSL built-in function proxies for use inside ``@shader_function`` bodies.

Each proxy is a plain Python function that returns an AST
:class:`~ombra.ast.FunctionCall` or :class:`~ombra.ast.ConstructorCall`
node. This lets ``@shader_function``-decorated code read naturally::

    from ombra.glsl import dot, vec3
    from ombra.decorators import shader_function

    @shader_function
    def luminance(c: Vec3) -> Float:
        return dot(c, vec3(0.299, 0.587, 0.114))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import ombra.types as t
from ombra.ast._expressions import ConstructorCall, Expr, FunctionCall, Literal
from ombra.types._inference import _vec_for
from ombra.types._scalars import Bool, Float, Int, UInt
from ombra.types._vectors import Vec2, Vec4, VectorType

if TYPE_CHECKING:
    from ombra.types._base import ShaderMeta


def _coerce_arg(a: Any) -> Expr:
    """Convert a Python primitive to a Literal AST node if needed."""
    if isinstance(a, Expr):
        return a
    if isinstance(a, float):
        return Literal(a, Float)
    if isinstance(a, bool):
        return Literal(a, Bool)
    if isinstance(a, int):
        return Literal(a, Int)
    raise TypeError(f"Cannot coerce {type(a).__name__} to GLSL expression")


def _coerce_args(args: tuple[Any, ...]) -> list[Expr]:
    """Coerce a tuple of arguments, converting Python primitives to Literals."""
    return [_coerce_arg(a) for a in args]


def _ctor(target_type: ShaderMeta) -> callable:
    """Create a type constructor proxy for *target_type*."""

    def _proxy(*args: Expr | float | int | bool) -> ConstructorCall:
        return ConstructorCall(target_type, _coerce_args(args), target_type)

    return _proxy


def _lazy_ctor(type_name: str) -> callable:
    """Create a lazily-resolved type constructor (avoids circular imports)."""

    def _proxy(*args: Expr | float | int | bool) -> ConstructorCall:

        target = getattr(t, type_name)
        return _ctor(target)(*args)

    return _proxy


vec2 = _lazy_ctor("Vec2")
vec3 = _lazy_ctor("Vec3")
vec4 = _lazy_ctor("Vec4")
ivec2 = _lazy_ctor("IVec2")
ivec3 = _lazy_ctor("IVec3")
ivec4 = _lazy_ctor("IVec4")
uvec2 = _lazy_ctor("UVec2")
uvec3 = _lazy_ctor("UVec3")
uvec4 = _lazy_ctor("UVec4")
mat2 = _lazy_ctor("Mat2")
mat3 = _lazy_ctor("Mat3")
mat4 = _lazy_ctor("Mat4")


def _builtin_same_type(name: str) -> Callable[..., Any]:
    """A built-in that returns the same type as its first argument."""

    def _proxy(*args: Any) -> Any:
        if not args:
            raise TypeError(f"{name}() requires at least one argument")
        coerced = _coerce_args(args)
        return FunctionCall(name, coerced, coerced[0].glsl_type)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def _builtin_scalar(name: str) -> Callable[..., Any]:
    """A built-in that always returns a scalar Float."""

    def _proxy(*args: Any) -> Any:

        return FunctionCall(name, _coerce_args(args), Float)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def _builtin_bool(name: str) -> Callable[..., Any]:
    """A built-in that always returns Bool."""

    def _proxy(*args: Any) -> Any:

        return FunctionCall(name, _coerce_args(args), Bool)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def _builtin_void(name: str) -> Callable[..., Any]:
    """A built-in that returns no value (void)."""

    def _proxy(*args: Any) -> Any:
        return FunctionCall(name, _coerce_args(args), None)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def _builtin_uint(name: str) -> Callable[..., Any]:
    """A built-in that always returns UInt."""

    def _proxy(*args: Any) -> Any:

        return FunctionCall(name, _coerce_args(args), UInt)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def _builtin_int(name: str) -> Callable[..., Any]:
    """A built-in that always returns Int."""

    def _proxy(*args: Any) -> Any:

        return FunctionCall(name, _coerce_args(args), Int)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


abs = _builtin_same_type("abs")
sign = _builtin_same_type("sign")
floor = _builtin_same_type("floor")
ceil = _builtin_same_type("ceil")
fract = _builtin_same_type("fract")
mod = _builtin_same_type("mod")
min = _builtin_same_type("min")
max = _builtin_same_type("max")
clamp = _builtin_same_type("clamp")
mix = _builtin_same_type("mix")
step = _builtin_same_type("step")
smoothstep = _builtin_same_type("smoothstep")
normalize = _builtin_same_type("normalize")
reflect = _builtin_same_type("reflect")
refract = _builtin_same_type("refract")
sin = _builtin_same_type("sin")
cos = _builtin_same_type("cos")
tan = _builtin_same_type("tan")
asin = _builtin_same_type("asin")
acos = _builtin_same_type("acos")
atan = _builtin_same_type("atan")
pow = _builtin_same_type("pow")
exp = _builtin_same_type("exp")
exp2 = _builtin_same_type("exp2")
log = _builtin_same_type("log")
log2 = _builtin_same_type("log2")
sqrt = _builtin_same_type("sqrt")
inversesqrt = _builtin_same_type("inversesqrt")

dot = _builtin_scalar("dot")
length = _builtin_scalar("length")
distance = _builtin_scalar("distance")
determinant = _builtin_scalar("determinant")

cross = _builtin_same_type("cross")


# Texture sampling — returns vec4.
def texture(sampler: Any, coord: Any) -> Any:
    """``texture(sampler, coord)`` → vec4."""

    return FunctionCall("texture", [sampler, coord], Vec4)


def texelFetch(sampler: Any, coord: Any, lod: Any) -> Any:
    """``texelFetch(sampler, coord, lod)`` → vec4."""

    return FunctionCall("texelFetch", [sampler, coord, lod], Vec4)


transpose = _builtin_same_type("transpose")
inverse = _builtin_same_type("inverse")

any = _builtin_bool("any")
all = _builtin_bool("all")


# Synchronization barriers (void).
barrier = _builtin_void("barrier")
memoryBarrier = _builtin_void("memoryBarrier")
memoryBarrierBuffer = _builtin_void("memoryBarrierBuffer")
memoryBarrierShared = _builtin_void("memoryBarrierShared")
memoryBarrierImage = _builtin_void("memoryBarrierImage")
memoryBarrierAtomicCounter = _builtin_void("memoryBarrierAtomicCounter")
groupMemoryBarrier = _builtin_void("groupMemoryBarrier")

# Atomic operations — return the *previous* value (uint).
atomicAdd = _builtin_uint("atomicAdd")
atomicMin = _builtin_uint("atomicMin")
atomicMax = _builtin_uint("atomicMax")
atomicAnd = _builtin_uint("atomicAnd")
atomicOr = _builtin_uint("atomicOr")
atomicXor = _builtin_uint("atomicXor")
atomicExchange = _builtin_uint("atomicExchange")
atomicCompSwap = _builtin_uint("atomicCompSwap")

# Atomic counter operations.
atomicCounterIncrement = _builtin_uint("atomicCounterIncrement")
atomicCounterDecrement = _builtin_uint("atomicCounterDecrement")
atomicCounter = _builtin_uint("atomicCounter")

# Image operations.
imageSize = _builtin_int("imageSize")
imageStore = _builtin_void("imageStore")


def imageLoad(image: Any, coord: Any) -> Any:
    """``imageLoad(image, coord)`` → vec4."""
    return FunctionCall("imageLoad", [image, coord], Vec4)


def imageAtomicAdd(image: Any, coord: Any, data: Any) -> Any:
    """``imageAtomicAdd(image, coord, data)`` → uint."""

    return FunctionCall("imageAtomicAdd", [image, coord, data], UInt)


def imageAtomicMin(image: Any, coord: Any, data: Any) -> Any:
    """``imageAtomicMin(image, coord, data)`` → uint."""

    return FunctionCall("imageAtomicMin", [image, coord, data], UInt)


def imageAtomicMax(image: Any, coord: Any, data: Any) -> Any:
    """``imageAtomicMax(image, coord, data)`` → uint."""

    return FunctionCall("imageAtomicMax", [image, coord, data], UInt)


def imageAtomicExchange(image: Any, coord: Any, data: Any) -> Any:
    """``imageAtomicExchange(image, coord, data)`` → uint."""

    return FunctionCall("imageAtomicExchange", [image, coord, data], UInt)


def imageAtomicCompSwap(image: Any, coord: Any, compare: Any, data: Any) -> Any:
    """``imageAtomicCompSwap(image, coord, compare, data)`` → uint."""

    return FunctionCall("imageAtomicCompSwap", [image, coord, compare, data], UInt)


sinh = _builtin_same_type("sinh")
cosh = _builtin_same_type("cosh")
tanh = _builtin_same_type("tanh")
asinh = _builtin_same_type("asinh")
acosh = _builtin_same_type("acosh")
atanh = _builtin_same_type("atanh")


round = _builtin_same_type("round")
roundEven = _builtin_same_type("roundEven")
trunc = _builtin_same_type("trunc")
fma = _builtin_same_type("fma")


isnan = _builtin_bool("isnan")
isinf = _builtin_bool("isinf")


floatBitsToInt = _builtin_int("floatBitsToInt")
floatBitsToUint = _builtin_uint("floatBitsToUint")


def intBitsToFloat(*args: Any) -> Any:

    return FunctionCall("intBitsToFloat", _coerce_args(args), Float)


def uintBitsToFloat(*args: Any) -> Any:

    return FunctionCall("uintBitsToFloat", _coerce_args(args), Float)


packUnorm2x16 = _builtin_uint("packUnorm2x16")
packSnorm2x16 = _builtin_uint("packSnorm2x16")
packUnorm4x8 = _builtin_uint("packUnorm4x8")
packSnorm4x8 = _builtin_uint("packSnorm4x8")
packHalf2x16 = _builtin_uint("packHalf2x16")


def unpackUnorm2x16(*args: Any) -> Any:

    return FunctionCall("unpackUnorm2x16", _coerce_args(args), Vec2)


def unpackSnorm2x16(*args: Any) -> Any:

    return FunctionCall("unpackSnorm2x16", _coerce_args(args), Vec2)


def unpackUnorm4x8(*args: Any) -> Any:

    return FunctionCall("unpackUnorm4x8", _coerce_args(args), Vec4)


def unpackSnorm4x8(*args: Any) -> Any:

    return FunctionCall("unpackSnorm4x8", _coerce_args(args), Vec4)


def unpackHalf2x16(*args: Any) -> Any:

    return FunctionCall("unpackHalf2x16", _coerce_args(args), Vec2)


bitfieldExtract = _builtin_same_type("bitfieldExtract")
bitfieldInsert = _builtin_same_type("bitfieldInsert")
bitfieldReverse = _builtin_same_type("bitfieldReverse")
bitCount = _builtin_int("bitCount")
findLSB = _builtin_int("findLSB")
findMSB = _builtin_int("findMSB")


def _builtin_bvec(name: str) -> Callable[..., Any]:
    """A built-in that returns a bvec of same size as first vector arg."""

    def _proxy(*args: Any) -> Any:

        coerced = _coerce_args(args)
        if coerced and hasattr(coerced[0], "glsl_type"):
            if issubclass(coerced[0].glsl_type, VectorType):
                return FunctionCall(
                    name, coerced, _vec_for(Bool, coerced[0].glsl_type.size)
                )
        return FunctionCall(name, coerced, Bool)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


lessThan = _builtin_bvec("lessThan")
lessThanEqual = _builtin_bvec("lessThanEqual")
greaterThan = _builtin_bvec("greaterThan")
greaterThanEqual = _builtin_bvec("greaterThanEqual")
equal = _builtin_bvec("equal")
notEqual = _builtin_bvec("notEqual")


def _builtin_vec4(name: str) -> Callable[..., Any]:
    """A built-in that always returns vec4."""

    def _proxy(*args: Any) -> Any:

        return FunctionCall(name, _coerce_args(args), Vec4)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


def _builtin_ivec(name: str, size: int = 2) -> Callable[..., Any]:
    """A built-in that returns an ivec of given size."""

    def _proxy(*args: Any) -> Any:

        return FunctionCall(name, _coerce_args(args), _vec_for(Int, size))

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


textureSize = _builtin_int("textureSize")
textureQueryLod = _builtin_vec4("textureQueryLod")
textureQueryLevels = _builtin_int("textureQueryLevels")
textureLod = _builtin_vec4("textureLod")
textureOffset = _builtin_vec4("textureOffset")
textureProj = _builtin_vec4("textureProj")
textureProjOffset = _builtin_vec4("textureProjOffset")
textureGrad = _builtin_vec4("textureGrad")
textureGradOffset = _builtin_vec4("textureGradOffset")
textureGather = _builtin_vec4("textureGather")
textureGatherOffset = _builtin_vec4("textureGatherOffset")
textureGatherOffsets = _builtin_vec4("textureGatherOffsets")
textureLodOffset = _builtin_vec4("textureLodOffset")
textureProjLod = _builtin_vec4("textureProjLod")
textureProjLodOffset = _builtin_vec4("textureProjLodOffset")
textureProjGrad = _builtin_vec4("textureProjGrad")
textureProjGradOffset = _builtin_vec4("textureProjGradOffset")


faceforward = _builtin_same_type("faceforward")
outerProduct = _builtin_same_type("outerProduct")
matrixCompMult = _builtin_same_type("matrixCompMult")


dFdx = _builtin_same_type("dFdx")
dFdy = _builtin_same_type("dFdy")
dFdxCoarse = _builtin_same_type("dFdxCoarse")
dFdyCoarse = _builtin_same_type("dFdyCoarse")
dFdxFine = _builtin_same_type("dFdxFine")
dFdyFine = _builtin_same_type("dFdyFine")
fwidth = _builtin_same_type("fwidth")
fwidthCoarse = _builtin_same_type("fwidthCoarse")
fwidthFine = _builtin_same_type("fwidthFine")


EmitVertex = _builtin_void("EmitVertex")
EndPrimitive = _builtin_void("EndPrimitive")


interpolateAtCentroid = _builtin_same_type("interpolateAtCentroid")
interpolateAtSample = _builtin_same_type("interpolateAtSample")
interpolateAtOffset = _builtin_same_type("interpolateAtOffset")


bvec2 = _lazy_ctor("BVec2")
bvec3 = _lazy_ctor("BVec3")
bvec4 = _lazy_ctor("BVec4")
dvec2 = _lazy_ctor("DVec2")
dvec3 = _lazy_ctor("DVec3")
dvec4 = _lazy_ctor("DVec4")
