"""Tests for shadekit.types — GLSL type system (Phase 2)."""

from dataclasses import dataclass

import pytest

from shadekit.types import (
    ArrayType,
    Bool,
    BVec2,
    BVec3,
    Double,
    DVec3,
    Float,
    Int,
    IVec3,
    IVec4,
    Mat2,
    Mat3,
    Mat4,
    Sampler2D,
    ScalarType,
    ShaderMeta,
    StructType,
    UInt,
    UVec3,
    Vec2,
    Vec3,
    Vec4,
    VectorType,
    infer_binary_type,
    shader_struct,
    validate_constructor,
)

# ── GlslType basics ──────────────────────────────────────────────────


class TestGlslType:
    def test_glsl_name(self):
        assert Float.glsl_name == "float"
        assert Vec3.glsl_name == "vec3"
        assert Mat4.glsl_name == "mat4"
        assert Sampler2D.glsl_name == "sampler2D"

    def test_str_returns_glsl_name(self):
        assert str(Float) == "float"
        assert str(Vec4) == "vec4"

    def test_repr_returns_glsl_name(self):
        assert repr(Mat3) == "mat3"

    def test_equality_by_glsl_name(self):
        # Dynamically created classes with same glsl_name compare equal
        DynFloat = ShaderMeta("float", (ScalarType,), {"glsl_name": "float"})
        assert Float == DynFloat
        DynVec3 = ShaderMeta(
            "vec3",
            (VectorType,),
            {
                "glsl_name": "vec3",
                "component_type": Float,
                "size": 3,
            },
        )
        assert Vec3 == DynVec3

    def test_hash_consistent(self):
        DynFloat = ShaderMeta("float", (ScalarType,), {"glsl_name": "float"})
        assert hash(Float) == hash(DynFloat)
        s = {Float, Vec3, Mat4}
        assert Float in s
        assert Vec3 in s

    def test_identity_singletons(self):
        assert Float is Float
        assert Vec3 is Vec3


# ── Scalar types ─────────────────────────────────────────────────────


class TestScalars:
    def test_all_scalars(self):
        for t, name in [
            (Float, "float"),
            (Double, "double"),
            (Int, "int"),
            (UInt, "uint"),
            (Bool, "bool"),
        ]:
            assert issubclass(t, ScalarType)
            assert t.glsl_name == name

    def test_metaclass(self):
        assert isinstance(Float, ShaderMeta)
        assert isinstance(Int, ShaderMeta)


# ── Vector types ─────────────────────────────────────────────────────


class TestVectors:
    def test_float_vectors(self):
        assert Vec2.component_type is Float
        assert Vec2.size == 2
        assert Vec3.size == 3
        assert Vec4.size == 4

    def test_int_vectors(self):
        assert IVec3.component_type is Int
        assert IVec3.size == 3

    def test_uint_vectors(self):
        assert UVec3.component_type is UInt

    def test_double_vectors(self):
        assert DVec3.component_type is Double

    def test_bool_vectors(self):
        assert BVec2.component_type is Bool
        assert BVec2.size == 2


# ── Matrix types ─────────────────────────────────────────────────────


class TestMatrices:
    def test_dimensions(self):
        assert Mat2.cols == 2 and Mat2.rows == 2
        assert Mat3.cols == 3 and Mat3.rows == 3
        assert Mat4.cols == 4 and Mat4.rows == 4


# ── Array types ──────────────────────────────────────────────────────


class TestArrayType:
    def test_basic(self):
        a = ArrayType(Vec3, 10)
        assert a.glsl_name == "vec3[10]"
        assert a.base is Vec3
        assert a.size == 10

    def test_equality(self):
        a1 = ArrayType(Float, 5)
        a2 = ArrayType(Float, 5)
        assert a1 == a2
        assert a1 is not a2

    def test_nested_array(self):
        inner = ArrayType(Float, 4)
        outer = ArrayType(inner, 3)
        assert outer.glsl_name == "float[4][3]"


# ── Struct types ─────────────────────────────────────────────────────


class TestStructType:
    def test_declaration(self):
        st = StructType("Material", {"albedo": Vec3, "roughness": Float})
        expected = "struct Material {\n    vec3 albedo;\n    float roughness;\n};"
        assert st.declaration() == expected

    def test_glsl_name(self):
        st = StructType("Light", {"position": Vec3})
        assert st.glsl_name == "Light"


class TestGlslStruct:
    def test_basic_decorator(self):
        @shader_struct
        @dataclass
        class PBR:
            albedo: Vec3
            roughness: Float
            metallic: Float

        assert hasattr(PBR, "__glsl_type__")
        assert hasattr(PBR, "__glsl_declaration__")
        assert isinstance(PBR.__glsl_type__, StructType)
        assert "vec3 albedo;" in PBR.__glsl_declaration__
        assert "float roughness;" in PBR.__glsl_declaration__
        assert "float metallic;" in PBR.__glsl_declaration__
        assert PBR.__glsl_declaration__.startswith("struct PBR {")

    def test_rejects_non_dataclass(self):
        with pytest.raises(TypeError, match="requires a dataclass"):

            @shader_struct
            class NotADataclass:
                x: Float

    def test_rejects_non_glsl_type_field(self):
        with pytest.raises(TypeError, match="expected a ShaderType"):

            @shader_struct
            @dataclass
            class Bad:
                x: int

    def test_field_order_preserved(self):
        @shader_struct
        @dataclass
        class Ordered:
            a: Float
            b: Vec3
            c: Mat4

        fields = list(Ordered.__glsl_type__.fields.keys())
        assert fields == ["a", "b", "c"]

    def test_struct_still_works_as_dataclass(self):
        @shader_struct
        @dataclass
        class V:
            position: Vec3
            normal: Vec3

        # Can still instantiate as a Python dataclass.
        v = V(position=(0, 0, 0), normal=(1, 0, 0))
        assert v.position == (0, 0, 0)


# ── Type inference ───────────────────────────────────────────────────


class TestInferBinaryType:
    # Arithmetic: scalar ⊕ scalar
    def test_float_plus_float(self):
        assert infer_binary_type("+", Float, Float) is Float

    def test_int_times_float_promotes(self):
        assert infer_binary_type("*", Int, Float) is Float

    def test_int_plus_uint_promotes(self):
        assert infer_binary_type("+", Int, UInt) is UInt

    # Arithmetic: vector ⊕ scalar
    def test_vec3_times_float(self):
        assert infer_binary_type("*", Vec3, Float) is Vec3

    def test_float_times_vec3(self):
        assert infer_binary_type("*", Float, Vec3) is Vec3

    # Arithmetic: vector ⊕ vector
    def test_vec4_plus_vec4(self):
        assert infer_binary_type("+", Vec4, Vec4) is Vec4

    def test_vec3_plus_vec2_fails(self):
        with pytest.raises(TypeError, match="sizes must match"):
            infer_binary_type("+", Vec3, Vec2)

    # Arithmetic: vector promotion
    def test_ivec3_plus_float_promotes(self):
        result = infer_binary_type("+", IVec3, Float)
        assert result is Vec3

    # Arithmetic: matrix * vector
    def test_mat4_times_vec4(self):
        assert infer_binary_type("*", Mat4, Vec4) is Vec4

    def test_mat3_times_vec3(self):
        assert infer_binary_type("*", Mat3, Vec3) is Vec3

    def test_mat4_times_vec3_fails(self):
        with pytest.raises(TypeError, match="dimension mismatch"):
            infer_binary_type("*", Mat4, Vec3)

    # Arithmetic: vector * matrix
    def test_vec4_times_mat4(self):
        assert infer_binary_type("*", Vec4, Mat4) is Vec4

    # Arithmetic: matrix ⊕ matrix
    def test_mat4_times_mat4(self):
        assert infer_binary_type("*", Mat4, Mat4) is Mat4

    def test_mat4_plus_mat4(self):
        assert infer_binary_type("+", Mat4, Mat4) is Mat4

    def test_mat3_plus_mat4_fails(self):
        with pytest.raises(TypeError, match="sizes must match"):
            infer_binary_type("+", Mat3, Mat4)

    # Arithmetic: matrix ⊕ scalar
    def test_mat4_times_float(self):
        assert infer_binary_type("*", Mat4, Float) is Mat4

    def test_float_times_mat4(self):
        assert infer_binary_type("*", Float, Mat4) is Mat4

    # Comparison
    def test_scalar_compare_returns_bool(self):
        assert infer_binary_type("<", Float, Float) is Bool
        assert infer_binary_type("==", Int, Int) is Bool

    def test_vec3_compare_returns_bvec3(self):
        assert infer_binary_type("==", Vec3, Vec3) is BVec3

    def test_vec_compare_size_mismatch(self):
        with pytest.raises(TypeError):
            infer_binary_type("==", Vec2, Vec3)

    # Logical
    def test_logical_and(self):
        assert infer_binary_type("&&", Bool, Bool) is Bool

    def test_logical_rejects_non_bool(self):
        with pytest.raises(TypeError, match="requires bool"):
            infer_binary_type("&&", Int, Bool)

    # Unknown operator
    def test_unknown_op(self):
        with pytest.raises(TypeError, match="Unknown operator"):
            infer_binary_type("@", Float, Float)


# ── Constructor validation ───────────────────────────────────────────


class TestValidateConstructor:
    # vec4 constructors
    def test_vec4_from_float(self):
        assert validate_constructor(Vec4, [Float]) is True

    def test_vec4_from_4_floats(self):
        assert validate_constructor(Vec4, [Float, Float, Float, Float]) is True

    def test_vec4_from_vec3_float(self):
        assert validate_constructor(Vec4, [Vec3, Float]) is True

    def test_vec4_from_vec2_vec2(self):
        assert validate_constructor(Vec4, [Vec2, Vec2]) is True

    def test_vec4_from_vec2_float_float(self):
        assert validate_constructor(Vec4, [Vec2, Float, Float]) is True

    def test_vec4_from_vec3_fails(self):
        # 3 components != 4
        assert validate_constructor(Vec4, [Vec3]) is False

    def test_vec4_from_mat4_fails(self):
        assert validate_constructor(Vec4, [Mat4]) is False

    def test_vec4_empty_args_fails(self):
        assert validate_constructor(Vec4, []) is False

    # mat4 constructors
    def test_mat4_from_float(self):
        assert validate_constructor(Mat4, [Float]) is True

    def test_mat4_from_4_vec4(self):
        assert validate_constructor(Mat4, [Vec4, Vec4, Vec4, Vec4]) is True

    def test_mat4_from_3_vec4_fails(self):
        assert validate_constructor(Mat4, [Vec4, Vec4, Vec4]) is False

    def test_mat3_from_3_vec3(self):
        assert validate_constructor(Mat3, [Vec3, Vec3, Vec3]) is True

    # IVec
    def test_ivec4_from_int(self):
        assert validate_constructor(IVec4, [Int]) is True
