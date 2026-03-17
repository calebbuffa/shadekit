"""GLSL sampler and image type classes."""

from ._base import ShaderType


class SamplerType(ShaderType):
    """Base for GLSL sampler types (``sampler2D``, ``samplerCube``, etc.)."""


class Sampler1D(SamplerType):
    glsl_name = "sampler1D"


class Sampler2D(SamplerType):
    glsl_name = "sampler2D"


class Sampler3D(SamplerType):
    glsl_name = "sampler3D"


class SamplerCube(SamplerType):
    glsl_name = "samplerCube"


class Sampler2DArray(SamplerType):
    glsl_name = "sampler2DArray"


class Sampler1DArray(SamplerType):
    glsl_name = "sampler1DArray"


class SamplerCubeArray(SamplerType):
    glsl_name = "samplerCubeArray"


class SamplerBuffer(SamplerType):
    glsl_name = "samplerBuffer"


class Sampler2DMS(SamplerType):
    glsl_name = "sampler2DMS"


class Sampler2DMSArray(SamplerType):
    glsl_name = "sampler2DMSArray"


class Sampler2DRect(SamplerType):
    glsl_name = "sampler2DRect"


class ISampler1D(SamplerType):
    glsl_name = "isampler1D"


class ISampler2D(SamplerType):
    glsl_name = "isampler2D"


class ISampler3D(SamplerType):
    glsl_name = "isampler3D"


class ISamplerCube(SamplerType):
    glsl_name = "isamplerCube"


class ISampler2DArray(SamplerType):
    glsl_name = "isampler2DArray"


class ISampler1DArray(SamplerType):
    glsl_name = "isampler1DArray"


class ISamplerCubeArray(SamplerType):
    glsl_name = "isamplerCubeArray"


class ISamplerBuffer(SamplerType):
    glsl_name = "isamplerBuffer"


class ISampler2DMS(SamplerType):
    glsl_name = "isampler2DMS"


class USampler1D(SamplerType):
    glsl_name = "usampler1D"


class USampler2D(SamplerType):
    glsl_name = "usampler2D"


class USampler3D(SamplerType):
    glsl_name = "usampler3D"


class USamplerCube(SamplerType):
    glsl_name = "usamplerCube"


class USampler2DArray(SamplerType):
    glsl_name = "usampler2DArray"


class USampler1DArray(SamplerType):
    glsl_name = "usampler1DArray"


class USamplerCubeArray(SamplerType):
    glsl_name = "usamplerCubeArray"


class USamplerBuffer(SamplerType):
    glsl_name = "usamplerBuffer"


class USampler2DMS(SamplerType):
    glsl_name = "usampler2DMS"


class Sampler1DShadow(SamplerType):
    glsl_name = "sampler1DShadow"


class Sampler2DShadow(SamplerType):
    glsl_name = "sampler2DShadow"


class SamplerCubeShadow(SamplerType):
    glsl_name = "samplerCubeShadow"


class Sampler1DArrayShadow(SamplerType):
    glsl_name = "sampler1DArrayShadow"


class Sampler2DArrayShadow(SamplerType):
    glsl_name = "sampler2DArrayShadow"


class SamplerCubeArrayShadow(SamplerType):
    glsl_name = "samplerCubeArrayShadow"


class Sampler2DRectShadow(SamplerType):
    glsl_name = "sampler2DRectShadow"


class ImageType(ShaderType):
    """Base for GLSL image types (``image2D``, ``uimage3D``, etc.)."""


class Image1D(ImageType):
    glsl_name = "image1D"


class Image2D(ImageType):
    glsl_name = "image2D"


class Image3D(ImageType):
    glsl_name = "image3D"


class ImageCube(ImageType):
    glsl_name = "imageCube"


class Image2DArray(ImageType):
    glsl_name = "image2DArray"


class Image1DArray(ImageType):
    glsl_name = "image1DArray"


class ImageCubeArray(ImageType):
    glsl_name = "imageCubeArray"


class ImageBuffer(ImageType):
    glsl_name = "imageBuffer"


class Image2DMS(ImageType):
    glsl_name = "image2DMS"


class IImage1D(ImageType):
    glsl_name = "iimage1D"


class IImage2D(ImageType):
    glsl_name = "iimage2D"


class IImage3D(ImageType):
    glsl_name = "iimage3D"


class IImageCube(ImageType):
    glsl_name = "iimageCube"


class IImage2DArray(ImageType):
    glsl_name = "iimage2DArray"


class UImage1D(ImageType):
    glsl_name = "uimage1D"


class UImage2D(ImageType):
    glsl_name = "uimage2D"


class UImage3D(ImageType):
    glsl_name = "uimage3D"


class UImageCube(ImageType):
    glsl_name = "uimageCube"


class UImage2DArray(ImageType):
    glsl_name = "uimage2DArray"


class AtomicUint(ShaderType):
    """GLSL ``atomic_uint`` type."""

    glsl_name = "atomic_uint"


class Void(ShaderType):
    """GLSL ``void`` type — for function return types."""

    glsl_name = "void"
