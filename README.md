# ombra

A Python library for composing GLSL shaders programmatically. Instead of writing shader source as raw strings, you describe them using Python — with a type system that maps directly to GLSL, an AST built through operator overloading, and a high-level `Program` API that handles stage assembly.

It targets GLSL 4.30 core by default and has no runtime dependencies. Python 3.12+ required.

---

## Quick start

```python
from ombra.glsl import Program
from ombra.types import Vec3, Vec4, Mat4, Sampler2D, Float
from ombra.glsl import vec4, texture, dot, normalize
from ombra.decorators import shader_function

# A reusable helper defined once, referenced in any stage
@shader_function
def luminance(c: Vec3) -> Float:
    return dot(c, vec4(0.299, 0.587, 0.114, 0.0).xyz)

prog = Program()

u_mvp   = prog.uniform("u_mvp",   Mat4,      stage="vertex")
u_tex   = prog.uniform("u_albedo", Sampler2D, stage="fragment", binding=0)
pos     = prog.vertex_input(0, "Pos", Vec3)
v_uv    = prog.varying("v_uv", Vec3)
f_color = prog.output(0, "f_color", Vec4)

@prog.vertex
def vs():
    return [
        Assignment(v_uv,          pos),
        Assignment("gl_Position", u_mvp * vec4(pos, 1.0)),
    ]

@prog.fragment
def fs():
    sample = texture(u_tex, v_uv)
    lum    = luminance(sample.xyz)
    return [
        Assignment(f_color, vec4(sample.xyz * lum, 1.0)),
        Return(Void),
    ]

vert_src, frag_src = prog.build()
```

`prog.build()` returns a pair of `Stage` objects. They're just strings with a `.kind` attribute and a `.save(path)` convenience method, so you can pass them directly to your OpenGL bindings.

---

## Types

Every GLSL type is a Python class. They're not meant to be instantiated — they're used as type annotations and carry metadata that drives code generation.

**Scalars**: `Float`, `Double`, `Int`, `UInt`, `Bool`

**Vectors**: `Vec2/3/4`, `DVec2/3/4`, `IVec2/3/4`, `UVec2/3/4`, `BVec2/3/4`

**Matrices**: `Mat2`, `Mat3`, `Mat4`

**Samplers and images**: the full set from GLSL 4.30 — `Sampler2D`, `Sampler3D`, `SamplerCube`, shadow samplers, integer and unsigned variants, image types (`Image2D`, `IImage2D`, etc.), and `AtomicUint`.

**Arrays**:

```python
from ombra.types import ArrayType, Float
FloatArray256 = ArrayType(Float, 256)  # → "float[256]"
```

**Structs**:

```python
from dataclasses import dataclass
from ombra.types import Vec3, Float
from ombra.decorators import shader_struct

@shader_struct
@dataclass
class Material:
    albedo: Vec3
    roughness: Float

# Material.__glsl_declaration__ → "struct Material { vec3 albedo; float roughness; };"
```

---

## Shader functions

`@shader_function` turns a Python function into a reusable GLSL function. The body runs at decoration time with `Variable` placeholder arguments, and the result is captured as an AST.

```python
from ombra.decorators import shader_function
from ombra.types import Vec3, Float
from ombra.glsl import dot, normalize, clamp

@shader_function
def diffuse(normal: Vec3, light_dir: Vec3) -> Float:
    n = normalize(normal)
    l = normalize(light_dir)
    return clamp(dot(n, l), 0.0, 1.0)

diffuse.signature()
# → "float diffuse(vec3 normal, vec3 light_dir)"
```

Calling the decorated function produces a `FunctionCall` AST node, not a Python value. Dependencies between shader functions are detected automatically — if `diffuse` calls another `@shader_function`, `Program.build()` will include it and sort them in the right order.

---

## AST

You generally don't need to build AST nodes by hand — the builtins and constructors do it for you. But when you need to, the full node set is available.

```python
from ombra.ast import Variable, Literal, BinaryOp, If, For, Assignment, Declaration

x = Variable("x", Float)
y = Variable("y", Float)

# Operator overloading builds the tree
expr = x * y + Literal(1.0, Float)   # BinaryOp(+, BinaryOp(*, x, y), Literal(1.0))

# Equality uses .eq() / .ne() — not == / != — because those have Python semantics
check = x.eq(y)   # BinaryOp(==, x, y)

# Control flow
branch = If(
    condition=x.gt(Literal(0.0, Float)),
    body=[Assignment(y, x)],
    else_body=[Assignment(y, -x)],
)
```

Supported statement types: `Declaration`, `Assignment`, `CompoundAssignment`, `Return`, `Discard`, `If`, `For`, `While`, `DoWhile`, `Switch`, `Break`, `Continue`, `ExpressionStatement`.

---

## Built-ins

All standard GLSL built-in functions are available from `ombra.glsl`:

```python
from ombra.glsl import (
    normalize, dot, cross, reflect, refract,
    texture, texelFetch,
    mix, clamp, smoothstep, step,
    sin, cos, atan, pow, sqrt, log2,
    abs, floor, ceil, fract, mod, sign,
    transpose, inverse, determinant,
    atomicAdd, barrier, memoryBarrier,
    dFdx, dFdy,
    # ... and many more
)
```

Constructors:

```python
from ombra.glsl import vec2, vec3, vec4, mat4, ivec2, uvec4
```

---

## Program API

`Program` is the main entry point for full shader programs. It tracks uniforms, inputs, outputs, varyings, SSBOs, structs, and compute resources across stages, then assembles everything into valid GLSL.

```python
prog = Program(version="430", profile="core")

# Extensions and defines
prog.extension("GL_ARB_gpu_shader5", "enable")
prog.define("MAX_LIGHTS", 8)
prog.define("HAS_NORMALS", stage="vertex")

# Vertex inputs
pos    = prog.vertex_input(0, "Pos",    Vec3)
normal = prog.vertex_input(1, "Normal", Vec3)

# Uniforms
u_mvp = prog.uniform("u_mvp",  Mat4,      stage="vertex")
u_tex = prog.uniform("u_tex",  Sampler2D, stage="fragment", binding=0)

# Varyings (between vertex and fragment)
v_normal  = prog.varying("v_normal", Vec3)
v_id      = prog.varying("v_id",     Int,  flat=True)
v_uv      = prog.varying("v_uv",     Vec2, noperspective=True)

# Fragment output
f_color = prog.output(0, "f_color", Vec4)

# SSBO
prog.storage(0, "Positions", "vec4 data[];", readonly=True, stage="vertex")

# Struct
prog.struct(Material.__glsl_type__, stage="fragment")

vert_src, frag_src = prog.build()
```

For compute shaders:

```python
shared_data = prog.shared("s_data", ArrayType(Float, 256))
prog.local_size(256)          # 1D
# prog.local_size(16, 16)     # 2D
compute_src = prog.build_compute()
```

You can also load existing GLSL and layer defines on top of it:

```python
prog = Program.from_glsl(existing_vert_src, existing_frag_src)
prog.define("ENABLE_SHADOWS")
vert, frag = prog.build()
```

---

## Compiler utilities

A few lower-level tools are exposed if you need them directly:

```python
from ombra.compiler import (
    fold_constants,            # constant folding pass over an AST
    eliminate_dead_functions,  # dead code elimination
    DependencyGraph,           # topological sort of @shader_function deps
    ShaderCache,               # hash-based shader source cache
    collect_transitive_deps,   # walk all transitive ShaderFunction dependencies
)
```

These run automatically inside `Program.build()`. You'd use them directly only if you're working at the `ShaderBuilder` level or writing your own pipeline.

---

## ShaderBuilder

`ShaderBuilder` is the lower-level API that `Program` wraps. Use it if you need direct control over what goes into each stage.

```python
from ombra.glsl import ShaderBuilder, ShaderStage

b = ShaderBuilder(version="430", profile="core")
b.add_define("FOO", stage=ShaderStage.VERTEX)
b.add_uniform("mat4", "u_mvp", ShaderStage.VERTEX)
b.add_varying("vec3", "v_normal")
b.add_vertex_lines(["void main() {", "  gl_Position = vec4(0.0);", "}"])

vert_src, frag_src = b.build()
```

`ShaderStage` is an `IntFlag`: `VERTEX`, `FRAGMENT`, `BOTH`, `COMPUTE`.

---
