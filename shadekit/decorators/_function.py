"""``@shader_function`` decorator — capture Python functions as shader AST.

Usage::

    from shadekit.decorators import shader_function
    from shadekit.types import Vec3, Float

    @shader_function
    def luminance(c: Vec3) -> Float:
        return dot(c, vec3(0.299, 0.587, 0.114))

    # luminance is now a ShaderFunction:
    # - luminance.definition  →  the full shader function definition (AST)
    # - luminance(some_expr)  →  FunctionCall AST node
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from shadekit.ast._expressions import Expr, FunctionCall, Variable
from shadekit.ast._statements import Return, Stmt
from shadekit.types._scalars import Float

if TYPE_CHECKING:
    from shadekit.types._base import ShaderMeta


class ShaderFunction:
    """A captured shader function definition.

    Calling an instance creates a :class:`FunctionCall` AST node.
    The ``body`` attribute holds the captured statements.

    Dependencies between ``ShaderFunction`` instances are resolved at
    emit time by :class:`~shadekit.compiler.DependencyGraph`, not stored
    on the function itself.
    """

    __slots__ = ("name", "params", "return_type", "body")

    def __init__(
        self,
        name: str,
        params: list[tuple[str, ShaderMeta]],
        return_type: ShaderMeta | None,
        body: list[Stmt],
    ) -> None:
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body

    def __call__(self, *args: Any) -> Any:
        """Create a ``FunctionCall`` AST node for this function."""
        if len(args) != len(self.params):
            raise TypeError(
                f"{self.name}() takes {len(self.params)} argument(s), got {len(args)}"
            )
        if self.return_type is None:
            ret = Float  # void functions still need a type for AST; callers use ExpressionStatement
        else:
            ret = self.return_type
        node = FunctionCall(self.name, list(args), ret)
        node._glsl_fn = self
        return node

    def signature(self) -> str:
        """Return the shader function signature (e.g., ``float luminance(vec3 c)``)."""
        ret = "void" if self.return_type is None else self.return_type.glsl_name
        params_str = ", ".join(
            f"{ptype.glsl_name} {pname}" for pname, ptype in self.params
        )
        return f"{ret} {self.name}({params_str})"


def shader_function(fn: Callable[..., Any]) -> ShaderFunction:
    """Decorator that captures a Python function as a shader function AST.

    The decorated function's body is executed once at decoration time
    with placeholder :class:`Variable` nodes as arguments.  The function
    must ``return`` an :class:`Expr`; this is captured as the function body.

    Type annotations on parameters must be shader type classes
    (``Float``, ``Vec3``, etc.).  The return annotation is used as the
    return type.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    # Extract parameter types.
    params: list[tuple[str, ShaderMeta]] = []
    placeholders: list[Variable] = []
    for pname, param in sig.parameters.items():
        if pname not in hints:
            raise TypeError(
                f"@shader_function: parameter '{pname}' of '{fn.__name__}' "
                f"must have a shader type annotation"
            )
        ptype = hints[pname]
        params.append((pname, ptype))
        placeholders.append(Variable(pname, ptype))

    # Return type.
    return_type: ShaderMeta | None = hints.get("return")
    if return_type is type(None):
        return_type = None

    # Execute the function body with placeholder variables.
    result = fn(*placeholders)

    # Capture the result as a Return statement.
    body: list[Stmt] = []
    if result is not None:
        if isinstance(result, Expr):
            body.append(Return(result))
        else:
            raise TypeError(
                f"@shader_function: '{fn.__name__}' must return an Expr or None, "
                f"got {type(result).__name__}"
            )

    return ShaderFunction(fn.__name__, params, return_type, body)
