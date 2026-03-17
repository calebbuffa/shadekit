"""Dead-code elimination for ShaderFunction collections.

Removes :class:`ShaderFunction` instances not reachable from a set of
entry-point function names.  Also detects unused uniforms in a
:class:`ShaderBuilder` context.

Usage::

    from ombra.compiler import eliminate_dead_functions
    live = eliminate_dead_functions(all_fns, entry_names={"main"})
"""

from __future__ import annotations

import re

from ombra.ast._expressions import FunctionCall, Variable
from ombra.compiler._ast_walk import find_called_names, walk_stmt
from ombra.decorators._function import ShaderFunction


def eliminate_dead_functions(
    functions: list[ShaderFunction],
    entry_names: set[str],
) -> list[ShaderFunction]:
    """Return only functions reachable (transitively) from *entry_names*.

    Preserves original ordering among the surviving functions.
    """
    by_name: dict[str, ShaderFunction] = {fn.name: fn for fn in functions}
    reachable: set[str] = set()

    # BFS from entry points.
    frontier = list(entry_names & set(by_name))
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        fn = by_name.get(name)
        if fn is None:
            continue
        called = find_called_names(fn.body) & set(by_name)
        frontier.extend(called - reachable)

    return [fn for fn in functions if fn.name in reachable]


def find_referenced_names(
    functions: list[ShaderFunction], lines: list[str]
) -> set[str]:
    """Collect all identifiers referenced in function bodies and raw GLSL lines.

    Useful for detecting unused uniforms: a uniform whose name does not
    appear in this set can safely be dropped.
    """
    names: set[str] = set()

    def _visitor(expr: object) -> None:
        if isinstance(expr, Variable):
            names.add(expr.name)
        elif isinstance(expr, FunctionCall):
            names.add(expr.func_name)

    for fn in functions:
        for stmt in fn.body:
            walk_stmt(stmt, _visitor)  # type: ignore[arg-type]

    _IDENT = re.compile(r"\b([A-Za-z_]\w*)\b")
    for line in lines:
        names.update(_IDENT.findall(line))

    return names
