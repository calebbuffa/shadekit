"""Dependency graph for GLSL function/struct ordering.

Resolves topological order so that callees are emitted before callers.
Detects circular dependencies.

Usage::

    from shadekit.compiler import DependencyGraph

    graph = DependencyGraph()
    graph.add(luminance_fn)
    graph.add(shade_fn)       # calls luminance
    ordered = graph.resolve() # [luminance_fn, shade_fn]
"""

from __future__ import annotations

from shadekit.compiler._ast_walk import find_called_names
from shadekit.decorators._function import ShaderFunction


class CircularDependencyError(Exception):
    """Raised when shader functions form a circular dependency."""


class DependencyGraph:
    """Collects :class:`ShaderFunction` instances and resolves emit order."""

    __slots__ = ("_functions", "_by_name")

    def __init__(self) -> None:
        self._functions: list[ShaderFunction] = []
        self._by_name: dict[str, ShaderFunction] = {}

    def add(self, fn: ShaderFunction) -> None:
        """Register a function in the graph."""
        if fn.name not in self._by_name:
            self._functions.append(fn)
            self._by_name[fn.name] = fn

    def resolve(self) -> list[ShaderFunction]:
        """Return functions in topological order (leaves first).

        Raises :class:`CircularDependencyError` if a cycle is detected.
        """
        # Build adjacency: fn → set of GlslFunctions it calls.
        deps: dict[str, set[str]] = {}
        for fn in self._functions:
            called = _find_called_names(fn)
            # Only keep references to functions we know about.
            deps[fn.name] = called & set(self._by_name)

        # Kahn's algorithm (topological sort).
        in_degree: dict[str, int] = {name: 0 for name in deps}
        for name, callees in deps.items():
            for callee in callees:
                in_degree[callee] = in_degree.get(callee, 0)  # ensure exists

        # Reverse: callee comes first → edges go caller → callee
        # in_degree counts how many callers point at you? No — standard topo:
        # We want leaves (no dependencies) first.
        # edge: caller → callee means callee must come first.
        # So in_degree of a node = number of nodes that depend on it? No.
        # Standard: in_degree = number of prerequisites.
        # For us: prerequisite of caller = callee.
        # edge direction: caller → callee (caller depends on callee).
        # in_degree of X = how many things X depends on = len(deps[X]).
        in_degree = {name: len(callees) for name, callees in deps.items()}

        # reverse adjacency: callee → list of callers
        rev: dict[str, list[str]] = {name: [] for name in deps}
        for name, callees in deps.items():
            for callee in callees:
                rev[callee].append(name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            name = queue.pop(0)
            order.append(name)
            for caller in rev.get(name, []):
                in_degree[caller] -= 1
                if in_degree[caller] == 0:
                    queue.append(caller)

        if len(order) != len(deps):
            remaining = set(deps) - set(order)
            raise CircularDependencyError(
                f"Circular dependency among: {', '.join(sorted(remaining))}"
            )

        return [self._by_name[name] for name in order]


def _find_called_names(fn: ShaderFunction) -> set[str]:
    """Walk the function body AST and return all called function names."""
    return find_called_names(fn.body)
