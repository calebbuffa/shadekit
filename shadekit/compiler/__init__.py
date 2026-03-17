"""Shader compiler — language-agnostic optimization, caching, and graph utilities.

GLSL-specific emission, assembly, and validation live in :mod:`shadekit.glsl`.

Public API::

    from shadekit.compiler import DependencyGraph, CircularDependencyError
    from shadekit.compiler import walk_expr, walk_stmt, find_called_names
    from shadekit.compiler import fold_constants, fold_expr, fold_stmt
    from shadekit.compiler import eliminate_dead_functions, find_referenced_names
    from shadekit.compiler import ShaderCache, hash_sources
"""

from shadekit.compiler._ast_walk import (
    collect_shader_functions,
    collect_transitive_deps,
    find_called_names,
    walk_expr,
    walk_stmt,
)
from shadekit.compiler._cache import ShaderCache, hash_sources
from shadekit.compiler._dce import eliminate_dead_functions, find_referenced_names
from shadekit.compiler._dependency_graph import CircularDependencyError, DependencyGraph
from shadekit.compiler._optimizer import fold_constants, fold_expr, fold_stmt

__all__ = [
    "DependencyGraph",
    "CircularDependencyError",
    "walk_expr",
    "walk_stmt",
    "find_called_names",
    "collect_shader_functions",
    "collect_transitive_deps",
    "fold_constants",
    "fold_expr",
    "fold_stmt",
    "eliminate_dead_functions",
    "find_referenced_names",
    "ShaderCache",
    "hash_sources",
]
