"""Shader compiler — language-agnostic optimization, caching, and graph utilities.

GLSL-specific emission, assembly, and validation live in :mod:`ombra.glsl`.

Public API::

    from ombra.compiler import DependencyGraph, CircularDependencyError
    from ombra.compiler import walk_expr, walk_stmt, find_called_names
    from ombra.compiler import fold_constants, fold_expr, fold_stmt
    from ombra.compiler import eliminate_dead_functions, find_referenced_names
    from ombra.compiler import ShaderCache, hash_sources
"""

from ombra.compiler._ast_walk import (
    collect_shader_functions,
    collect_transitive_deps,
    find_called_names,
    walk_expr,
    walk_stmt,
)
from ombra.compiler._cache import ShaderCache, hash_sources
from ombra.compiler._dce import eliminate_dead_functions, find_referenced_names
from ombra.compiler._dependency_graph import CircularDependencyError, DependencyGraph
from ombra.compiler._optimizer import fold_constants, fold_expr, fold_stmt

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
