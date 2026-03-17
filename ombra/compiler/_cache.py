"""Shader caching — content-addressable deduplication of generated GLSL.

Computes a hash of the final GLSL source and returns a cache hit when
the same shader has been generated before, avoiding redundant GPU
recompilation in the host application.

Usage::

    from ombra.compiler import ShaderCache

    cache = ShaderCache()
    key, (vert, frag) = cache.get_or_build(builder)
    # Returns same key if builder produces identical output.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ombra.glsl._builder import ShaderBuilder


class ShaderCache:
    """Content-addressed cache for compiled shader source strings.

    The cache key is a SHA-256 digest of the concatenated source(s).
    Supports both raster (vertex + fragment) and compute shaders.
    """

    __slots__ = ("_cache", "_compute_cache")

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, str]] = {}
        self._compute_cache: dict[str, str] = {}

    def get_or_build(self, builder: ShaderBuilder) -> tuple[str, tuple[str, str]]:
        """Return ``(cache_key, (vert_src, frag_src))``.

        On a cache miss the builder's :meth:`build` is called and the
        result stored.  On a cache hit the stored sources are returned.
        """
        vert, frag = builder.build()
        key = _hash_sources(vert, frag)
        if key not in self._cache:
            self._cache[key] = (vert, frag)
        return key, self._cache[key]

    def get_or_build_compute(self, builder: ShaderBuilder) -> tuple[str, str]:
        """Return ``(cache_key, compute_src)``.

        On a cache miss the builder's :meth:`build_compute` is called
        and the result stored.
        """
        source = builder.build_compute()
        key = _hash_sources(source)
        if key not in self._compute_cache:
            self._compute_cache[key] = source
        return key, self._compute_cache[key]

    def contains(self, key: str) -> bool:
        """Check whether *key* is in the cache."""
        return key in self._cache or key in self._compute_cache

    def invalidate(self, key: str) -> None:
        """Remove a specific entry from the cache."""
        self._cache.pop(key, None)
        self._compute_cache.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._compute_cache.clear()

    def __len__(self) -> int:
        return len(self._cache) + len(self._compute_cache)


def hash_sources(*sources: str) -> str:
    """Return a hex-digest cache key for the given shader source(s).

    Accepts one or more GLSL source strings.  Each source is separated by
    a null byte in the hash to prevent collisions between concatenated
    strings.
    """
    return _hash_sources(*sources)


def _hash_sources(*sources: str) -> str:
    h = hashlib.sha256()
    for i, src in enumerate(sources):
        if i:
            h.update(b"\x00")
        h.update(src.encode("utf-8"))
    return h.hexdigest()
