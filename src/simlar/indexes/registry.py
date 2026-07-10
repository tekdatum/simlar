"""Index factory and registry.

Usage::

    from simlar.indexes.registry import register, build, load_from_directory

    @register("my_index")
    class MyIndex(VectorIndex): ...

    idx = build("my_index", param=value)
    idx = load_from_directory("/path/to/saved/index")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simlar.contracts import Index

_REGISTRY: dict[str, type[Index]] = {}


def register(name: str):
    """Class decorator to register an index type under *name*."""

    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def build(name: str, **kwargs) -> Index:
    """Instantiate a registered index by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown index type {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def load_from_directory(directory: str) -> Index:
    """Load any registered index from its saved directory.

    Reads ``config.json`` → ``index_type`` field → calls the matching ``cls.load()``.
    """
    config_path = Path(directory) / "config.json"
    meta = json.loads(config_path.read_text())
    index_type = meta["index_type"]
    if index_type not in _REGISTRY:
        raise KeyError(
            f"Saved index type {index_type!r} is not in registry. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[index_type].load(directory)
