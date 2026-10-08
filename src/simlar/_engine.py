"""Which index cores the installed simlar-engine build provides.

simlar-engine ships in more than one implementation (Cython, C), and a build doesn't
necessarily include every index core. Each index wrapper imports its core through
`engine_core()`: if the core is missing, the wrapper module still imports - so `import simlar`
works on any engine build - and constructing or loading that index raises
`IndexUnavailableError` naming what's missing.
"""

from __future__ import annotations

from importlib import import_module

_UNAVAILABLE: dict[str, str] = {}  # index type -> the engine symbol it's missing, and why


class IndexUnavailableError(ImportError):
    """The installed simlar-engine build doesn't provide this index's core."""


def engine_core(index_type: str, module: str, name: str):
    """`module.name` from the engine, or None (recorded against `index_type`) if this build lacks it."""
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        # several symbols can back one index type (e.g. filtering); report the first missing
        _UNAVAILABLE.setdefault(index_type, f"{module}.{name} ({type(exc).__name__}: {exc})")
        return None


class _UnavailableMeta(type):
    def __getattr__(cls, attr: str):
        # Only reached for attributes the stand-in lacks - e.g. `.load(...)` or SQLFilter.where(...).
        # Dunders stay plain AttributeErrors so hasattr()/inspect probing doesn't raise.
        if attr.startswith("__"):
            raise AttributeError(attr)
        require_core(cls._index_type, None)


def engine_class(index_type: str, module: str, name: str) -> type:
    """Like engine_core(), for an engine class simlar subclasses or re-exports: if this build
    lacks it, a stand-in that raises IndexUnavailableError when constructed or used."""
    cls = engine_core(index_type, module, name)
    if cls is not None:
        return cls

    def __init__(self, *args, **kwargs):
        require_core(index_type, None)

    # Defined outright, not left to _UnavailableMeta: a subclass's `super().load(...)` searches
    # class dicts only and never reaches the metaclass __getattr__.
    def load(cls, *args, **kwargs):
        require_core(index_type, None)

    return _UnavailableMeta(
        name,
        (),
        {"_index_type": index_type, "__init__": __init__, "load": classmethod(load)},
    )


def require_core(index_type: str, core):
    """`core`, or raise IndexUnavailableError if engine_core() couldn't import it."""
    if core is None:
        raise IndexUnavailableError(
            f"The {index_type!r} index is unavailable: the installed simlar-engine build "
            f"doesn't provide {_UNAVAILABLE[index_type]}. "
            "simlar.unavailable_indexes() lists every index this build lacks."
        )
    return core


def unavailable_indexes() -> dict[str, str]:
    """{index type: the missing engine symbol and why} for every index this engine build lacks."""
    return dict(_UNAVAILABLE)
