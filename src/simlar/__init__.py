from simlar.contracts import SearchResult, TextIndex, VectorIndex
from simlar.fusion import ReciprocalRankFusion
from simlar.indexes.helix_index import HelixIndex
from simlar.indexes.lookup_index import LookupIndex
from simlar.indexes.registry import load_from_directory, register
from simlar.indexes.relevance_index import RelevanceIndex
from simlar.indexes.simlar_engine import SimlarEngine
from simlar.indexes.streaming_index import StreamingHelixIndex as StreamingHybridIndex

_MIN_ENGINE_VERSION = (1, 1, 0)


def _check_engine_version() -> None:
    """Best-effort compatibility guard between this wrapper and simlar-engine.

    simlar-engine is a proprietary binary installed as a separate package
    (see docs/installation.md) — there is no packaging-level dependency pin
    tying the two together, so a too-old engine would otherwise fail with a
    confusing TypeError deep inside a search() call instead of a clear error
    at import time. Silently skipped when the installed engine doesn't
    expose __version__ (e.g. a test double, or a build that predates this
    check) rather than treated as a failure — this is a floor, not a full
    dependency resolver.
    """
    import simlar_engine

    engine_version = getattr(simlar_engine, "__version__", None)
    if engine_version is None:
        return
    try:
        parsed = tuple(int(p) for p in engine_version.split(".")[:3])
    except ValueError:
        return
    if parsed < _MIN_ENGINE_VERSION:
        raise RuntimeError(
            f"simlar requires simlar-engine >= {'.'.join(map(str, _MIN_ENGINE_VERSION))}, "
            f"but {engine_version} is installed. See docs/installation.md for the "
            "compatible version matrix."
        )


_check_engine_version()

__all__ = [
    # Contracts — extension points
    "SearchResult",
    "TextIndex",
    "VectorIndex",
    # Fusion
    "ReciprocalRankFusion",
    # Indexes
    "RelevanceIndex",
    "SimlarEngine",
    "HelixIndex",
    "StreamingHybridIndex",
    "LookupIndex",
    # Registry
    "register",
    "load_from_directory",
]
