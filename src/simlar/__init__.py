from simlar.contracts import SearchResult, TextIndex, VectorIndex
from simlar.fusion import ReciprocalRankFusion
from simlar.indexes.helix_index import HelixIndex
from simlar.indexes.registry import load_from_directory, register
from simlar.indexes.relevance_index import RelevanceIndex
from simlar.indexes.simlar_engine import SimlarEngine
from simlar.indexes.streaming_index import StreamingHelixIndex as StreamingHybridIndex

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
    # Registry
    "register",
    "load_from_directory",
]
