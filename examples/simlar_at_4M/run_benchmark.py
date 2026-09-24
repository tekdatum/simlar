"""Run the bake-off: 4,000,000 MS MARCO passages, 25,000 labelled queries, seven engines.

Before this runs end to end -- README.md, "Prerequisites", has the detail:

    conda env create -f environment.yml && conda activate simlar-4m

    docker run -d --name qdrant-bakeoff -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ./docker/milvus/start.sh                # --fresh discards the data volume

    curl http://localhost:6333/healthz      # qdrant -> healthz check passed
    curl http://localhost:9091/healthz      # milvus -> OK

qdrant and milvus are servers, timed with the gRPC hop included because that is how they are
deployed. If one is not up its row fails and the rest of the field still runs. Both also read
their footprint from `docker stats` rather than from this process, so the docker CLI has to be
on PATH and usable without sudo.

Then:

    python run_benchmark.py

This is the configuration behind
https://tekdatum.com/blog/simlar-4m-passage-benchmark/ -- budget roughly 30 GB of RAM and a
couple of hours the first time, most of it encoding the corpus. That encode is cached, so a
second run starts at the indexing.

Everything is a constant below rather than a command-line flag. Edit and re-run.
"""
import time
from pathlib import Path

from bakeoff.environment import configure

# Before anything else is imported: several engines read their thread count once, at import,
# and cache it. Capping them after the fact would silently hand one engine fewer cores.
CORES = configure()

from bakeoff.contenders.chroma import ChromaContender  # noqa: E402
from bakeoff.contenders.faiss_flat import FaissContender  # noqa: E402
from bakeoff.contenders.milvus import MilvusContender  # noqa: E402,F401
from bakeoff.contenders.qdrant import QdrantContender  # noqa: E402
from bakeoff.contenders.simlar_hybrid import SimlarHybrid, SimlarHybridLookup  # noqa: E402
from bakeoff.contenders.turbovec_quant import TurboVecContender  # noqa: E402
from bakeoff.corpus import MsMarcoLoader  # noqa: E402
from bakeoff.embedding import Embedder, EmbeddingCache  # noqa: E402
from bakeoff.evaluation import BakeOff  # noqa: E402
from bakeoff.reporting import Report  # noqa: E402

# Shrinking SAMPLE_SIZE is the obvious way to get a quick answer and it does not give you one:
# simlar's fusion weights and candidate depths were tuned for collections in the millions --
# `text_k` is 20,000 candidates, 2% of this corpus but a fifth of a 100,000-passage one -- so a
# small run reports simlar below the meaning-only engines on settings that are simply wrong for
# the size. Run it as it stands, or retune the depths with it.
SAMPLE_SIZE = 4_000_000
QUERIES_CAP = 25_000        # cap the number of queries to this many; None for all
SPLIT = "train"
MODEL_NAME = "all-MiniLM-L6-v2"

K_MAX = 100         # rank once this deep; shallower cutoffs read off the same lists
K_EVAL = 100        # the cutoff the headline table reports; has to be one of CUTOFFS
# A ladder rather than every integer up to K_MAX: at K_MAX=100 the recall curve would
# otherwise be a hundred columns wide, and the interesting shape is in the octaves.
CUTOFFS = (1, 5, 10, 25, 50, 100)
K_VALUES = sorted({k for k in CUTOFFS if k <= K_MAX} | {K_EVAL})

ROUNDS = 2          # the last round is reported, so nobody pays first-run start-up costs twice

HERE = Path(__file__).parent
SUBFOLDER = f"(idx~{SAMPLE_SIZE} queries~{QUERIES_CAP})"
RESULTS_DIR = HERE / "results"
EMBEDDING_CACHE = HERE / "embeddings" / f"embeddings_{SUBFOLDER}"   # None to always re-encode

CONTENDERS = (
    TurboVecContender,      # dense index algorithms
    FaissContender,

    SimlarHybridLookup,     # simlar: vectors, words, and the two fused
    SimlarHybrid,

    QdrantContender,        # vector databases -- both servers, see docker/ and their modules
    ChromaContender,
    MilvusContender,
)


def main() -> None:
    print(f"CPU-only run: {CORES} cores, no thread caps")

    print(f"\nLoading MS MARCO ({SPLIT} split), {SAMPLE_SIZE:,} passages ...")
    dataset = MsMarcoLoader(SAMPLE_SIZE, SPLIT, queries_cap=QUERIES_CAP).load()
    print(f"  {dataset.describe()}")

    print(f"\nEmbedding {len(dataset.corpus):,} texts with {MODEL_NAME} ...")
    cache = EmbeddingCache(EMBEDDING_CACHE) if EMBEDDING_CACHE else None
    vectors = Embedder(MODEL_NAME, cache=cache).encode(dataset.corpus.questions)
    print(f"  vectors: {vectors.shape}")

    bake_off = BakeOff(dataset, vectors, K_MAX, K_VALUES)

    print(f"\nRunning {len(CONTENDERS)} engines, {ROUNDS} round(s) ...")
    results = []
    for round_number in range(1, ROUNDS + 1):
        print(f"\n--- round {round_number} of {ROUNDS} ---")
        results = bake_off.run(CONTENDERS)      # only the last round is kept

    report = Report(results, K_EVAL, K_VALUES)
    report.print_all(num_queries=len(dataset.query_ids))

    written = report.save(RESULTS_DIR / f"{SUBFOLDER} - {time.strftime('%Y-%m-%d %H.%M.%S')}")
    print("\nWritten:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
