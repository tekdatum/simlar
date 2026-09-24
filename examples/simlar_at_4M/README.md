# simlar at 4 million passages

The benchmark behind [*simlar at 4 Million: Faster and More Accurate Than the
Field*](https://tekdatum.com/blog/simlar-4m-passage-benchmark/), packaged so you can run it
yourself.

Seven search engines index the same 4,000,000 MS MARCO passages, answer the same 25,000
questions whose correct answer is known, and are timed on the same machine with the same
vectors. What comes out is one table: how often each engine put the right passage in its
top 100, how long a query took, how long the index took to build, and how much memory it held.

| | |
|---|---|
| **Engines** | simlar (two setups), FAISS, turbovec, Qdrant, Chroma, Milvus (standalone server) |
| **Corpus** | MS MARCO v2.1, `train` split, deduplicated |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, unit-normalised, float32 |
| **Score** | recall at 1, 5, 10, 25, 50 and 100 against one human-marked answer per query |

## What it costs to run

**About ~30 GB of RAM and ~1.5 hours**, most of that encoding the corpus. That encode
happens once: vectors are cached to disk under `embeddings/`, keyed by a digest of the model and
every text in order, so a second run starts at the indexing. The cache file is ~6.2 GB.

The encode runs on a CUDA GPU and nothing else does — all searching is on the CPU. Without a
card, pass `device="cpu"` to `Embedder` in `run_benchmark.py` and expect that step to take
considerably longer.

Shrinking `SAMPLE_SIZE` is the obvious way to get a quick answer and it does not give you one.
simlar's parameters are the recommended for the current corpus size (4M) and the recall at 100 this benchmark has, that means that if you modify the script shrinking the corpus size or trying with a different recall@k, the is highly probable that the result could improved because the parameters are not tuned for that size. To avoid this the incoming versions of simlar will do this automatically.

## Prerequisites

Four things, in this order. Nothing here is optional except the GPU.

### 1. Hardware

About **~30 GB of RAM** and a couple of hours for the first run — see [What it costs to
run](#what-it-costs-to-run) above — plus 6.2 GB of disk for the embedding cache under
`embeddings/`. You can either comment out the contenders you do not want to run, or set `ROUNDS=1` in `run_benchmark.py` to run each engine once and get a quick answer.

### 2. The conda environment

```bash
conda env create -f environment.yml
conda activate simlar-4m
```

`environment.yml` pins every version the published run used, and takes everything except python
itself from pip: faiss-cpu, chromadb and pymilvus all ship wheels, and mixing conda-forge and pip
builds of the same native library is how a benchmark ends up measuring two different BLAS
implementations.

`simlar-engine` is the proprietary binary that `simlar` wraps. It is a separate package under a
[commercial EULA](../../EULA.md), so `simlar` does not pull it in as a dependency — it is named
explicitly in `environment.yml`, and installing it means accepting
those terms. See [the installation docs](../../docs/installation.md) for the version matrix.

### 3. The two servers

Qdrant and Milvus do not run in-process. They are measured as the servers they actually are, with
the gRPC hop inside the timings, because that is how both are deployed.

Qdrant is one line:

```bash
docker run -d --name qdrant-bakeoff -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Milvus is a script, because it is not one line:

```bash
./docker/milvus/start.sh            # --fresh discards the data volume and starts clean
```

That is Milvus Standalone v3.0.1 with embedded etcd and local storage — one container, no compose
file, no MinIO — and the script blocks until the server reports healthy. Two files under
`docker/milvus/` are mounted into it and **both are load-bearing**:

- `embedEtcd.yaml` — the image ships no such file, and with `ETCD_USE_EMBED=true` and nothing
  mounted there the embedded etcd segfaults during startup: the container exits 134 in under a
  second, over and over.
- `user.yaml` — raises the proxy's gRPC message limits from their stock 64 MB to 2 GiB − 1, the
  most a single protobuf message can ever be, so the insert chunk size is the benchmark's choice
  rather than the wire's. Nothing else is overridden; the numbers in the table are otherwise
  stock Milvus.

Recreating either container by hand is the usual way this breaks — a plain `docker run` without
those two mounts gives you a Milvus that crash-loops.

The container **names** are load-bearing too: `qdrant-bakeoff` and `milvus-bakeoff` are what
`bakeoff/contenders/qdrant.py` and `bakeoff/contenders/milvus.py` pass to `docker stats`.

### 4. Docker usable without sudo

Both server contenders read their index footprint from `docker stats` rather than from this
process — the index is not in this process at all, so the benchmark's own RSS (Resident Set Size) would report a
four-million-passage collection as a few MB and put them top of the table on a measurement of
nothing. The `docker` CLI therefore has to be on PATH and runnable as your own user.

### Check before you start

```bash
curl http://localhost:6333/healthz      # qdrant  -> healthz check passed
curl http://localhost:9091/healthz      # milvus  -> OK
docker stats --no-stream qdrant-bakeoff milvus-bakeoff
```

Run these *before* the benchmark rather than discovering the answer mid-way: a server that is
down when its turn comes costs that engine its round. Without either server those rows fail and
the rest of the field still runs — that is true of every engine here. A failure prints its
traceback and costs one row, never the run.

## Running it

```bash
python run_benchmark.py
```

There are no flags. Every setting is a constant at the top of `run_benchmark.py` — edit and
re-run. `CONTENDERS` is the field, `ROUNDS` is how many times it runs (the last one is
reported, so nobody is penalised for start-up costs they only ever pay once), and
`EMBEDDING_CACHE` can be set to `None` to always re-encode.

Each run writes `comparison.csv` and `recall_curve.csv` into a timestamped folder under
`results/`. `simlar_at_4M.ipynb` does the same thing a cell at a time, with the published
numbers alongside for comparison; open it with `jupyter lab simlar_at_4M.ipynb`.

## Notes

**Chroma** is left unpatched. It builds the index, then fails the query call with `too many SQL
variables` — SQLite's bound-parameter ceiling, reached inside Chroma's own query planner on a
request of 1,000 queries × 100 results. It is the size of the request, not of the collection: at
250 queries per call the same index answers all 25,000 questions. `QUERY_CAP` in
`bakeoff/contenders/chroma.py` is the line to change if you want that row.

## Layout

```
run_benchmark.py          the entry point: constants, the field, and main()
environment.yml           the conda environment
simlar_at_4M.ipynb        the same run, a cell at a time
docker/milvus/            start.sh plus the two configs Milvus Standalone will not boot without
bakeoff/
  corpus.py               MS MARCO -> passages, queries, one answer each
  embedding.py            encode once, cache, reuse
  contenders/base.py      the stopwatch every engine is timed by
  contenders/*.py         one file per engine, each its own configuration
  evaluation.py           build, warm, rank, score -- and survive a crash
  scoring.py              what one measurement is
  reporting.py            the two tables
```

The measurement rules worth knowing before you read a number off this:

- **Build time** is until the index is ready to search, including Qdrant waiting for its HNSW
  graph to finish in the background. An engine that returns early from `build` and finishes the
  job on the first query would otherwise look fast twice.
- **Memory** is the resident set the built index added to the process — the only unit that
  counts an index living in Rust or C++ the same way it counts one in Python. Qdrant's is read
  from its container instead, because its index is not in this process at all.
- **Warm-up** is one throwaway search after the build, off the clock, so that whatever an engine
  defers to its first call does not land on whichever engine happens to run first.
