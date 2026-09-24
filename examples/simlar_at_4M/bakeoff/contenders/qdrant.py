"""Qdrant -- a production-grade Rust engine, measured as the server it actually is.

`QdrantClient(":memory:")` is not Qdrant. It is a pure-Python reference implementation the client
ships for testing, and its per-point `np.append` reallocates the whole array every time, making
indexing O(n^2): 334 s and 14 GB for a million passages, none of it the engine's doing. Publishing
that under Qdrant's name would be measuring the test double. So this contender needs the real one:

    docker run -d --name qdrant-bakeoff -p 6333:6333 -p 6334:6334 qdrant/qdrant

The gRPC hop is inside the timings, which is the honest choice -- it is how Qdrant is deployed.
If the server is not up this engine fails its own row and the rest of the bake-off carries on.
"""

from __future__ import annotations

import subprocess
import time

from qdrant_client import QdrantClient
from qdrant_client.models import CollectionStatus, Distance, QueryRequest, VectorParams

from bakeoff.contenders.base import Contender

COLLECTION = "passages"
CONTAINER = "qdrant-bakeoff"

HOST, HTTP_PORT, GRPC_PORT = "localhost", 6333, 6334
REQUEST_TIMEOUT = 600           # seconds; a million-point upload is not a quick call

UPLOAD_BATCH = 1000             # points per request
QUERY_CAP = 1000                # queries per request, so one batch is not a giant payload
INDEXING_TIMEOUT = 900          # seconds to wait for the optimizer to settle
RELEASE_TIMEOUT = 60            # seconds to wait for a dropped collection's memory to come back
RELEASE_TOLERANCE = 5           # MB; below this the reading has stopped falling

UNITS = {"B": 1 / 1024**2, "KIB": 1 / 1024, "MIB": 1, "GIB": 1024, "TIB": 1024**2}


class QdrantContender(Contender):
    """The real engine over gRPC, with vectors handed over as numpy rather than Python lists."""

    name = "qdrant"
    family = "vector-db"

    def __init__(self, index_ids, texts, matrix) -> None:
        super().__init__(index_ids, texts, matrix)
        self.index = QdrantClient(host=HOST, port=HTTP_PORT, grpc_port=GRPC_PORT,
                                  prefer_grpc=True, timeout=REQUEST_TIMEOUT)
        self._drop_stale_collection()

    def _drop_stale_collection(self) -> None:
        """Clear a previous run's collection here, before `build` reads the baseline.

        The server outlives the benchmark, so a second run starts with the first run's index
        still resident. Dropping it inside `_build` would put the free *after* the baseline and
        subtract it from this run's footprint -- a repeat run reported -128.7 MB. Freeing is
        asynchronous, so this waits for the memory to actually come back before returning.
        """
        if not self.index.collection_exists(COLLECTION):
            return
        self.index.delete_collection(COLLECTION)
        settled = self.footprint_mb()
        deadline = time.monotonic() + RELEASE_TIMEOUT
        while time.monotonic() < deadline:
            current = self.footprint_mb()
            if current >= settled - RELEASE_TOLERANCE:      # stopped falling
                return
            settled = current

    def footprint_mb(self) -> float:
        """The container's memory, not this process's.

        The index lives in the server, so the benchmark's own RSS only ever sees the client:
        it would report a million-passage collection as a few MB and put qdrant top of the
        table on a measurement of nothing.
        """
        usage = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
            capture_output=True, text=True, check=True).stdout
        amount = usage.split("/")[0].strip()             # "130.3MiB / 91.81GiB"
        digits = len(amount) - len(amount.lstrip("0123456789."))
        return float(amount[:digits]) * UNITS[amount[digits:].upper()]

    def _warm_up(self):
        self.index.query_batch_points(
            collection_name=COLLECTION,
            requests=[QueryRequest(query=self.matrix[0].tolist(), limit=1)],
        )

    def _build(self, texts, matrix):
        self.index = QdrantClient(host=HOST, port=HTTP_PORT, grpc_port=GRPC_PORT,
                                  prefer_grpc=True, timeout=REQUEST_TIMEOUT)
        if self.index.collection_exists(COLLECTION):
            self.index.delete_collection(COLLECTION)        # left over from an earlier run
        self.index.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=matrix.shape[1], distance=Distance.COSINE),
        )
        # The matrix goes over as numpy. `upsert` with a list of PointStruct would first turn every
        # row into 384 separate Python floats -- 15 KB a point against 1.5 KB of actual data.
        self.index.upload_collection(collection_name=COLLECTION, vectors=matrix,
                                     ids=range(len(matrix)), batch_size=UPLOAD_BATCH, wait=True)
        self._await_green()

    def _await_green(self) -> None:
        """Wait for the optimizer to finish, and count that as build time.

        `wait=True` only means the points landed. Qdrant then builds its HNSW graph in the
        background, and searching during that would time an index still under construction --
        against engines whose build call returns finished. Waiting here puts the cost where the
        other engines carry it.
        """
        deadline = time.monotonic() + INDEXING_TIMEOUT
        while self.index.get_collection(COLLECTION).status != CollectionStatus.GREEN:
            if time.monotonic() > deadline:
                raise TimeoutError(f"qdrant still indexing after {INDEXING_TIMEOUT}s")
            time.sleep(0.05)

    def _rank(self, query_texts, query_matrix, k):
        ranked = []
        for start in range(0, len(query_matrix), QUERY_CAP):
            batch = query_matrix[start:start + QUERY_CAP]
            responses = self.index.query_batch_points(
                collection_name=COLLECTION,
                requests=[QueryRequest(query=vector, limit=k) for vector in batch.tolist()],
            )
            ranked.extend(self.ids_at(hit.id for hit in response.points)
                          for response in responses)
        return ranked
