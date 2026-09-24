from __future__ import annotations

import subprocess
import time

from pymilvus import MilvusClient, connections, Collection

from bakeoff.contenders.base import Contender

COLLECTION = "passages"
CONTAINER = "milvus-bakeoff"
URI = "http://localhost:19530"

INSERT_BATCH = 5_000
QUERY_CAP = 16_384 # https://milvus.io/docs/single-vector-search.md

INDEXING_TIMEOUT = 1800         # seconds to wait for AUTOINDEX to finish building
RELEASE_TIMEOUT = 60            # seconds to wait for a dropped collection's memory to come back
RELEASE_TOLERANCE = 5           # MB; below this the reading has stopped falling

UNITS = {"B": 1 / 1024**2, "KIB": 1 / 1024, "MIB": 1, "GIB": 1024, "TIB": 1024**2}


class MilvusContender(Contender):
    """The standalone server over gRPC, with vectors handed over as numpy rather than lists."""

    name = "milvus"
    family = "vector-db"

    def __init__(self, index_ids, texts, matrix) -> None:
        super().__init__(index_ids, texts, matrix)
        self.index = MilvusClient(uri=URI)
        self._drop_stale_collection()

    def _drop_stale_collection(self) -> None:
        """Clear a previous run's collection here, before `build` reads the baseline.

        The server outlives the benchmark, so a second run starts with the first run's index
        still resident. Dropping it inside `_build` would put the free *after* the baseline and
        subtract it from this run's footprint. Freeing is asynchronous, so this waits for the
        memory to actually come back before returning.
        """
        if not self.index.has_collection(COLLECTION):
            return
        self.index.drop_collection(COLLECTION)
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
        it would report a four-million-passage collection as a few MB and put Milvus top of
        the table on a measurement of nothing.
        """
        usage = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER],
            capture_output=True, text=True, check=True).stdout
        amount = usage.split("/")[0].strip()
        digits = len(amount) - len(amount.lstrip("0123456789."))
        return float(amount[:digits]) * UNITS[amount[digits:].upper()]

    def _warm_up(self):
        self.index.search(collection_name=COLLECTION, data=self.matrix[:1],
                          limit=1, output_fields=["id"])

    def _build(self, texts, matrix):
        # 1. Standard fast setup using MilvusClient
        self.index = MilvusClient(uri=URI)
        if self.index.has_collection(COLLECTION):
            self.index.drop_collection(COLLECTION)
            
        self.index.create_collection(
            collection_name=COLLECTION, 
            dimension=matrix.shape[1],
            metric_type="COSINE"
        )
        
        # 2. Switch to ORM connection for column-based inserts
        connections.connect(uri=URI)
        collection = Collection(COLLECTION)
        
        # 3. Columnar batch insert
        for start in range(0, len(matrix), INSERT_BATCH):
            end = min(start + INSERT_BATCH, len(matrix))
            block = matrix[start:end] 
            ids = list(range(start, end))
            # Pass data as columns: [id_column, vector_column]
            collection.insert([ids, block])
            
        # 4. Finalize
        collection.flush()
        self._await_indexed()
        collection.load()

    def _await_indexed(self) -> None:
        """Wait for AUTOINDEX to finish, and count that as build time.

        `insert` returning only means the rows landed. Milvus then builds the index on the sealed
        segments in the background, and searching during that would time an index still under
        construction -- against engines whose build call returns finished. Waiting here puts the
        cost where the other engines already carry it.
        """
        deadline = time.monotonic() + INDEXING_TIMEOUT
        while True:
            progress = self.index.describe_index(COLLECTION, "vector")
            if progress["state"] == "Finished" and not progress.get("pending_index_rows", 0):
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"milvus still indexing after {INDEXING_TIMEOUT}s: "
                    f"{progress.get('indexed_rows')}/{progress.get('total_rows')} rows")
            # time.sleep(0.05)

    def _rank(self, query_texts, query_matrix, k):
        ranked = []
        for start in range(0, len(query_matrix), QUERY_CAP):
            responses = self.index.search(
                collection_name=COLLECTION, data=query_matrix[start:start + QUERY_CAP],
                limit=k, output_fields=["id"])
            ranked.extend(self.ids_at(hit["id"] for hit in response) for response in responses)
        return ranked
