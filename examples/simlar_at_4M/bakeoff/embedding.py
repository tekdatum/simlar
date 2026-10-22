import hashlib
import os
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from bakeoff.environment import available_cores


class EmbeddingCache:
    """Vectors saved to disk so a repeat run skips the encode.

    The file is named after a digest of exactly what went into it -- the model, and every text
    in order -- because the vectors are used positionally: `vectors[positions_for(ids)]`. A file
    that matched the wrong corpus would not raise, it would quietly score the wrong rows, so
    "the key describes the contents" is the property worth paying for here. Change the model, the
    sample size, the split, or the loader's dedup, and the digest changes with it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path_for(self, model_name: str, texts: list[str]) -> Path:
        digest = hashlib.blake2b(model_name.encode(), digest_size=12)
        for text in texts:
            digest.update(text.encode())
            digest.update(b"\0")        # separator, so ["ab","c"] and ["a","bc"] differ
        stem = model_name.replace("/", "_")
        return self.directory / f"{stem}-{len(texts)}-{digest.hexdigest()}.npy"

    def load(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        return np.load(path)

    def save(self, path: Path, vectors: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename: a run killed mid-save leaves no half-written file
        # that the next run would happily load as if it were complete.
        partial = path.with_suffix(".partial.npy")
        np.save(partial, vectors)
        os.replace(partial, path)


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda",
                 cache: EmbeddingCache | None = None) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device={device} but no CUDA available")
        torch.set_num_threads(available_cores())
        self.model_name = model_name
        self.device = device
        self.cache = cache
        self._encoder: SentenceTransformer | None = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:          # loaded lazily so constructing is cheap
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if self.cache is None:
            return self._encode(texts, show_progress)

        path = self.cache.path_for(self.model_name, texts)
        vectors = self.cache.load(path)
        if vectors is not None:
            print(f"  reusing cached embeddings: {path}", flush=True)
            return vectors

        vectors = self._encode(texts, show_progress)
        self.cache.save(path, vectors)
        print(f"  cached embeddings: {path}", flush=True)
        return vectors

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.encoder.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=show_progress)
        return vectors.astype(np.float32)
