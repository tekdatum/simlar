# sim_LAR

Python search library combining keyword, semantic, and hybrid search into a single ranked result list — with first-class LangChain, LlamaIndex, and Haystack support.
---

## Installation

The open wrapper  and the engine is on public PyPI:

```bash
pip install simlar
```


```bash
pip install simlar-engine
```

## Quick start

### Keyword search

No embeddings required. Index your text and search.

```python
from simlar import RelevanceIndex

corpus = [
    "cancer treatment with immunotherapy",
    "machine learning transformers",
    "climate change renewable energy",
]
ids = [f"doc_{i}" for i in range(len(corpus))]

idx = RelevanceIndex()
idx.add(ids=ids, texts=corpus)

results = idx.search("immunotherapy clinical trial", k=1)
for r in results:
    print(r.rank, r.id, f"{r.score:.4f}")
```

### Semantic search

Provide pre-computed embedding vectors alongside your document IDs.

```python
import numpy as np
from simlar import SimlarEngine

# vectors: (n_docs, dim) float32, ideally L2-normalised
vectors = np.load("corpus_embeddings.npy")
ids     = [f"doc_{i}" for i in range(len(vectors))]

idx = SimlarEngine()
idx.add(ids=ids, vectors=vectors)

query_vec = np.load("query_embedding.npy")  # shape (1, dim)
results = idx.search(query_vec, k=5)
for r in results:
    print(r.rank, r.id, f"{r.score:.4f}")
```

### Hybrid search

Combine keyword relevance and semantic similarity into one ranked list.

```python
import numpy as np
from simlar import HelixIndex, RelevanceIndex, SimlarEngine, ReciprocalRankFusion

corpus  = ["cancer treatment with immunotherapy", "machine learning transformers", ...]
ids     = [f"doc_{i}" for i in range(len(corpus))]
vectors = np.load("corpus_embeddings.npy")  # shape (n, dim)

index = HelixIndex(
    text_index=RelevanceIndex(),
    vector_index=SimlarEngine(),
    fusion=ReciprocalRankFusion(),
    top_k=20,
)
index.add(ids=ids, texts=corpus, vectors=vectors)

query_vec = np.load("query_embedding.npy")
results = index.search(
    query_text="immunotherapy clinical trial",
    query_vector=query_vec,
    k=10,
)
for r in results:
    print(r.rank, r.id, f"{r.score:.4f}")
```

## Framework integrations

sim_LAR works as a drop-in component in:

- **LangChain** — [docs/examples/04_langchain_integration.ipynb](docs/examples/04_langchain_integration.ipynb)
- **LlamaIndex** — [docs/examples/06_llamaindex_integration.ipynb](docs/examples/06_llamaindex_integration.ipynb)
- **Haystack** — [docs/examples/05_haystack_integration.ipynb](docs/examples/05_haystack_integration.ipynb)

## Index types at a glance

| Index | Best for |
|-------|----------|
| `RelevanceIndex` | Keyword search over text — no embeddings required |
| `SimlarEngine` | Semantic search over pre-computed embedding vectors |
| `HelixIndex` | Both signals combined; corpus fits in memory |
| `StreamingHybridIndex` | Both signals; very large corpora added in batches |

## Producing embeddings

sim_LAR is model-agnostic — it accepts any `(n, dim)` float32 NumPy array. Use whichever embedding library fits your project. Example with [sentence-transformers](https://sbert.net/):

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("WhereIsAI/UAE-Large-V1")

corpus_vectors = model.encode(corpus, normalize_embeddings=True)   # (n, dim) float32
query_vec      = model.encode(["my search query"], normalize_embeddings=True)  # (1, dim)
```

## Saving and loading indexes

Every index type shares the same save/load interface.

```python
# Save
index.save("/path/to/dir")

# Load (type is detected automatically)
from simlar import load_from_directory
index = load_from_directory("/path/to/dir")
```

## Extending sim_LAR

Register a custom index class to make it compatible with `load_from_directory()` and the rest of the library:

```python
from simlar import VectorIndex, register

@register("my_index")
class MyIndex(VectorIndex):
    ...
```

## Contributing

Contributions to simlar are welcome. Because this is a dual-licensed open-core project, we require all contributors to agree to a [Contributor License Agreement (CLA)](./CONTRIBUTING.md) before we can merge. This lets us keep the open-core model viable and continue offering simlar under Apache 2.0. See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

We do not accept contributions to the proprietary engine.

## Trademarks

"sim_LAR", "TekDatum", and associated logos are trademarks of TekDatum. The Apache 2.0 license for the open code does **not** grant rights to use these marks. You may build on and redistribute the open code, but you may not use our names or logos in a way that implies endorsement or that misrepresents the origin of a fork. See [TRADEMARKS.md](./TRADEMARKS.md).

## Documentation

| Document | Description |
|----------|-------------|
| [Concepts](docs/concepts.md) | Architecture and design decisions |
| [API Reference](docs/concepts.md) | Full public API |
| [Examples](docs/examples/) | Runnable scripts |

## License

- The contents of this repository are licensed under the **Apache License, Version 2.0** — see [LICENSE](./LICENSE).
- The proprietary `simlar-engine` binary is licensed under a **Commercial EULA** — see [EULA.md](./EULA.md).
- Third-party components and their licenses are listed in [NOTICE](./NOTICE).

## License Boundary

This project uses an **open-core** model. It has two layers with **different licenses**:

| Layer | Package | License | Where it lives |
|---|---|---|---|
| **Open wrapper + SDK** | `simlar` | Apache License 2.0 | This repo · public PyPI |
| **Proprietary engine** | `simlar-engine` | Commercial EULA (closed binary) | public PyPI |

**What this means in practice:**

- The code **in this repository** is free and open under Apache 2.0. You can read it, fork it, modify it, and build on it, including commercially, subject to the Apache 2.0 terms.
- The wrapper depends on a **separate, proprietary binary package** (`simlar-engine`) that contains TekDatum's core IP. That binary is **not** open source. Installing and using it requires accepting the [Commercial EULA](./EULA.md) and, for production use, a license from TekDatum.
- Calling the proprietary engine through this open SDK does **not** make your own code subject to the EULA — your application code is yours. The EULA governs only the proprietary binary itself.

If you only want to read or contribute to the open layer, you never need a license. If you want to **run** the full product, you need the engine.

## What's open and what's not

**Open (Apache 2.0, in this repo):**
- The SDK and public API surface (`RelevanceIndex`, `HelixIndex`, `StreamingHybridIndex`, `ReciprocalRankFusion`)
- Framework adapters (LangChain / LlamaIndex / Haystack)
- The `load_from_directory()` loader and `@register` extension API
- Configuration schema, type stubs, examples, and docs

**Proprietary (Commercial EULA, separate binary):**
- The core `SimlarEngine` vector search implementation
- Compiled ranking and retrieval algorithms

We keep this boundary deliberate and documented so you always know which terms apply to which code.

---

© 2026 TekDatum.
