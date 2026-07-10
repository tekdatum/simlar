# Tests

The public test suite runs **without** the proprietary `simlar-engine` installed.
`tests/conftest.py` injects a lightweight stub via `pytest_configure` (which runs
before any test module is imported), so `import simlar` works in all test files.

## Running locally

```bash
# Option A — with the real engine (broader coverage)
pip install simlar-engine -

# Option B — stub-only mode (no engine access required)
pip install numpy pytest
pytest tests/ -v
```

## Test layout

| File | What it tests |
|---|---|
| `conftest.py` | Injects `simlar_engine` stubs; provides `make_results()` helper |
| `test_contracts.py` | `SearchResult`, `IndexConfig` (pure Python dataclasses) |
| `test_registry.py` | `@register`, `build()`, `load_from_directory()` |
| `test_persistence.py` | `write_config()`, `read_config()` |
| `integrations/test_langchain.py` | `SimlarVectorStore` with mock embeddings |

## Notes
- The stub `_HelixCore`, `_SimlarCore`, etc. return deterministic dummy results.
  Do not use these tests to validate search quality — they only verify the integration
  contracts (correct types, shapes, persistence format).
