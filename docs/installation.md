# Installation

## Packages

`simlar` (this repo) and `simlar-engine` (the proprietary binary it depends on) are two separately versioned packages, both published on public PyPI:

```bash
pip install simlar
pip install simlar-engine
```

`simlar-engine` is licensed under a [Commercial EULA](../EULA.md), not Apache 2.0 — installing it means accepting those terms. `simlar` itself only depends on `numpy`; `simlar-engine` is not declared as a normal packaging dependency because it isn't Apache-licensed, so it must be installed as a separate, deliberate step.

## Compatibility matrix

There is no packaging-level version pin between the two packages, so it's up to you to keep them within a compatible range. `simlar` checks this at import time on a best-effort basis (see below) and raises a clear error if it can tell the installed engine is too old — but that check only catches known-bad combinations, not the reverse (an engine that's *newer* than anything this version of `simlar` was written against).

| `simlar` version | requires `simlar-engine` |
|---|---|
| >= 1.1.0 | >= 1.1.0 |
| 1.0.x | >= 1.0.0, < 1.1.0 |

When in doubt, install matching minor versions of both packages.

## Optional extras

Framework integrations are opt-in:

```bash
pip install "simlar[langchain]"
pip install "simlar[llama_index]"
pip install "simlar[haystack]"
```
