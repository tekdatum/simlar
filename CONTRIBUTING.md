# Contributing to [PRODUCT NAME]

Thanks for your interest in contributing! This document explains how to contribute to the **open** wrapper and SDK, and the legal sign-off we require before we can merge.

> **Scope:** Contributions are accepted to the open-source wrapper and SDK in this repository (Apache License 2.0). We do **not** accept external contributions to the proprietary engine.

---

## Why we require a contribution agreement

This is a dual-licensed **open-core** project: the wrapper is Apache 2.0, and a separate proprietary engine is offered commercially. To keep this model viable — and to keep offering the wrapper under Apache 2.0 — we need clear rights to the contributions we merge. We use one of the two standard mechanisms below. **[Choose ONE and delete the other before publishing.]**

### Developer Certificate of Origin (DCO) 

The [DCO](https://developercertificate.org/) is a per-commit attestation that you wrote the code (or have the right to submit it) and agree it can be distributed under the project's license. There is no separate document to sign — you certify by adding a `Signed-off-by` line to each commit:

```bash
git commit -s -m "Your commit message"
```

This appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and a real email. Our CI checks for this line on every commit in a pull request.

> **Note for an open-core business:** the DCO does **not** assign copyright or grant you (the maintainer) a relicensing right beyond the project's existing license. If you may later want to relicense the open layer or move code between the open and proprietary sides, prefer Option B (CLA). [Discuss with counsel.]


---

## Development setup

```bash
git clone https://github.com/tekdatum/simlar.git
cd simlar
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

To run the full product locally you also need the proprietary engine, which requires a license. Most wrapper/SDK contributions can be developed and tested against the public interfaces and mocks/fixtures without it; see [tests/README] for how we stub the engine in CI.

## Workflow

1. Open or comment on an issue describing the change, so we can align before you invest time.
2. Fork and create a branch: `git checkout -b feature/short-description`.
3. Make focused commits (signed off if we use DCO).
4. Add or update tests and docs.
5. Run checks locally:
   ```bash
   ruff check . && ruff format --check .
   mypy src
   pytest
   ```
6. Open a pull request against `main`. Fill in the PR template and link the issue.

## What we look for

- Clear, focused changes — one logical change per PR.
- Tests for new behavior and bug fixes.
- No breaking changes to the public SDK API without discussion (we follow [SemVer / versioning policy]).
- No code copied from incompatible-license sources. By contributing, you confirm the code is your original work or appropriately licensed and that you have the right to submit it.
- Do not add dependencies under copyleft licenses (GPL/AGPL/LGPL) without prior maintainer approval — they can complicate the open-core boundary.

## Reporting security issues

Please do **not** open public issues for security vulnerabilities. Email [security@tekdatum.com] — see [SECURITY.md] for our disclosure policy.

## Code of conduct

This project follows our [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

## Questions

Reach us at [community@tekdatum.com].
