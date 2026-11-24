# Contribute to InstructReplay

[Report a bug](https://github.com/xiaofengShi/InstructReplay/issues/new?template=bug.yml) · [Share a reproduction](https://github.com/xiaofengShi/InstructReplay/issues/new?template=reproduction.yml) · [Browse open issues](https://github.com/xiaofengShi/InstructReplay/issues)

## Share a useful report

For a bug, include the release or commit, Python and PyTorch versions, the smallest command that reproduces it, and the expected and actual result. For a reproduction, include the initial checkpoint revision, domain and replay data provenance, training configuration, evaluation protocol, and initial and adapted benchmark scores. Partial and failed runs are useful when the failure stage is clear.

Use synthetic inputs or public, licensed examples. Remove credentials, private hostnames and local paths from commands and logs. Report only the aggregate scores and provenance that you can share publicly.

## Send a pull request

- Open an issue to discuss substantive changes to training or evaluation semantics. Small fixes to code and documentation can go straight to a pull request.
- In the description, identify the behavior that changes and the test or example that demonstrates it.
- Run `python -m pytest -q`, `ruff check src tests tools`, and `python tools/package.py` in the [CPU environment](README.md#run-training-and-tests). For playground changes, run `node --test tests/playground.test.mjs`.
- When a change adds a public file, include it in `publish-files.json` so tagged builds contain the same public files.
