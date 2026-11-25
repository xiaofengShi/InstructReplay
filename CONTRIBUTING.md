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

## Build the project homepage

Use Node.js 24. Edit the React source in `site/src/home.tsx` and its styles in `site/styles/home.css`. The build bundles the published `docs/results.csv` into the rendered results table and writes `docs/index.html` for GitHub Pages.

```bash
npm ci --prefix site
npm run build --prefix site
python -m http.server 8000 --directory docs
```

Commit the generated HTML alongside its source changes. CI rebuilds it from the lockfile and checks for drift. The vendored BoardUI components and tokens come from the [public registry](https://www.boardui.com/getting-started) under the [MIT license](site/LICENSE.boardui).

The README and social result cards can be regenerated with `python tools/render_result_cards.py` in an environment with Matplotlib. The script uses the published aggregate means in [the results guide](docs/results.md).

## Publish a source release

Maintainers keep the Python project, module and citation versions in sync. Push an annotated `vX.Y.Z` tag with the release notes. The release workflow checks the tag, packages the public allowlist and publishes the source ZIP and checksum. If publication fails before a release exists, rerun the failed job or run **Publish source release** from Actions with the existing tag.
