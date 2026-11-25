# InstructReplay

**Add domain knowledge to an instruction-tuned model while retaining its general assistant behavior.**

[**Paper (arXiv:2611.32918)**](https://arxiv.org/abs/2611.32918) · [**Project homepage**](https://xiaofengshi.github.io/InstructReplay/) · [**Try the metrics**](https://xiaofengshi.github.io/InstructReplay/try/) · [**Train on your data**](docs/training.md) · [**Download a release**](#verify-a-release)

[![CPU checks](https://github.com/xiaofengShi/InstructReplay/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/xiaofengShi/InstructReplay/actions/workflows/tests.yml)
[![Apache-2.0 license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

InstructReplay saves the initial model’s own assistant replies to general instructions. During factual fine-tuning, domain answers use cross-entropy and saved replies use forward KL from the frozen initial model. The same replay buffer can be reused for another domain on that checkpoint.

<p align="center">
  <img alt="Across six settings, mean forgetting was 28.07 versus 2.44 percentage points and mean factual acquisition was 15.23 versus 15.49 points." src="docs/assets/result-card.png" width="100%">
</p>

## Start here

| Your goal | First step |
| --- | --- |
| Inspect the reported experiments | [Explore all six settings](https://xiaofengshi.github.io/InstructReplay/#results), [download the results CSV](docs/results.csv), and [read the control comparisons](docs/results.md#replay-controls). |
| Check the evaluation math | [Open the playground](https://xiaofengshi.github.io/InstructReplay/try/) or [run the CPU example](#run-the-metrics-example). Both use the same synthetic scores. |
| Adapt a model | [Generate replay and fine-tune](docs/training.md) on a GPU execution host using a local checkpoint and licensed data. |
| Report a run or help improve the project | [Share a reproduction](https://github.com/xiaofengShi/InstructReplay/issues/new?template=reproduction.yml), [report a bug](https://github.com/xiaofengShi/InstructReplay/issues/new?template=bug.yml), or read [Contributing](CONTRIBUTING.md). |

## What the study measured

| Mean across six settings | Standard fine-tuning | InstructReplay |
| --- | ---: | ---: |
| Domain acquisition ↑ | +15.23 pp | **+15.49 pp** |
| General forgetting ↓ | 28.07 pp | **2.44 pp** |
| Median IFEval response length relative to the initial model | 0.11× | **0.85×** |

InstructReplay had the highest acquisition-minus-forgetting score **in all six settings** against seven tested adaptation baselines. The same reference loss produced a mean IFEval drop of **43.3 pp** on domain answers, **14.2 pp** on unprompted samples, and **2.6 pp** on the model’s own assistant turns. The BOS and assistant-turn conditions matched their response-token budgets.

The study used SmolLM3-3B, Qwen3-4B, and Phi-4-mini-instruct, with CommonCorpus energy/engineering and PMC biomedical questions. Domain acquisition uses model-judge confirmation on 512 closed-book questions per domain. The results cover 300-step runs on these checkpoints. [Read the settings, controls, and limitations.](docs/results.md)

## Run the metrics example

From the root of a clone, run the public metric CLI using Python 3.10+ and the included synthetic scores. This example uses the standard library and runs on CPU.

```bash
git clone https://github.com/xiaofengShi/InstructReplay.git
cd InstructReplay
PYTHONPATH=src python3 -m instructreplay.metrics docs/try/example.json --output metrics.json
cat metrics.json
```

Expected values in percentage points are **A = 12.5**, **F = 3**, **W = 5**, and **S = 9.5**. The output path must be new. The CLI takes already-scored inputs. [See the score format and benchmark protocol.](docs/evaluation.md)

## Run training and tests

1. Supply a local instruction-tuned checkpoint, licensed domain QA and general instructions.
2. [Generate replies once and train on paired examples](docs/training.md). The study used a 1:1 replay ratio and KL weight 0.25.
3. Score the initial and adapted checkpoints under the same benchmark protocol, then [aggregate the scores](docs/evaluation.md).

Use the appropriate PyTorch 2.6 build for your runtime. Install the `train` extra for GPU training. For the CPU test suite, use Python 3.10 in a virtual environment.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The release includes the replay generator, trainer, configurations, metric CLI, and reported aggregate results. Bring your own model weights, licensed training inputs, and evaluation harnesses. The synthetic example checks score aggregation, and a new GPU run constructs a new replay buffer. See the [evaluation guide](docs/evaluation.md) for the paper protocol and scorer boundaries.

## Verify a release

Download the matching `instructreplay-vX.Y.Z.zip` and `.zip.sha256` assets from the [GitHub release](https://github.com/xiaofengShi/InstructReplay/releases/latest). In the directory containing both files, run the following commands. The example uses v0.1.1.

```bash
sha256sum --check instructreplay-v0.1.1.zip.sha256
unzip instructreplay-v0.1.1.zip
cd instructreplay
PYTHONPATH=src python3 -m instructreplay.metrics docs/try/example.json --output metrics.json
```

On macOS, use `shasum -a 256 --check instructreplay-v0.1.1.zip.sha256` for the checksum.

## Citation and license

Xiaofeng Shi, Qian Kou, and Hua Zhou, Beijing Academy of Artificial Intelligence (BAAI). *InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning*. 2026. [arXiv:2611.32918](https://arxiv.org/abs/2611.32918) · [Citation metadata](CITATION.cff)

```bibtex
@misc{shi2026instructreplay,
  title         = {{InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning}},
  author        = {Shi, Xiaofeng and Kou, Qian and Zhou, Hua},
  year          = {2026},
  eprint        = {2611.32918},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.32918}
}
```

[Apache License 2.0](LICENSE) · [Attribution notices](NOTICE) · [Project homepage](https://xiaofengshi.github.io/InstructReplay/)
