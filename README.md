# InstructReplay

**Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning**

Xiaofeng Shi, Qian Kou, Hua Zhou · Beijing Academy of Artificial Intelligence (BAAI)


InstructReplay preserves an initial instruction-tuned model's behavior by replaying its own responses to general instructions during domain fine-tuning. A frozen copy of the initial model supplies full-vocabulary next-token distributions on those saved responses. The objective combines response-only domain cross-entropy with forward KL on replay prefixes, with each sequence normalized by its own response length.

## Install

Use Python 3.10 and install a PyTorch 2.6 build appropriate for your CUDA driver before installing this package. The training stack uses Transformers 4.57.1, Accelerate 1.6.0 and DeepSpeed 0.14.3.

```bash
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python -m pytest
```

## Run

1. Obtain a local initial model and your own licensed domain and replay instructions. Keep the native tokenizer and chat template with the model weights. No training data, model weights, benchmark questions or credentials are distributed here.
2. Generate saved-token responses once with `instructreplay-generate`.
3. Fine-tune on paired domain/replay records with `instructreplay-train`.
4. Run the benchmark scorers described in [evaluation](docs/evaluation.md), then aggregate the five scores with `instructreplay-metrics`.

See [training](docs/training.md) for the complete commands, data contract, host binding and input hashes. GPU commands require an explicitly inspected execution host distinct from the coordinator.

Install the optional training engine with `python -m pip install -e '.[train]'` in the GPU environment. CPU checks can run without DeepSpeed or Triton installed.

## Package structure

| Path | Responsibility |
| --- | --- |
| `src/instructreplay/data.py` | Exact saved-token masks, paired datasets and right padding |
| `src/instructreplay/losses.py` | Sequence-normalized cross-entropy and forward KL |
| `src/instructreplay/trainer.py` | Frozen reference, runtime identity and joint objective |
| `src/instructreplay/generate.py` | Offline instruction-response replay generation |
| `src/instructreplay/train.py` | One-pass distributed training and run receipts |
| `src/instructreplay/metrics.py` | Five-benchmark A/F/W/S aggregation |
| `src/instructreplay/models.py`, `io.py` | Native model loading, content hashes and JSONL |
| `configs/` | Method hyperparameters and ZeRO-2 configuration |
| `tests/` | CPU numerical, masking, data, runtime and metric checks |

## Reproducibility

This release contains the method implementation and a portable training interface. Its extracted loss/data/runtime functions are covered by CPU regression tests. The distributed command is a new interface and requires validation on the chosen GPU runtime before attributing a run to the published results.

The paper uses three model families, two domains, 9,600 domain/replay pairs and 300 optimizer steps. Complete row-level redistribution permission for the experimental data mixture is not established, so this repository cannot reproduce the reported training or benchmark results by itself. The paper's companion supplement supplies final result tables and checks; it is distributed separately from the software.

## License and citation

Original software is available under [Apache-2.0](LICENSE). Model, data and external benchmark licenses remain with their respective owners. Author and citation metadata are in [CITATION.cff](CITATION.cff).
