# Training

Generation and training consume a local checkpoint and JSONL files. Run them on a CUDA GPU execution host distinct from the coordinator. Use a Python 3.10 virtual environment with the PyTorch 2.6 build for that host's CUDA runtime installed first. From the root of a clone or an extracted [source release](../README.md#verify-a-release), install the training dependencies.

```bash
python -m pip install -e '.[train]'
instructreplay-generate --help
instructreplay-train --help
```

## Data contract

Each domain JSONL record has `input` (the native chat-serialized prompt, ending at the assistant generation prefix) and `output` (the target response). Plain records append the tokenizer's EOS during encoding. Each replay record additionally stores `pretokenized_input_ids` and `pretokenized_labels`: these preserve the generated tokens exactly, with prompt labels set to `-100`. A cap-terminated response receives no invented EOS. Prompt/padding tokens never contribute to the loss.

For example, a replay prompt can use this JSONL structure.

```json
{"messages":[{"role":"user","content":"Explain why seasons change."}]}
```

Replay prompts are JSONL objects containing `messages` with only system/user turns and a final user turn, or `input` with an already serialized prompt. Assistant answers from the instruction collection are not used by generation. The native chat template is applied with `enable_thinking=False` to `messages`.

Inputs are paired by line index. The training interface requires equal counts and exactly one pass: `max_steps * effective_batch_size` records on each side. Long rows are rejected. Use 2,304 tokens for Qwen3-4B and 2,048 for SmolLM3-3B/Phi-4-mini. The example configuration has the latter limit; change `max_length` to 2304 before a Qwen run.

## Bind the execution host

On the coordinator, compute the hostname SHA-256. On the inspected GPU host, paste that value and compute the GPU host's own value in the execution shell. These values are run-local.

```bash
# Run on the coordinator.
python -c 'import hashlib, socket; print(hashlib.sha256(socket.gethostname().encode()).hexdigest())'
```

```bash
# Run on the inspected GPU host.
export INSTRUCTREPLAY_COORDINATOR_HOST_SHA256='<paste the coordinator hash>'
export INSTRUCTREPLAY_REMOTE_HOST_SHA256="$(python -c 'import hashlib, socket; print(hashlib.sha256(socket.gethostname().encode()).hexdigest())')"
export INSTRUCTREPLAY_REMOTE_GPU_CONFIRMED=1
```

## Generate replay and train

Set these paths in the same shell. Choose new directories for replay and training outputs. This example uses SmolLM3 or Phi-4-mini with a 2,048-token limit.

```bash
MODEL=/path/to/initial-checkpoint
PROMPTS=/path/to/general-prompts.jsonl
DOMAIN=/path/to/domain-qa.jsonl
REPLAY_DIR=outputs/replay-example
RUN_DIR=outputs/training-example

MODEL_SHA256="$(python -c 'import sys; from instructreplay.models import model_tree_sha256; print(model_tree_sha256(sys.argv[1]))' "$MODEL")"
PROMPTS_SHA256="$(python -c 'import sys; from instructreplay.io import sha256_file; print(sha256_file(sys.argv[1]))' "$PROMPTS")"
DOMAIN_SHA256="$(python -c 'import sys; from instructreplay.io import sha256_file; print(sha256_file(sys.argv[1]))' "$DOMAIN")"

CUDA_VISIBLE_DEVICES=0 instructreplay-generate \
  --model "$MODEL" --model-sha256 "$MODEL_SHA256" \
  --prompts "$PROMPTS" --prompts-sha256 "$PROMPTS_SHA256" \
  --output "$REPLAY_DIR" --max-length 2048

REPLAY_SHA256="$(python -c 'import sys; from instructreplay.io import sha256_file; print(sha256_file(sys.argv[1]))' "$REPLAY_DIR/replay.jsonl")"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=29501 -m instructreplay.train \
  --model "$MODEL" --model-sha256 "$MODEL_SHA256" \
  --domain "$DOMAIN" --domain-sha256 "$DOMAIN_SHA256" \
  --replay "$REPLAY_DIR/replay.jsonl" --replay-sha256 "$REPLAY_SHA256" \
  --replay-receipt "$REPLAY_DIR/receipt.json" \
  --config configs/train.json --deepspeed configs/zero2.json --output "$RUN_DIR"
```

For Qwen, change both the generation limit and training configuration to 2304. `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, `--master_port` and output locations are selected per run. Effective domain batch size stays 32; the interface derives gradient accumulation from world size and microbatch size and rejects a nonintegral result.

## Objective and runtime

`L = CE_domain + 0.25 * KL(M0 || student)_replay`, with independent response-token means within each sequence followed by a mean over sequences. Log-softmax and KL arithmetic use FP32. A value-preserving analytical gradient makes the equal-logit KL fixed point exactly zero. Reference inference runs without gradients and in evaluation mode, with no shared trainable parameters.

Training uses BF16, constant learning rate `1e-5`, no warmup, AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, gradient clipping 1, ZeRO-2 and non-reentrant gradient checkpointing. The retention forward uses the unwrapped DeepSpeed module; the domain forward uses the managed model. Initial identity checks compare parameters, buffers, architecture, tying, attention implementation and optimizer binding. The original RoPE buffer is restored before the first forward.

Generation samples one prompt at a time at temperature 1, top-p 1, top-k 0 and a 512-token cap, with a model/prompt-specific deterministic 32-bit seed. The CLI accepts `--seed` values from 0 through 4,294,967,295. This portable generation interface constructs new replay buffers; it does not recreate the paper's original sharded sampling stream. No new run inherits the paper's verification status.

Run outputs include content identities and a completion receipt. Keep these with the output checkpoint. Validate a real backward step, the initial fixed point, a nonzero-KL gradient, and checkpoint save/reload on your selected runtime before reporting a new experiment. Use the [evaluation guide](evaluation.md) to score the resulting checkpoint and compute the metrics.
