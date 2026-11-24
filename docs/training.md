# Training

The command operates on local model and JSONL assets supplied by the user. It does not download data, use an API service or select a remote machine.

## Data contract

Each domain JSONL record has `input` (the native chat-serialized prompt, ending at the assistant generation prefix) and `output` (the target response). Plain records append the tokenizer's EOS during encoding. Each replay record additionally stores `pretokenized_input_ids` and `pretokenized_labels`: these preserve the generated tokens exactly, with prompt labels set to `-100`. A cap-terminated response receives no invented EOS. Prompt/padding tokens never contribute to the loss.

Replay prompts are JSONL objects containing `messages` with only system/user turns and a final user turn, or `input` with an already serialized prompt. Assistant answers from the instruction collection are not used by generation. The native chat template is applied with `enable_thinking=False` to `messages`.

Inputs are paired by line index. The training interface requires equal counts and exactly one pass: `max_steps * effective_batch_size` records on each side. Long rows are rejected. Use 2,304 tokens for Qwen3-4B and 2,048 for SmolLM3-3B/Phi-4-mini. The example configuration has the latter limit; change `max_length` to 2304 before a Qwen run.

## Bind the execution host

On the coordinator, record its hostname SHA-256. On an inspected remote GPU host, export that hash and the execution host's own hash. These values are run-local and should not be committed.

```bash
python -c 'import hashlib,socket; print(hashlib.sha256(socket.gethostname().encode()).hexdigest())'
export INSTRUCTREPLAY_COORDINATOR_HOST_SHA256='<coordinator hostname hash>'
export INSTRUCTREPLAY_REMOTE_HOST_SHA256='<execution hostname hash>'
export INSTRUCTREPLAY_REMOTE_GPU_CONFIRMED=1
```

Set `MODEL`, `PROMPTS`, `DOMAIN` and output paths to your local assets. Compute the model tree hash with `instructreplay.models.model_tree_sha256`; compute file hashes with `sha256sum`. Paths are supplied at runtime.

```bash
python -c 'import os; from instructreplay.models import model_tree_sha256; print(model_tree_sha256(os.environ["MODEL"]))'
sha256sum "$PROMPTS" "$DOMAIN"

CUDA_VISIBLE_DEVICES=0 instructreplay-generate \
  --model "$MODEL" --model-sha256 "$MODEL_SHA256" \
  --prompts "$PROMPTS" --prompts-sha256 "$PROMPTS_SHA256" \
  --output "$REPLAY_DIR" --max-length 2048

sha256sum "$REPLAY_DIR/replay.jsonl"

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

Run outputs include content identities and a completion receipt. Keep these with the output checkpoint. Before reporting a new experiment, validate a real backward step, the initial fixed point, a nonzero-KL gradient and checkpoint save/reload on the selected runtime.
