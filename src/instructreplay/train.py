"""One-pass paired training with explicit local assets and a frozen reference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import PairedCollator, PairedDataset, PromptTargetDataset, encode_prompt_target
from .io import read_jsonl, sha256_file, write_new_json
from .models import load_model, model_tree_sha256, require_remote_gpu


def load_records(path, tokenizer, max_length):
    """Reject overflow rather than silently changing saved training prefixes."""
    rows = read_jsonl(path)
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), str):
            raise ValueError(f"row {index}: expected input/output strings")
        record = {"prompt": row["input"], "target": row["output"], "weight": 1.0}
        for key in ("pretokenized_input_ids", "pretokenized_labels"):
            if key in row:
                record[key] = row[key]
        encoded = encode_prompt_target(record, tokenizer, max_length=10**12)
        if (len(encoded["input_ids"]) > max_length
                or not any(label != -100 for label in encoded["labels"][1:])):
            raise ValueError(f"row {index}: overlong sequence or empty causal response")
        normalized.append(record)
    return normalized


def validate_schedule(config, *, world_size, domain_count, retention_count):
    import math

    for name in ("max_steps", "max_length", "effective_batch_size", "micro_batch_size"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (not math.isfinite(config["kl_coefficient"]) or config["kl_coefficient"] < 0
            or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("finite nonnegative KL weight and positive learning rate required")
    batch = config["effective_batch_size"]
    divisor = world_size * config["micro_batch_size"]
    if world_size <= 0 or batch % divisor:
        raise ValueError("effective batch must be divisible by world size times micro batch")
    if domain_count != retention_count or domain_count != config["max_steps"] * batch:
        raise ValueError("one pass requires equal domain/replay counts = steps * effective batch")
    return batch // divisor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--domain-sha256", required=True)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--replay-receipt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--deepspeed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    host = require_remote_gpu()
    if (sha256_file(args.domain) != args.domain_sha256
            or sha256_file(args.replay) != args.replay_sha256
            or model_tree_sha256(args.model) != args.model_sha256):
        raise ValueError("model or input identity differs from the supplied hashes")
    receipt = json.loads(args.replay_receipt.read_text())
    if (receipt["model_sha256"] != args.model_sha256
            or receipt["output_sha256"] != args.replay_sha256):
        raise ValueError("replay must come from the exact initial model")
    config = json.loads(args.config.read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    import torch
    from transformers import AutoTokenizer, TrainingArguments, set_seed
    from .trainer import make_joint_trainer_class

    if torch.cuda.device_count() != world:
        raise ValueError("visible GPU count must match the distributed world size")
    torch.cuda.set_device(local_rank)
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    domain = load_records(args.domain, tokenizer, config["max_length"])
    replay = load_records(args.replay, tokenizer, config["max_length"])
    accumulation = validate_schedule(config, world_size=world, domain_count=len(domain), retention_count=len(replay))
    dataset = PairedDataset(
        PromptTargetDataset(domain, tokenizer, config["max_length"]),
        PromptTargetDataset(replay, tokenizer, config["max_length"]),
    )
    student, teacher = load_model(args.model), load_model(args.model)
    teacher.requires_grad_(False)
    teacher.to(torch.device("cuda", local_rank)).eval()
    trainer_class = make_joint_trainer_class(
        teacher, coefficient=config["kl_coefficient"],
        student_forward_mode="deepspeed_unwrapped_module",
        audit_before_forward=True, preserve_buffer_names=("model.rotary_emb.inv_freq",),
        measure_backward=True, strict_runtime_identity=True,
        allowed_config_difference_keys=("bos_token_id", "pad_token_id", "use_cache"),
    )
    training = TrainingArguments(
        output_dir=str(args.output / "checkpoint"), max_steps=config["max_steps"],
        learning_rate=config["learning_rate"], lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        per_device_train_batch_size=config["micro_batch_size"],
        gradient_accumulation_steps=accumulation, max_grad_norm=1.0,
        bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=str(args.deepspeed), report_to=[], save_strategy="no", eval_strategy="no",
        remove_unused_columns=False, seed=config["seed"], data_seed=config["seed"],
        logging_steps=5,
    )
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        write_new_json(args.output / "launch.json", {
            "model_sha256": args.model_sha256, "domain_sha256": args.domain_sha256,
            "replay_sha256": args.replay_sha256, "config_sha256": sha256_file(args.config),
            "deepspeed_sha256": sha256_file(args.deepspeed), "host_sha256": host,
            "world_size": world, "config": config,
        })
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    trainer = trainer_class(model=student, args=training, train_dataset=dataset,
                            data_collator=PairedCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    trainer.train()
    if trainer.state.global_step != config["max_steps"]:
        raise ValueError("training stopped before the requested final step")
    trainer.save_model(str(args.output / "checkpoint"))
    if rank == 0:
        tokenizer.save_pretrained(args.output / "checkpoint")
        write_new_json(args.output / "completion.json", {
            "optimizer_steps": trainer.state.global_step,
            "checkpoint_sha256": model_tree_sha256(args.output / "checkpoint"),
        })


if __name__ == "__main__":
    main()
