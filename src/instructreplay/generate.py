"""Generate a saved-token replay buffer from instruction prompts and exact M0."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .io import canonical_bytes, read_jsonl, sha256_file, write_new_json
from .models import load_model, model_tree_sha256, require_remote_gpu


def serialize_prompt(row, tokenizer):
    if "messages" in row:
        messages = row["messages"]
        if (not isinstance(messages, list) or not messages
                or messages[-1].get("role") != "user"
                or any(m.get("role") not in {"system", "user"} for m in messages)):
            raise ValueError("replay prompts require system/user messages ending in user")
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    if not isinstance(row.get("input"), str) or not row["input"]:
        raise ValueError("expected messages or a nonempty chat-serialized input")
    return row["input"]


def prompt_seed(seed, model_sha256, prompt):
    payload = canonical_bytes([seed, model_sha256, prompt])
    # Transformers also seeds NumPy, which requires an unsigned 32-bit seed.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def generation_eos_token_id(tokenizer):
    if (tokenizer.eos_token_id == 199999
            and tokenizer.convert_tokens_to_ids("<|end|>") == 200020):
        return [200020, 199999]
    return tokenizer.eos_token_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--prompts-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    host = require_remote_gpu()
    if (model_tree_sha256(args.model) != args.model_sha256
            or sha256_file(args.prompts) != args.prompts_sha256):
        raise ValueError("initial model or prompt identity mismatch")
    import torch
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    rows = read_jsonl(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = load_model(args.model).to("cuda").eval()
    prompts = [serialize_prompt(row, tokenizer) for row in rows]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "replay.jsonl"
    with output.open("xb") as stream, torch.no_grad():
        for prompt in prompts:
            row_seed = prompt_seed(args.seed, args.model_sha256, prompt)
            set_seed(row_seed)
            encoded = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False,
            ).to(model.device)
            ids = model.generate(
                **encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=512, eos_token_id=generation_eos_token_id(tokenizer),
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                              else tokenizer.eos_token_id),
            )[0].tolist()
            prompt_count = encoded.input_ids.shape[1]
            if len(ids) > args.max_length or len(ids) <= prompt_count:
                raise ValueError("replay output exceeds the training limit or has no response")
            stream.write(canonical_bytes({
                "input": prompt, "output": tokenizer.decode(ids[prompt_count:], skip_special_tokens=False),
                "pretokenized_input_ids": ids,
                "pretokenized_labels": [-100] * prompt_count + ids[prompt_count:],
                "sampling_seed": row_seed,
            }))
    write_new_json(args.output / "receipt.json", {
        "model_sha256": args.model_sha256, "prompts_sha256": args.prompts_sha256,
        "output_sha256": sha256_file(output), "rows": len(rows), "seed": args.seed,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 512,
        "max_length": args.max_length, "host_sha256": host,
    })


if __name__ == "__main__":
    main()
