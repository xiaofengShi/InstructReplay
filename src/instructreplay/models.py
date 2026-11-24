"""Load native model implementations without architecture substitution."""
from __future__ import annotations

import hashlib
from pathlib import Path


def model_tree_sha256(directory):
    root = Path(directory).resolve(strict=True)
    if not (root / "config.json").is_file():
        raise ValueError("model directory requires config.json")
    names = [p for p in sorted(root.rglob("*")) if p.is_file() and ".cache" not in p.parts]
    digest = hashlib.sha256()
    for path in names:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )


def require_remote_gpu():
    """Require explicit binding to an execution host distinct from the coordinator."""
    import os
    import socket

    actual = hashlib.sha256(socket.gethostname().encode()).hexdigest()
    if (os.environ.get("INSTRUCTREPLAY_REMOTE_GPU_CONFIRMED") != "1"
            or os.environ.get("INSTRUCTREPLAY_REMOTE_HOST_SHA256") != actual
            or os.environ.get("INSTRUCTREPLAY_COORDINATOR_HOST_SHA256") in (None, "", actual)):
        raise ValueError("bind an inspected remote host and a distinct coordinator before GPU work")
    return actual
