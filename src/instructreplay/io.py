"""Content hashes and JSONL input without machine-specific configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL must contain nonempty object records")
    return rows


def write_new_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical_bytes(value))
