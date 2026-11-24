"""Saved-token response masks and right-padded causal language-model batches."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

def encode_prompt_target(record: dict[str, Any], tokenizer, max_length: int = 4096) -> dict[str, Any]:
    if "pretokenized_input_ids" in record or "pretokenized_labels" in record:
        input_ids = record.get("pretokenized_input_ids")
        labels = record.get("pretokenized_labels")
        if (
            not isinstance(input_ids, list)
            or not isinstance(labels, list)
            or len(input_ids) != len(labels)
            or not input_ids
            or not all(isinstance(value, int) and value >= 0 for value in input_ids)
            or not all(
                isinstance(value, int) and (value == -100 or value >= 0)
                for value in labels
            )
            or all(value == -100 for value in labels)
        ):
            raise ValueError("invalid pretokenized SFT input_ids/labels")
        if len(input_ids) > max_length:
            raise ValueError("pretokenized SFT record exceeds max_length")
        for input_id, label in zip(input_ids, labels, strict=True):
            if label != -100 and label != input_id:
                raise ValueError("pretokenized SFT label differs from its input token")
        return {
            "input_ids": list(input_ids),
            "attention_mask": [1] * len(input_ids),
            "labels": list(labels),
            "weights": float(record["weight"]),
        }
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(record["prompt"], add_special_tokens=False).input_ids
    target_ids = tokenizer(record["target"] + eos, add_special_tokens=False).input_ids
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    if all(label == -100 for label in labels) and target_ids:
        keep = target_ids[-min(len(target_ids), max_length) :]
        input_ids = keep
        labels = keep
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "weights": float(record["weight"]),
    }


class PromptTargetDataset:
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int = 4096):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return encode_prompt_target(self.records[idx], self.tokenizer, max_length=self.max_length)


@dataclass
class WeightedCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "weights": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
            batch["weights"].append(item["weights"])
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "weights": torch.tensor(batch["weights"], dtype=torch.float),
        }


class PairedDataset:
    def __init__(self, domain, retention):
        if len(domain) != len(retention):
            raise ValueError("paired dataset lengths differ")
        self.domain, self.retention = domain, retention

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, index):
        return {"domain": self.domain[index], "retention": self.retention[index], "pair_index": index}


class PairedCollator:
    def __init__(self, pad):
        self.collator = WeightedCollator(pad)

    def __call__(self, rows):
        import torch
        return {**{name: self.collator([r[name] for r in rows]) for name in ("domain", "retention")},
                "pair_index": torch.tensor([r["pair_index"] for r in rows])}
