import json
from types import SimpleNamespace

import pytest
import torch
from transformers import set_seed

from instructreplay.data import PairedDataset, WeightedCollator, encode_prompt_target
from instructreplay.generate import generation_eos_token_id, main as generate_main, prompt_seed, serialize_prompt
from instructreplay.train import load_records, validate_schedule


class Tokenizer:
    eos_token = "!"

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def test_saved_cap_tokens_are_not_retokenized_or_given_eos():
    record = {"prompt": "ignored", "target": "ignored", "weight": 1,
              "pretokenized_input_ids": [1, 2, 3], "pretokenized_labels": [-100, 2, 3]}
    encoded = encode_prompt_target(record, Tokenizer(), max_length=3)
    assert encoded["input_ids"] == [1, 2, 3]
    assert encoded["labels"] == [-100, 2, 3]
    record["pretokenized_labels"][-1] = 4
    with pytest.raises(ValueError, match="differs"):
        encode_prompt_target(record, Tokenizer(), max_length=3)


def test_right_padding_masks_and_pairs():
    rows = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "weights": 1},
            {"input_ids": [1], "attention_mask": [1], "labels": [1], "weights": 1}]
    batch = WeightedCollator(0)(rows)
    assert torch.equal(batch["labels"], torch.tensor([[-100, 2], [1, -100]]))
    assert batch["attention_mask"][1].tolist() == [1, 0]
    with pytest.raises(ValueError, match="lengths"):
        PairedDataset(rows, rows[:1])


def test_public_input_rejects_truncation_and_empty_causal_target(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"input": "123", "output": "45"}) + "\n")
    with pytest.raises(ValueError, match="overlong"):
        load_records(path, Tokenizer(), 4)
    assert len(load_records(path, Tokenizer(), 6)) == 1
    path.write_text(json.dumps({"input": "", "output": "", "pretokenized_input_ids": [1],
                                "pretokenized_labels": [1]}) + "\n")
    with pytest.raises(ValueError, match="empty causal"):
        load_records(path, Tokenizer(), 4)


def test_schedule_preserves_effective_dose_and_rejects_partial_pass():
    config = {"max_steps": 300, "max_length": 2048, "effective_batch_size": 32,
              "micro_batch_size": 1, "kl_coefficient": 0.25, "learning_rate": 1e-5}
    assert validate_schedule(config, world_size=8, domain_count=9600, retention_count=9600) == 4
    assert validate_schedule(config, world_size=4, domain_count=9600, retention_count=9600) == 8
    with pytest.raises(ValueError):
        validate_schedule(config, world_size=3, domain_count=9600, retention_count=9600)
    with pytest.raises(ValueError):
        validate_schedule(config, world_size=8, domain_count=9600, retention_count=9599)


def test_prompt_generation_rejects_publisher_answer_and_seed_is_order_independent():
    with pytest.raises(ValueError):
        serialize_prompt({"messages": [{"role": "assistant", "content": "answer"}]}, Tokenizer())
    assert prompt_seed(42, "model", "instruction") == prompt_seed(42, "model", "instruction")
    assert prompt_seed(42, "model", "instruction") != prompt_seed(42, "other", "instruction")
    assert prompt_seed(42, "model", "instruction") != prompt_seed(42, "model", "other")
    phi = SimpleNamespace(eos_token_id=199999, convert_tokens_to_ids=lambda _: 200020)
    assert generation_eos_token_id(phi) == [200020, 199999]


def test_replay_seed_works_with_transformers_and_invalid_cli_seed_is_rejected(capsys):
    # The earlier 63-bit hash of this prompt was rejected by NumPy inside set_seed.
    seed = prompt_seed(42, "0" * 64, "Explain rain.")
    set_seed(seed)
    first = torch.rand(4)
    set_seed(seed)
    assert torch.equal(torch.rand(4), first)

    with pytest.raises(SystemExit) as error:
        generate_main(["--model", "model", "--model-sha256", "sha", "--prompts", "prompts",
                       "--prompts-sha256", "sha", "--output", "output", "--seed", str(2**32)])
    assert error.value.code == 2
    assert "--seed must be an unsigned 32-bit integer" in capsys.readouterr().err
