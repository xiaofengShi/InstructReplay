import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from instructreplay.data import PairedCollator, PairedDataset
from instructreplay.losses import masked_forward_kl, sequence_ce
from instructreplay.trainer import compare_runtime_models, make_joint_trainer_class, restore_runtime_buffers


def tiny_model():
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config)


def test_original_fp32_rope_buffer_is_restored_after_low_precision_cast():
    teacher = tiny_model().bfloat16()
    teacher.model.rotary_emb.inv_freq = torch.tensor([1.0, 0.1234567])
    student = copy.deepcopy(teacher)
    original = student.model.rotary_emb.inv_freq.clone()
    student.bfloat16()
    assert compare_runtime_models(student, teacher)["buffers"]["differences"]
    restore_runtime_buffers(student, {"model.rotary_emb.inv_freq": original})
    assert not compare_runtime_models(student, teacher)["buffers"]["differences"]
    assert student.model.rotary_emb.original_inv_freq is student.model.rotary_emb.inv_freq


def test_real_cpu_backward_accumulation_reference_isolation_and_save_reload(tmp_path):
    torch.manual_seed(9)
    student = tiny_model()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    manual = copy.deepcopy(student)
    reference_weights = {n: p.clone() for n, p in teacher.named_parameters()}
    # Exercise a nonzero reference loss as well as domain cross-entropy.
    with torch.no_grad():
        student.lm_head.weight.add_(0.01 * torch.randn_like(student.lm_head.weight))
        manual.load_state_dict(student.state_dict())
    domain, replay = [], []
    for i in range(2):
        for rows, offset in ((domain, 0), (replay, 4)):
            ids = [1, 2 + i + offset, 3, 4]
            rows.append({"input_ids": ids, "attention_mask": [1] * 4,
                         "labels": [-100, -100, 3, 4], "weights": 1.0})
    collate = PairedCollator(0)
    dataset = PairedDataset(domain, replay)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer_class = make_joint_trainer_class(teacher)
    trainer = trainer_class(
        model=student, args=TrainingArguments(output_dir=str(tmp_path / "run"), use_cpu=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=1,
        max_grad_norm=0, report_to=[], save_strategy="no", remove_unused_columns=False,
        disable_tqdm=True, seed=9), train_dataset=dataset, data_collator=collate,
        optimizers=(optimizer, scheduler),
    )
    manual.train()
    manual_optimizer = torch.optim.SGD(manual.parameters(), lr=0.01)
    for row in dataset:
        batch = collate([row])
        d, r = batch["domain"], batch["retention"]
        with torch.no_grad():
            ref = teacher(input_ids=r["input_ids"], attention_mask=r["attention_mask"]).logits
        logits = manual(input_ids=r["input_ids"], attention_mask=r["attention_mask"]).logits
        kl, _ = masked_forward_kl(teacher_logits=ref, student_logits=logits,
                                  labels=r["labels"], weights=r["weights"])
        ce = sequence_ce(manual(input_ids=d["input_ids"], attention_mask=d["attention_mask"]).logits, d["labels"])
        ((ce + 0.25 * kl) / 2).backward()
    manual_optimizer.step()
    trainer.train()
    assert trainer.state.global_step == 1
    for a, e in zip(student.parameters(), manual.parameters()):
        assert torch.allclose(a, e, atol=2e-8, rtol=2e-6)
    assert all(p.grad is None and torch.equal(p, reference_weights[n]) for n, p in teacher.named_parameters())
    trainer.save_model(str(tmp_path / "saved"))
    restored = LlamaForCausalLM.from_pretrained(tmp_path / "saved", attn_implementation="sdpa")
    student.eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(student(ids).logits, restored(ids).logits)
