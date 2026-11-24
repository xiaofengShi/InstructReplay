"""Frozen-reference Trainer and pre-forward model identity checks."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence
from .losses import M0SoftKLTrainingError, masked_forward_kl
from .io import canonical_bytes
from .losses import sequence_ce

def compare_runtime_models(student: Any, teacher: Any) -> dict[str, Any]:
    """Content-free exact audit, including nonpersistent buffers absent from saves."""
    import torch

    def compare_tensors(left: Mapping, right: Mapping) -> dict:
        differences = []
        for name in sorted(left.keys() | right.keys()):
            a, b = left.get(name), right.get(name)
            if a is None or b is None:
                differences.append({"name": name, "missing": "student" if a is None else "teacher"})
                continue
            equal = a.shape == b.shape and torch.equal(a.detach(), b.detach().to(a.device))
            if not equal or a.dtype != b.dtype:
                differences.append({
                    "name": name, "student_dtype": str(a.dtype), "teacher_dtype": str(b.dtype),
                    "student_shape": list(a.shape), "teacher_shape": list(b.shape),
                    "values_equal": equal,
                    "max_abs_delta": float((a.detach().float() - b.detach().to(a.device).float()).abs().max()) if a.shape == b.shape else None,
                })
        return {"student_count": len(left), "teacher_count": len(right), "differences": differences}

    def aliases(model: Any) -> list[list[str]]:
        groups: dict[int, list[str]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            groups.setdefault(id(parameter), []).append(name)
        return sorted(sorted(names) for names in groups.values() if len(names) > 1)

    a_config, b_config = student.config.to_dict(), teacher.config.to_dict()
    a_modules = {name: f"{type(module).__module__}.{type(module).__qualname__}" for name, module in student.named_modules()}
    b_modules = {name: f"{type(module).__module__}.{type(module).__qualname__}" for name, module in teacher.named_modules()}
    return {
        "parameters": compare_tensors(dict(student.named_parameters()), dict(teacher.named_parameters())),
        "buffers": compare_tensors(dict(student.named_buffers()), dict(teacher.named_buffers())),
        "student_tied_parameter_groups": aliases(student), "teacher_tied_parameter_groups": aliases(teacher),
        "cross_model_shared_parameters": sorted(name for name, p in student.named_parameters() if any(p is q for q in teacher.parameters())),
        "config_differences": {key: {"student": a_config.get(key), "teacher": b_config.get(key)} for key in sorted(a_config.keys() | b_config.keys()) if a_config.get(key) != b_config.get(key)},
        "module_type_differences": {key: {"student": a_modules.get(key), "teacher": b_modules.get(key)} for key in sorted(a_modules.keys() | b_modules.keys()) if a_modules.get(key) != b_modules.get(key)},
        "student_attention_implementation": student.config._attn_implementation,
        "teacher_attention_implementation": teacher.config._attn_implementation,
        "student_training": student.training, "teacher_training": teacher.training,
    }


def restore_runtime_buffers(model: Any, snapshots: Mapping[str, Any]) -> None:
    """Restore explicitly selected original buffers, never rounded upcasts."""
    for name, original in snapshots.items():
        module_name, attribute = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        current = module.get_buffer(attribute)
        if current.shape != original.shape:
            raise M0SoftKLTrainingError(f"buffer shape changed: {name}")
        restored = original.to(device=current.device).clone()
        setattr(module, attribute, restored)
        if attribute == "inv_freq" and hasattr(module, "original_inv_freq"):
            module.original_inv_freq = restored


def runtime_identity_errors(audit: Mapping[str, Any], allowed_config_keys: Sequence[str] = ()) -> list[str]:
    """Fail closed on the complete pre-forward identity contract, not weights alone."""
    after = audit.get("after_buffer_restore", audit)
    errors = []
    for field in ("parameters", "buffers"):
        if after[field]["differences"]:
            errors.append(field)
    for field in ("parameter_binding_changed_names", "optimizer_unbound_trainable_parameters"):
        if audit[field]:
            errors.append(field)
    for field in ("module_type_differences", "cross_model_shared_parameters"):
        if after[field]:
            errors.append(field)
    if after["student_tied_parameter_groups"] != after["teacher_tied_parameter_groups"]:
        errors.append("tying")
    if not (after["student_attention_implementation"] == after["teacher_attention_implementation"] == "sdpa"):
        errors.append("attention_backend")
    if set(after["config_differences"]) - set(allowed_config_keys):
        errors.append("unapproved_config_difference")
    if not audit["optimizer_learning_rates"] or any(not math.isfinite(x) or x <= 0 for x in audit["optimizer_learning_rates"]):
        errors.append("optimizer_learning_rate")
    return errors


def make_soft_kl_trainer_class(
    teacher_model: Any, *, student_forward_mode: str = "deepspeed_engine",
    audit_before_forward: bool = False,
    preserve_buffer_names: Sequence[str] = (),
    measure_backward: bool = False,
    strict_runtime_identity: bool = False,
    allowed_config_difference_keys: Sequence[str] = (),
):
    import torch
    from transformers import Trainer

    if (preserve_buffer_names or measure_backward) and not audit_before_forward:
        raise M0SoftKLTrainingError("buffer preservation and backward measurement require runtime audit")
    if student_forward_mode not in {"deepspeed_engine", "deepspeed_unwrapped_module"}:
        raise M0SoftKLTrainingError(
            f"unsupported student_forward_mode: {student_forward_mode}"
        )

    class M0SoftKLTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            self.teacher_model = teacher_model
            self.student_forward_mode = student_forward_mode
            self.observed_student_wrapper_type = ""
            self.teacher_model.eval()
            self.kl_count = 0
            self.kl_sum = 0.0
            self.kl_min = math.inf
            self.kl_max = -math.inf
            self.kl_max_abs_token = 0.0
            self.kl_max_abs_probability_delta = 0.0
            self.kl_valid_tokens = 0
            self.initial_parameter_ids = {name: id(p) for name, p in self.model.named_parameters()}
            self.runtime_audit = None
            self.buffer_snapshots = {name: self.model.get_buffer(name).detach().clone() for name in preserve_buffer_names}
            self.gradient_max = torch.zeros((), device=self.accelerator.device)
            self.gradient_hook_calls = 0
            self.minimum_observed_lr = math.inf
            self.gradient_hooks = []

        def record_gradient(self, gradient: Any) -> None:
            self.gradient_max.copy_(torch.maximum(self.gradient_max, gradient.detach().abs().max().float()))
            self.gradient_hook_calls += 1

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs.pop("labels")
            weights = inputs.pop("weights")
            if audit_before_forward and self.runtime_audit is None:
                managed = getattr(model, "module", model)
                self.runtime_audit = compare_runtime_models(managed, self.teacher_model)
                self.runtime_audit["parameter_binding_changed_names"] = [name for name, p in managed.named_parameters() if self.initial_parameter_ids.get(name) != id(p)]
                self.runtime_audit["optimizer_learning_rates"] = [float(group["lr"]) for group in self.optimizer.param_groups]
                optimizer = model.optimizer
                low_precision_groups = getattr(optimizer, "bit16_groups", [])
                bound_ids = {id(p) for group in low_precision_groups for p in group}
                self.runtime_audit["optimizer_unbound_trainable_parameters"] = [name for name, p in managed.named_parameters() if p.requires_grad and id(p) not in bound_ids]
                if self.buffer_snapshots:
                    restore_runtime_buffers(managed, self.buffer_snapshots)
                    self.runtime_audit["after_buffer_restore"] = compare_runtime_models(managed, self.teacher_model)
                    after = self.runtime_audit["after_buffer_restore"]
                    if not strict_runtime_identity and (after["parameters"]["differences"] or after["buffers"]["differences"] or after["module_type_differences"]):
                        raise M0SoftKLTrainingError("runtime values still differ after restoring original buffers")
                audit_path = Path(self.args.output_dir).parent / f"runtime-before-forward-rank{self.args.process_index}.json"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("xb") as stream:
                    stream.write(canonical_bytes(self.runtime_audit))
                if strict_runtime_identity:
                    errors = runtime_identity_errors(self.runtime_audit, allowed_config_difference_keys)
                    failed = torch.tensor(int(bool(errors)), device=self.accelerator.device)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
                    if failed.item():
                        raise M0SoftKLTrainingError(f"pre-forward runtime identity failed on at least one rank: {errors}")
                if measure_backward:
                    self.gradient_hooks = [p.register_hook(self.record_gradient) for p in managed.parameters() if p.requires_grad]
            if measure_backward:
                self.minimum_observed_lr = min(self.minimum_observed_lr, *(float(group["lr"]) for group in self.optimizer.param_groups))
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs, use_cache=False)
            self.observed_student_wrapper_type = (
                f"{type(model).__module__}.{type(model).__qualname__}"
            )
            student_forward_model = model
            if self.student_forward_mode == "deepspeed_unwrapped_module":
                student_forward_model = getattr(model, "module", None)
                if student_forward_model is None:
                    raise M0SoftKLTrainingError(
                        "deepspeed_unwrapped_module requires a wrapped student model"
                    )
            student_outputs = student_forward_model(**inputs, use_cache=False)
            loss, stats = masked_forward_kl(
                teacher_logits=teacher_outputs.logits,
                student_logits=student_outputs.logits,
                labels=labels,
                weights=weights,
            )
            self.kl_count += 1
            self.kl_sum += stats["loss"]
            self.kl_min = min(self.kl_min, stats["loss"])
            self.kl_max = max(self.kl_max, stats["loss"])
            self.kl_max_abs_token = max(
                self.kl_max_abs_token, stats["max_abs_token_kl"]
            )
            self.kl_max_abs_probability_delta = max(
                self.kl_max_abs_probability_delta,
                stats["max_abs_probability_delta"],
            )
            self.kl_valid_tokens += stats["valid_tokens"]
            return (loss, student_outputs) if return_outputs else loss

    return M0SoftKLTrainer


def make_joint_trainer_class(teacher_model, *, coefficient=0.25, **identity_options):
    import torch
    from torch.utils.data import SequentialSampler

    base = make_soft_kl_trainer_class(teacher_model, **identity_options)

    class InstructReplayTrainer(base):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            kl = super().compute_loss(model, dict(inputs["retention"]))
            domain = dict(inputs["domain"])
            labels = domain.pop("labels")
            domain.pop("weights")
            outputs = model(**domain, use_cache=False)
            ce = sequence_ce(outputs.logits, labels)
            loss = ce + coefficient * kl
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite domain/replay loss")
            return (loss, outputs) if return_outputs else loss

    return InstructReplayTrainer
