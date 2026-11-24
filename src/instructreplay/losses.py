"""Response-masked full-vocabulary forward KL with an exact identity fixed point."""
from __future__ import annotations
from typing import Any

class M0SoftKLTrainingError(ValueError):
    """Raised when a frozen identity or training invariant differs."""


def masked_forward_kl(
    *,
    teacher_logits: Any,
    student_logits: Any,
    labels: Any,
    weights: Any,
) -> tuple[Any, dict[str, Any]]:
    """Forward KL on next-token positions selected by the exact response mask."""

    import torch
    import torch.nn.functional as functional

    if teacher_logits.shape != student_logits.shape:
        raise M0SoftKLTrainingError("teacher/student logits must have identical shape")
    if labels.shape != teacher_logits.shape[:2]:
        raise M0SoftKLTrainingError("labels must match logits batch and sequence dimensions")
    if weights.ndim != 1 or weights.shape[0] != labels.shape[0]:
        raise M0SoftKLTrainingError("weights must have one scalar per sequence")
    teacher_logp = functional.log_softmax(teacher_logits[..., :-1, :].float(), dim=-1)
    student_logp = functional.log_softmax(student_logits[..., :-1, :].float(), dim=-1)
    teacher_p = teacher_logp.exp()
    student_p = student_logp.exp()
    kl_value = torch.sum(teacher_p * (teacher_logp - student_logp), dim=-1)
    probability_delta = student_p - teacher_p
    # The mathematically equivalent direct autograd expression can leave a tiny
    # non-zero gradient at p_student == p_teacher because sum(p_teacher) is not
    # represented as exactly one. Adam can amplify that numerical residue.  Use
    # the exact analytical forward-KL gradient q - p while retaining the exact
    # forward KL value. This makes the identity fixed point exact in floating
    # point whenever the separately evaluated logits are identical.
    gradient_proxy = torch.sum(
        probability_delta.detach() * student_logits[..., :-1, :].float(),
        dim=-1,
    )
    per_token = gradient_proxy - gradient_proxy.detach() + kl_value.detach()
    mask = labels[..., 1:].ne(-100)
    token_counts = mask.sum(dim=1)
    if bool(token_counts.eq(0).any().item()):
        raise M0SoftKLTrainingError("every soft-KL sequence requires a response token")
    sequence_kl = (per_token * mask).sum(dim=1) / token_counts
    loss = (sequence_kl * weights.to(sequence_kl.device)).mean()
    selected = per_token[mask]
    stats = {
        "valid_tokens": int(mask.sum().item()),
        "loss": float(loss.detach().item()),
        "min_token_kl": float(selected.min().detach().item()),
        "max_token_kl": float(selected.max().detach().item()),
        "max_abs_token_kl": float(selected.abs().max().detach().item()),
        "max_abs_probability_delta": float(
            probability_delta.detach().abs().amax(dim=-1)[mask].max().item()
        ),
    }
    return loss, stats


def sequence_ce(logits, labels):
    import torch.nn.functional as F
    target = labels[:, 1:]
    token_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                target.reshape(-1), ignore_index=-100, reduction="none").reshape(target.shape)
    mask = target.ne(-100)
    if bool(mask.sum(1).eq(0).any()):
        raise ValueError("empty Domain target")
    return ((token_loss * mask).sum(1) / mask.sum(1)).mean()
