import pytest
import torch
import torch.nn.functional as F

from instructreplay.losses import masked_forward_kl, sequence_ce


def test_nonzero_kl_and_parameter_gradient_match_independent_fp64_oracle():
    torch.manual_seed(4)
    layer = torch.nn.Linear(5, 7).double()
    features = torch.randn(2, 5, 5, dtype=torch.float64)
    teacher = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 1], [-100, -100, -100, -100, 3]])
    weights = torch.tensor([1.0, 0.5])
    student = layer(features)
    actual, _ = masked_forward_kl(teacher_logits=teacher, student_logits=student, labels=labels, weights=weights)
    p = teacher[:, :-1].detach().softmax(-1)
    token_kl = (p * (teacher[:, :-1].detach().log_softmax(-1) - student[:, :-1].log_softmax(-1))).sum(-1)
    mask = labels[:, 1:] != -100
    expected = (((token_kl * mask).sum(1) / mask.sum(1)) * weights).mean()
    ga = torch.autograd.grad(actual, tuple(layer.parameters()), retain_graph=True)
    ge = torch.autograd.grad(expected, tuple(layer.parameters()))
    assert actual.item() == pytest.approx(expected.item(), rel=1e-6)
    for a, e in zip(ga, ge):
        assert torch.allclose(a, e, rtol=2e-5, atol=1e-7)
    assert teacher.grad is None


def test_exact_reference_fixed_point_has_zero_value_and_zero_gradient():
    torch.manual_seed(7)
    student = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, -100, -100, 2, -100]])
    loss, stats = masked_forward_kl(teacher_logits=student.detach().clone(), student_logits=student,
                                   labels=labels, weights=torch.ones(2))
    loss.backward()
    assert loss.item() == stats["max_abs_token_kl"] == 0.0
    assert torch.count_nonzero(student.grad) == 0


def test_prompt_padding_and_final_position_do_not_contribute():
    teacher = torch.zeros(1, 4, 5)
    student = torch.randn(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, -100]])
    loss, _ = masked_forward_kl(teacher_logits=teacher, student_logits=student,
                              labels=labels, weights=torch.ones(1))
    loss.backward()
    assert torch.count_nonzero(student.grad[:, [0, 2, 3]]) == 0
    assert torch.count_nonzero(student.grad[:, 1]) > 0
    with pytest.raises(ValueError, match="response token"):
        masked_forward_kl(teacher_logits=teacher, student_logits=student,
                          labels=torch.full_like(labels, -100), weights=torch.ones(1))


def test_ce_weights_sequences_equally_and_uses_next_token_shift():
    torch.manual_seed(11)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, -100, -100, -100, 5]])
    expected = (F.cross_entropy(logits[0, :4], labels[0, 1:])
                + F.cross_entropy(logits[1, 3:4], labels[1, 4:5])) / 2
    assert torch.allclose(sequence_ce(logits, labels), expected)
