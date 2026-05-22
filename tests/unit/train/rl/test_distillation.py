import torch

from prime_rl.trainer.rl.distillation import compute_reverse_kl_terms


def test_reverse_kl_full_is_zero_for_identical_logits():
    logits = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 0.25, 0.0]]])
    labels = torch.tensor([[0, 2]])

    terms = compute_reverse_kl_terms(
        student_logits=logits,
        teacher_logits=logits,
        labels=labels,
        estimator="rb_full",
        top_k=3,
    )

    assert torch.allclose(terms.kl, torch.zeros_like(terms.kl), atol=1e-6)
    assert torch.allclose(
        terms.selected_k1,
        torch.zeros_like(terms.selected_k1),
        atol=1e-6,
    )


def test_reverse_kl_topk_matches_full_when_k_covers_vocab():
    student = torch.tensor([[[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]]])
    teacher = torch.tensor([[[0.0, 1.0, -0.5], [0.3, -0.2, 0.4]]])
    labels = torch.tensor([[0, 1]])

    full = compute_reverse_kl_terms(
        student_logits=student,
        teacher_logits=teacher,
        labels=labels,
        estimator="rb_full",
        top_k=3,
    )
    topk = compute_reverse_kl_terms(
        student_logits=student,
        teacher_logits=teacher,
        labels=labels,
        estimator="rb_topk",
        top_k=3,
    )

    assert torch.allclose(topk.kl, full.kl, atol=1e-6)
    assert torch.allclose(topk.topk_mass, torch.ones_like(topk.kl), atol=1e-6)


def test_k1_sample_is_selected_logprob_difference():
    student = torch.tensor([[[2.0, 0.0, -1.0]]])
    teacher = torch.tensor([[[0.0, 1.0, -0.5]]])
    labels = torch.tensor([[1]])

    terms = compute_reverse_kl_terms(
        student_logits=student,
        teacher_logits=teacher,
        labels=labels,
        estimator="k1_sample",
        top_k=1,
    )
    expected = (
        torch.log_softmax(student, dim=-1)[..., 1]
        - torch.log_softmax(teacher, dim=-1)[..., 1]
    )

    assert torch.allclose(terms.kl, expected)
    assert torch.allclose(terms.selected_k1, expected)
