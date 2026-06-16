from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


DistillationEstimator = Literal["rb_full", "rb_topk", "k1_sample"]


@dataclass
class ReverseKLDistillationTerms:
    kl: Tensor
    selected_k1: Tensor
    topk_mass: Tensor | None = None


def compute_reverse_kl_terms(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    estimator: DistillationEstimator,
    top_k: int,
) -> ReverseKLDistillationTerms:
    """Compute per-token reverse-KL terms, KL(student || teacher).

    `k1_sample` uses only the sampled label logprob difference. The RB modes
    use the student distribution to Rao-Blackwellize over vocabulary support;
    `rb_topk` truncates that support to the top-k student tokens for memory.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if labels.shape != student_logits.shape[:-1]:
        raise ValueError(
            "labels shape must match logits without vocab dimension, got "
            f"labels={tuple(labels.shape)} logits={tuple(student_logits.shape)}"
        )

    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    selected_student = torch.gather(
        student_logp, dim=-1, index=labels.unsqueeze(-1),
    ).squeeze(-1)
    selected_teacher = torch.gather(
        teacher_logp, dim=-1, index=labels.unsqueeze(-1),
    ).squeeze(-1)
    selected_k1 = selected_student - selected_teacher

    if estimator == "k1_sample":
        return ReverseKLDistillationTerms(kl=selected_k1, selected_k1=selected_k1)

    vocab_size = student_logits.shape[-1]
    if estimator == "rb_full" or top_k >= vocab_size:
        student_prob = student_logp.exp()
        kl = (student_prob * (student_logp - teacher_logp)).sum(dim=-1)
        return ReverseKLDistillationTerms(
            kl=kl,
            selected_k1=selected_k1,
            topk_mass=torch.ones_like(kl),
        )

    if estimator != "rb_topk":
        raise ValueError(f"unknown distillation estimator: {estimator!r}")

    k = max(1, min(int(top_k), vocab_size))
    top_logp, top_idx = torch.topk(student_logp, k=k, dim=-1)
    top_teacher_logp = torch.gather(teacher_logp, dim=-1, index=top_idx)
    top_prob = top_logp.exp()
    kl = (top_prob * (top_logp - top_teacher_logp)).sum(dim=-1)
    return ReverseKLDistillationTerms(
        kl=kl,
        selected_k1=selected_k1,
        topk_mass=top_prob.sum(dim=-1),
    )
