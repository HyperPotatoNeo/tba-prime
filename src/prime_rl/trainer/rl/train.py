import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

from contextlib import nullcontext
import gc
import math
import os
import time

# D5 debug: memory trace at every micro-batch boundary. Guarded by env
# var so non-debug runs are unaffected. When KVE_MEM_TRACE=1, each
# micro-batch logs its path, pre/post-forward/backward memory_allocated
# and max_memory_allocated, so we can see exactly where memory grows
# across the compaction -> text transition on step 1 (see D5
# investigation notes in plans/phase3_training_integration.md).
_KVE_MEM_TRACE = os.environ.get("KVE_MEM_TRACE") == "1"
_KVE_STACK_TRACE = os.environ.get("KVE_STACK_TRACE") == "1"


def _mem_snap(tag: str) -> str:
    if not _KVE_MEM_TRACE:
        return ""
    import torch as _t
    alloc = _t.cuda.memory_allocated() / 1e9
    peak = _t.cuda.max_memory_allocated() / 1e9
    reserved = _t.cuda.memory_reserved() / 1e9
    try:
        import psutil as _psutil

        rss = _psutil.Process(os.getpid()).memory_info().rss / 1e9
        rss_part = f" rss={rss:.3f}GB"
    except Exception:
        rss_part = ""
    return f"[MEM] {tag} alloc={alloc:.3f}GB peak={peak:.3f}GB reserved={reserved:.3f}GB{rss_part}"


def _trace_snap(tag: str) -> str:
    if _KVE_MEM_TRACE:
        return _mem_snap(tag)
    return f"[STACK] {tag}"


def _pad_ratio(useful_tokens: int, padded_tokens: int) -> float:
    if useful_tokens <= 0:
        return 1.0
    return padded_tokens / useful_tokens


def _stack_trace_summary(
    micro_batch_group,
    micro_batch,
    *,
    flex_compaction_group: bool,
) -> str:
    full_lens = [int(mb["input_ids"].shape[1]) for mb in micro_batch_group]
    batch_rows = int(micro_batch["input_ids"].shape[0])
    batch_seq_len = int(micro_batch["input_ids"].shape[1])
    full_useful = sum(full_lens)
    full_padded = batch_rows * batch_seq_len
    try:
        loss_tokens = int(micro_batch["loss_mask"].sum().item())
    except Exception:
        loss_tokens = -1

    parts = [
        f"full_lens={full_lens}",
        f"full_tokens={full_useful}/{full_padded}",
        f"full_pad={_pad_ratio(full_useful, full_padded):.3f}x",
        f"loss_tokens={loss_tokens}",
    ]
    if micro_batch.get("flex_stack_mode") is not None:
        parts.append(f"flex_stack_mode={micro_batch.get('flex_stack_mode')}")

    writer_len_fn = globals().get("compute_flex_mask_writer_len")
    if flex_compaction_group and writer_len_fn is not None:
        writer_lens = []
        for mb in micro_batch_group:
            calls = mb.get("calls")
            if calls is None:
                writer_lens = []
                break
            writer_lens.append(int(writer_len_fn(calls)))
        if writer_lens:
            writer_useful = sum(writer_lens)
            if micro_batch.get("flex_stack_mode") == "horizontal":
                writer_padded = writer_useful
            else:
                writer_padded = len(writer_lens) * max(writer_lens)
            parts.extend(
                [
                    f"writer_lens={writer_lens}",
                    f"writer_tokens={writer_useful}/{writer_padded}",
                    f"writer_pad={_pad_ratio(writer_useful, writer_padded):.3f}x",
                ]
            )

    return " " + " ".join(parts)
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.trainer.models.layers.attn import substitute_ring_attn
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.multi_ckpt import setup_multi_checkpoint_manager
from prime_rl.trainer.optim import setup_optimizer, setup_multi_optimizer
from prime_rl.trainer.scheduler import setup_scheduler, setup_multi_scheduler
from prime_rl.configs.trainer import DefaultLossConfig, TrainerConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader
from prime_rl.trainer.rl.microbatch_stacking import (
    compaction_metric_events_by_row,
    is_flex_compaction_stackable_micro_batch,
    make_micro_batch_groups,
    pack_horizontal_flex_compaction_micro_batches,
    split_packed_batch,
    stack_flex_compaction_micro_batches,
    stack_standard_micro_batches,
)
from prime_rl.utils.cp import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    setup_cp_params,
    shard_for_cp,
)
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    setup_loss_fn,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.rl.distillation import compute_reverse_kl_terms
from prime_rl.trainer.model import (
    forward,
    setup_tokenizer,
    setup_model,
    is_tt_moe_model,
    get_load_balance_stats,
)

try:
    from kv_eviction.segmented_forward import (
        _build_pre_trim_plan,
        batched_flex_mask_segmented_forward,
        compute_flex_mask_writer_len,
        compute_num_segments,
        compute_per_call_bptt_window_forward_counts,
        compute_num_per_call_forwards,
        flex_mask_segmented_forward,
        per_call_segmented_forward,
        packed_flex_mask_segmented_forward,
        selected_logprob_batched_flex_mask_segmented_forward,
        selected_logprob_flex_mask_segmented_forward,
        selected_logprob_packed_flex_mask_segmented_forward,
        segmented_forward,
    )
except ImportError:
    # kv_eviction is optional; only required when TrainerConfig.compaction
    # is enabled and TrainingSamples carry compaction_events. Import failure
    # is tolerated at module load; the assertion at the dispatch site will
    # raise with a clearer error if compaction is actually used.
    segmented_forward = None  # type: ignore[assignment]
    _build_pre_trim_plan = None  # type: ignore[assignment]
    batched_flex_mask_segmented_forward = None  # type: ignore[assignment]
    compute_flex_mask_writer_len = None  # type: ignore[assignment]
    compute_num_segments = None  # type: ignore[assignment]
    compute_per_call_bptt_window_forward_counts = None  # type: ignore[assignment]
    flex_mask_segmented_forward = None  # type: ignore[assignment]
    per_call_segmented_forward = None  # type: ignore[assignment]
    packed_flex_mask_segmented_forward = None  # type: ignore[assignment]
    selected_logprob_batched_flex_mask_segmented_forward = None  # type: ignore[assignment]
    selected_logprob_flex_mask_segmented_forward = None  # type: ignore[assignment]
    selected_logprob_packed_flex_mask_segmented_forward = None  # type: ignore[assignment]
    compute_num_per_call_forwards = None  # type: ignore[assignment]
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    GarbageCollection,
    MemoryProfiler,
    Tensors,
    export_benchmark_json,
    filter_rl_trainer_tensor_stats_for_wandb,
    get_zero_gradient_ratio,
    get_ckpt_disk_metrics,
    setup_torch_distributed,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.trainer.runs import setup_multi_run_manager, Progress, get_multi_run_manager
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.metrics_server import HealthServer, MetricsServer, RunStats
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit, resolve_latest_ckpt_step, to_col_format
from ring_flash_attn import substitute_hf_flash_attn
from torchtitan.distributed.utils import clip_grad_norm_


def _get_dp_max_micro_batch_seq_lens(
    micro_batches,
    *,
    dp_group,
    dp_world_size: int,
    compaction_enabled: bool = False,
    flex_compaction_enabled: bool = False,
    cp_enabled: bool = False,
    lora_enabled: bool = False,
    multi_run_enabled: bool = False,
    flex_compaction_stackable: list[bool] | None = None,
) -> list[int]:
    """Return per-index sequence lengths maximized across data-parallel ranks."""
    seq_lens = []
    for idx, micro_batch in enumerate(micro_batches):
        seq_len = int(micro_batch["input_ids"].shape[1])
        globally_flex_stackable = (
            True
            if flex_compaction_stackable is None
            else bool(flex_compaction_stackable[idx])
        )
        if (
            globally_flex_stackable
            and compute_flex_mask_writer_len is not None
            and is_flex_compaction_stackable_micro_batch(
                micro_batch,
                compaction_enabled=compaction_enabled,
                flex_compaction_enabled=flex_compaction_enabled,
                cp_enabled=cp_enabled,
                lora_enabled=lora_enabled,
                multi_run_enabled=multi_run_enabled,
            )
        ):
            seq_len = int(compute_flex_mask_writer_len(micro_batch["calls"]))
        seq_lens.append(seq_len)
    if dp_world_size <= 1:
        return seq_lens

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    local_count = torch.tensor([len(seq_lens)], dtype=torch.int64, device=device)
    min_count = local_count.clone()
    max_count = local_count.clone()
    dist.all_reduce(min_count, op=dist.ReduceOp.MIN, group=dp_group)
    dist.all_reduce(max_count, op=dist.ReduceOp.MAX, group=dp_group)
    if int(min_count.item()) != int(max_count.item()):
        raise RuntimeError(
            "micro-batch stack token budget requires all data-parallel ranks "
            "to have the same number of local micro-batches, got "
            f"min={int(min_count.item())} max={int(max_count.item())}"
        )

    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    dist.all_reduce(seq_lens_tensor, op=dist.ReduceOp.MAX, group=dp_group)
    return [int(x) for x in seq_lens_tensor.cpu().tolist()]


def _flatten_nested_list(items):
    flat = []
    for item in items or []:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _get_dp_flex_compaction_stackable(
    micro_batches,
    *,
    dp_group,
    dp_world_size: int,
    compaction_enabled: bool,
    flex_compaction_enabled: bool,
    cp_enabled: bool,
    lora_enabled: bool,
    multi_run_enabled: bool,
) -> list[bool]:
    """Return per-index flex-compaction stackability agreed across DP ranks."""
    local_flags = [
        int(
            is_flex_compaction_stackable_micro_batch(
                micro_batch,
                compaction_enabled=compaction_enabled,
                flex_compaction_enabled=flex_compaction_enabled,
                cp_enabled=cp_enabled,
                lora_enabled=lora_enabled,
                multi_run_enabled=multi_run_enabled,
            )
        )
        for micro_batch in micro_batches
    ]
    if dp_world_size <= 1:
        return [bool(flag) for flag in local_flags]

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    local_count = torch.tensor([len(local_flags)], dtype=torch.int64, device=device)
    min_count = local_count.clone()
    max_count = local_count.clone()
    dist.all_reduce(min_count, op=dist.ReduceOp.MIN, group=dp_group)
    dist.all_reduce(max_count, op=dist.ReduceOp.MAX, group=dp_group)
    if int(min_count.item()) != int(max_count.item()):
        raise RuntimeError(
            "flex compaction stacking requires all data-parallel ranks to "
            "have the same number of local micro-batches, got "
            f"min={int(min_count.item())} max={int(max_count.item())}"
        )

    flags = torch.tensor(local_flags, dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=dp_group)
    return [bool(int(flag)) for flag in flags.cpu().tolist()]


def _kve_top_mismatch_n() -> int:
    raw = os.environ.get("KVE_TOP_MISMATCH_TOKENS", "")
    if raw == "":
        raw = os.environ.get("KVE_LOG_TOP_MISMATCH_TOKENS", "")
    if raw == "":
        return 0
    if raw.lower() in {"1", "true", "yes", "on"}:
        return 20
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _kve_call_mismatch_n() -> int:
    raw = os.environ.get("KVE_CALL_MISMATCH_TOP", "")
    if raw == "":
        raw = os.environ.get("KVE_LOG_CALL_MISMATCH_SUMMARY", "")
    if raw == "":
        return 0
    if raw.lower() in {"1", "true", "yes", "on"}:
        return 20
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _kve_decode_one(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        try:
            text = tokenizer.convert_ids_to_tokens([token_id])[0]
        except Exception:
            text = ""
    return repr(text).replace("\n", "\\n")


def _kve_find_call_plan(call_plans, full_logit_start: int, full_logit_end: int):
    if not call_plans:
        return None
    for plan in call_plans:
        if (
            int(plan.get("post_start", -1)) == full_logit_start
            and int(plan.get("post_end", -1)) == full_logit_end
        ):
            return plan
    for plan in call_plans:
        start = int(plan.get("post_start", -1))
        end = int(plan.get("post_end", -1))
        if start <= full_logit_start and full_logit_end <= end:
            return plan
    return None


def _kve_collect_top_mismatch_records(
    *,
    records: list[dict],
    counts: dict[str, int],
    top_n: int,
    scope: str,
    step: int,
    micro_step: int,
    full_logit_start: int,
    full_logit_end: int,
    tgt_start: int,
    tgt_end: int,
    trainer_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    entropy: torch.Tensor | None,
    loss_config,
    call_plan,
) -> None:
    if top_n <= 0:
        return
    with torch.no_grad():
        t = trainer_logprobs.detach().float().squeeze(0)
        inf = inference_logprobs.detach().float().squeeze(0)
        adv = advantages.detach().float().squeeze(0)
        m_loss = loss_mask.detach().bool().squeeze(0)
        lbl = labels.detach().squeeze(0)
        diff = t - inf
        prob_diff = torch.exp(t) - torch.exp(inf)
        if isinstance(loss_config, DefaultLossConfig):
            dppo_high = prob_diff > loss_config.dppo_mask_high
            dppo_low = prob_diff < -loss_config.dppo_mask_low
            dppo_mask = torch.where(adv > 0, dppo_high, dppo_low)
        else:
            dppo_mask = torch.zeros_like(m_loss)
        keep_mask = m_loss & ~dppo_mask
        normalized_scope = scope.lower()
        if normalized_scope in {"loss", "loss_mask", "trainable", "all"}:
            select = m_loss
        elif normalized_scope in {"masked", "dppo_masked"}:
            select = m_loss & dppo_mask
        else:
            select = keep_mask
            normalized_scope = "keep"
        counts["considered"] = counts.get("considered", 0) + int(select.sum().item())
        if not select.any():
            return
        mismatch_kl = torch.exp(diff) - diff - 1.0
        idx_all = torch.nonzero(select, as_tuple=False).flatten()
        vals = mismatch_kl[idx_all]
        k = min(top_n, int(vals.numel()))
        top_vals, top_pos = torch.topk(vals, k=k)
        idx = idx_all[top_pos]
        ent = entropy.detach().float().squeeze(0) if entropy is not None else None
        call_idx = call_plan.get("call_idx") if call_plan is not None else None
        for rank, (local_i, kl_val) in enumerate(zip(idx.tolist(), top_vals.tolist()), start=1):
            token_id = int(lbl[local_i].detach().cpu().item())
            logit_pos = full_logit_start + int(local_i)
            target_pos = tgt_start + int(local_i)
            rec = {
                "step": int(step),
                "micro_step": int(micro_step),
                "rank_in_segment": rank,
                "kl": float(kl_val),
                "diff": float(diff[local_i].detach().cpu().item()),
                "trainer_lp": float(t[local_i].detach().cpu().item()),
                "inference_lp": float(inf[local_i].detach().cpu().item()),
                "prob_diff": float(prob_diff[local_i].detach().cpu().item()),
                "advantage": float(adv[local_i].detach().cpu().item()),
                "entropy": (
                    float(ent[local_i].detach().cpu().item())
                    if ent is not None else None
                ),
                "token_id": token_id,
                "target_pos": target_pos,
                "logit_pos": logit_pos,
                "tgt_start": int(tgt_start),
                "tgt_end": int(tgt_end),
                "logit_start": int(full_logit_start),
                "logit_end": int(full_logit_end),
                "scope": normalized_scope,
                "dppo_masked": bool(dppo_mask[local_i].detach().cpu().item()),
                "keep": bool(keep_mask[local_i].detach().cpu().item()),
                "call_idx": int(call_idx) if call_idx is not None else None,
            }
            if call_plan is not None:
                rec.update({
                    "has_admission": bool(call_plan.get("has_admission", False)),
                    "sub_len": call_plan.get("sub_len"),
                    "comp_len": call_plan.get("comp_len"),
                    "pad_len": call_plan.get("pad_len"),
                    "nuf_len": call_plan.get("nuf_len"),
                    "writer_offset": call_plan.get("writer_offset"),
                    "admission_offset_after": call_plan.get("admission_offset_after"),
                    "admission_total_evicted": call_plan.get("admission_total_evicted"),
                    "admission_nuf_len": call_plan.get("admission_nuf_len"),
                    "synthetic_cached_tokens": call_plan.get("synthetic_cached_tokens"),
                    "synthetic_nuf_len": call_plan.get("synthetic_nuf_len"),
                    "synthetic_offset_after": call_plan.get("synthetic_offset_after"),
                    "new_content_start_in_sub": call_plan.get("new_content_start_in_sub"),
                })
            records.append(rec)


def _kve_log_top_mismatch_records(logger, tokenizer, records: list[dict], counts: dict[str, int], top_n: int) -> None:
    if top_n <= 0 or not records:
        return
    rows = sorted(records, key=lambda r: r["kl"], reverse=True)[:top_n]
    scope = rows[0].get("scope", "keep")
    logger.warning(
        f"[KVE-TOP-KL] step={rows[0]['step']} "
        f"micro_step={rows[0]['micro_step']} scope={scope} "
        f"considered={counts.get('considered', 0)} "
        f"segments_sampled={len(records)} top_n={len(rows)}"
    )
    for i, rec in enumerate(rows, start=1):
        token_text = _kve_decode_one(tokenizer, rec["token_id"])
        entropy_text = (
            "None" if rec["entropy"] is None else f"{rec['entropy']:.6g}"
        )
        logger.warning(
            f"[KVE-TOP-KL] #{i:02d} kl={rec['kl']:.6g} "
            f"diff={rec['diff']:.6g} T_lp={rec['trainer_lp']:.6g} "
            f"V_lp={rec['inference_lp']:.6g} "
            f"entropy={entropy_text} "
            f"adv={rec['advantage']:.6g} prob_diff={rec['prob_diff']:.6g} "
            f"pos={rec['target_pos']} logit_pos={rec['logit_pos']} "
            f"tok={rec['token_id']} text={token_text} keep={int(rec['keep'])} "
            f"dppo_masked={int(rec['dppo_masked'])} "
            f"call={rec.get('call_idx')} admission={rec.get('has_admission')} "
            f"range=[{rec['tgt_start']},{rec['tgt_end']}) "
            f"logit_range=[{rec['logit_start']},{rec['logit_end']}) "
            f"sub_len={rec.get('sub_len')} comp_len={rec.get('comp_len')} "
            f"pad_len={rec.get('pad_len')} nuf={rec.get('nuf_len')} "
            f"adm_nuf={rec.get('admission_nuf_len')} "
            f"evicted={rec.get('admission_total_evicted')} "
            f"synth_cached={rec.get('synthetic_cached_tokens')} "
            f"synth_nuf={rec.get('synthetic_nuf_len')} "
            f"writer_off={rec.get('writer_offset')} "
            f"adm_off={rec.get('admission_offset_after')} "
            f"new_start={rec.get('new_content_start_in_sub')}"
        )


def _kve_collect_call_mismatch_record(
    *,
    records: list[dict],
    counts: dict[str, int],
    scope: str,
    step: int,
    micro_step: int,
    full_logit_start: int,
    full_logit_end: int,
    tgt_start: int,
    tgt_end: int,
    trainer_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    entropy: torch.Tensor | None,
    loss_config,
    call_plan,
) -> None:
    with torch.no_grad():
        t = trainer_logprobs.detach().float().squeeze(0)
        inf = inference_logprobs.detach().float().squeeze(0)
        adv = advantages.detach().float().squeeze(0)
        m_loss = loss_mask.detach().bool().squeeze(0)
        lbl = labels.detach().squeeze(0)
        diff = t - inf
        prob_diff = torch.exp(t) - torch.exp(inf)
        if isinstance(loss_config, DefaultLossConfig):
            dppo_high = prob_diff > loss_config.dppo_mask_high
            dppo_low = prob_diff < -loss_config.dppo_mask_low
            dppo_mask = torch.where(adv > 0, dppo_high, dppo_low)
        else:
            dppo_mask = torch.zeros_like(m_loss)
        keep_mask = m_loss & ~dppo_mask
        normalized_scope = scope.lower()
        if normalized_scope in {"loss", "loss_mask", "trainable", "all"}:
            select = m_loss
        elif normalized_scope in {"masked", "dppo_masked"}:
            select = m_loss & dppo_mask
        else:
            select = keep_mask
            normalized_scope = "keep"
        n_selected = int(select.sum().item())
        counts["considered"] = counts.get("considered", 0) + n_selected
        counts["segments"] = counts.get("segments", 0) + 1
        if n_selected == 0:
            return
        mismatch_kl = torch.exp(diff) - diff - 1.0
        selected_kl = mismatch_kl[select]
        selected_diff = diff[select]
        max_kl, local_selected_idx = torch.max(selected_kl, dim=0)
        selected_positions = torch.nonzero(select, as_tuple=False).flatten()
        local_i = int(selected_positions[int(local_selected_idx.item())].item())
        token_id = int(lbl[local_i].detach().cpu().item())
        ent = entropy.detach().float().squeeze(0) if entropy is not None else None
        rec = {
            "step": int(step),
            "micro_step": int(micro_step),
            "scope": normalized_scope,
            "n": n_selected,
            "mean_kl": float(selected_kl.mean().detach().cpu().item()),
            "max_kl": float(max_kl.detach().cpu().item()),
            "mean_abs_diff": float(selected_diff.abs().mean().detach().cpu().item()),
            "max_abs_diff": float(selected_diff.abs().max().detach().cpu().item()),
            "max_diff": float(diff[local_i].detach().cpu().item()),
            "max_trainer_lp": float(t[local_i].detach().cpu().item()),
            "max_inference_lp": float(inf[local_i].detach().cpu().item()),
            "max_entropy": (
                float(ent[local_i].detach().cpu().item())
                if ent is not None else None
            ),
            "max_advantage": float(adv[local_i].detach().cpu().item()),
            "token_id": token_id,
            "target_pos": tgt_start + local_i,
            "logit_pos": full_logit_start + local_i,
            "tgt_start": int(tgt_start),
            "tgt_end": int(tgt_end),
            "logit_start": int(full_logit_start),
            "logit_end": int(full_logit_end),
            "call_idx": (
                int(call_plan.get("call_idx"))
                if call_plan is not None and call_plan.get("call_idx") is not None
                else None
            ),
        }
        if call_plan is not None:
            rec.update({
                "has_admission": bool(call_plan.get("has_admission", False)),
                "sub_len": call_plan.get("sub_len"),
                "comp_len": call_plan.get("comp_len"),
                "pad_len": call_plan.get("pad_len"),
                "nuf_len": call_plan.get("nuf_len"),
                "writer_offset": call_plan.get("writer_offset"),
                "admission_offset_after": call_plan.get("admission_offset_after"),
                "admission_total_evicted": call_plan.get("admission_total_evicted"),
                "admission_nuf_len": call_plan.get("admission_nuf_len"),
                "synthetic_cached_tokens": call_plan.get("synthetic_cached_tokens"),
                "synthetic_nuf_len": call_plan.get("synthetic_nuf_len"),
                "synthetic_offset_after": call_plan.get("synthetic_offset_after"),
                "new_content_start_in_sub": call_plan.get("new_content_start_in_sub"),
            })
        records.append(rec)


def _kve_log_call_mismatch_records(
    logger,
    tokenizer,
    records: list[dict],
    counts: dict[str, int],
    top_n: int,
) -> None:
    if top_n <= 0 or not records:
        return
    rows = sorted(records, key=lambda r: r["max_kl"], reverse=True)[:top_n]
    threshold_raw = os.environ.get("KVE_CALL_MISMATCH_THRESHOLD", "0.1")
    try:
        threshold = float(threshold_raw)
    except ValueError:
        threshold = 0.1
    first_bad = None
    for rec in sorted(records, key=lambda r: (r["call_idx"] is None, r["call_idx"] or -1, r["tgt_start"])):
        if rec["max_kl"] >= threshold:
            first_bad = rec
            break
    first_text = "none"
    if first_bad is not None:
        first_text = (
            f"call={first_bad.get('call_idx')} "
            f"max={first_bad['max_kl']:.6g} "
            f"mean={first_bad['mean_kl']:.6g} "
            f"admission={first_bad.get('has_admission')} "
            f"range=[{first_bad['tgt_start']},{first_bad['tgt_end']})"
        )
    logger.warning(
        f"[KVE-CALL-KL] step={records[0]['step']} "
        f"micro_step={records[0]['micro_step']} scope={records[0]['scope']} "
        f"considered={counts.get('considered', 0)} "
        f"segments={counts.get('segments', 0)} calls_with_tokens={len(records)} "
        f"threshold={threshold:g} first_bad={first_text} top_n={len(rows)}"
    )
    for i, rec in enumerate(rows, start=1):
        token_text = _kve_decode_one(tokenizer, rec["token_id"])
        entropy_text = (
            "None"
            if rec["max_entropy"] is None
            else f"{rec['max_entropy']:.6g}"
        )
        logger.warning(
            f"[KVE-CALL-KL] #{i:02d} call={rec.get('call_idx')} "
            f"max={rec['max_kl']:.6g} mean={rec['mean_kl']:.6g} "
            f"n={rec['n']} mean_abs_diff={rec['mean_abs_diff']:.6g} "
            f"max_abs_diff={rec['max_abs_diff']:.6g} "
            f"max_diff={rec['max_diff']:.6g} "
            f"T_lp={rec['max_trainer_lp']:.6g} "
            f"V_lp={rec['max_inference_lp']:.6g} "
            f"entropy={entropy_text} adv={rec['max_advantage']:.6g} "
            f"tok={rec['token_id']} text={token_text} "
            f"pos={rec['target_pos']} logit_pos={rec['logit_pos']} "
            f"admission={rec.get('has_admission')} "
            f"range=[{rec['tgt_start']},{rec['tgt_end']}) "
            f"logit_range=[{rec['logit_start']},{rec['logit_end']}) "
            f"sub_len={rec.get('sub_len')} comp_len={rec.get('comp_len')} "
            f"pad_len={rec.get('pad_len')} nuf={rec.get('nuf_len')} "
            f"adm_nuf={rec.get('admission_nuf_len')} "
            f"evicted={rec.get('admission_total_evicted')} "
            f"synth_cached={rec.get('synthetic_cached_tokens')} "
            f"synth_nuf={rec.get('synthetic_nuf_len')} "
            f"writer_off={rec.get('writer_offset')} "
            f"adm_off={rec.get('admission_offset_after')} "
            f"new_start={rec.get('new_content_start_in_sub')}"
        )


@clean_exit
def train(config: TrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting RL trainer in {world} in {config.output_dir}")

    # Print warning if running in benchmark mode
    if config.bench is not None:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Setup heartbeat (only on rank 0)
    heart = None
    if config.heartbeat is not None and world.is_master:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat.url)

    # Setup metrics server (full on master, health-only on other nodes' local rank 0)
    metrics_server = None
    health_server = None
    if config.metrics_server is not None and world.local_rank == 0:
        if world.is_master:
            logger.info(f"Initializing metrics server on port {config.metrics_server.port}")
            metrics_server = MetricsServer(config.metrics_server)
            metrics_server.start()
        else:
            logger.info(f"Initializing health server on port {config.metrics_server.port}")
            health_server = HealthServer(config.metrics_server.port, config.metrics_server.host)
            health_server.start()

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=config.model.fsdp_cpu_offload
    )
    torch.set_float32_matmul_precision("high")

    # Setup multi run manager and offsets (including LoRA validation/scaling hooks if applicable)
    multi_run_manager = setup_multi_run_manager(
        config.output_dir, config.max_concurrent_runs, torch.device("cuda", world.local_rank), config.model.lora
    )

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)

    # Legacy segmented_forward still cannot safely run BPTT windows
    # under multi-rank FSDP2: ranks can enter backward at different
    # segment counts. The per-call dispatch has explicit per-window
    # forward padding below, so it is allowed through this early guard.
    if (
        config.compaction.window_size > 0
        and config.compaction.bptt_segments != 1
        and not config.compaction.per_call_dispatch
        and config.compaction.masked_forward_dispatch != "flex_attention"
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        raise ValueError(
            "Compaction with trainer.compaction.bptt_segments != 1 is not "
            "supported under multi-rank FSDP2 when "
            "trainer.compaction.per_call_dispatch is false. Got world_size="
            f"{dist.get_world_size()}, bptt_segments="
            f"{config.compaction.bptt_segments}. Enable per_call_dispatch, "
            "set bptt_segments=1, or run on a single GPU."
        )

    # For single-run, check for checkpoint to resume from
    checkpoint_step = None
    if config.max_concurrent_runs == 1:
        # Set up checkpoint manager for single-run
        logger.info(f"Initializing checkpoint managers ({config.ckpt})")
        ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt, config.model.lora)

        if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = config.ckpt.resume_step
    else:
        # Multi-run uses per-run checkpointing via MultiCheckpointManager
        ckpt_manager, weight_ckpt_manager = setup_multi_checkpoint_manager(config.output_dir)
        logger.info("Initialized multi-run checkpoint manager")

    # Initialize the model and tokenizer
    logger.info(f"Initializing model ({config.model})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Set up the loss function
    logger.info(f"Setting up loss function ({config.loss})")
    loss_fn = setup_loss_fn(config.loss)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")

    if config.max_concurrent_runs == 1:
        optimizer = setup_optimizer(
            config.optim,
            list(model.named_parameters()),
            parallel_dims,
            lora=config.model.lora is not None,
            cpu_offload=config.model.optim_cpu_offload,
        )
        scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    else:
        optimizer = setup_multi_optimizer(config.optim, parallel_dims)
        scheduler = setup_multi_scheduler(optimizer, config.scheduler, config.max_steps)

        # Register checkpoint loading callback at index 1 (after scheduler creation at index 0)
        def load_run_checkpoint(_optimizer, idx: int) -> None:
            ckpt_manager.load_run(idx, optimizer, scheduler)

        optimizer.register_post_creation_callback(load_run_checkpoint, index=1)

    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    # Set up weight broadcast (skip when using fake data since there's no inference server)
    if config.data.fake:
        weight_broadcast = None
        logger.info("Skipping weight broadcast setup (fake data mode)")
    else:
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        weight_broadcast = setup_weight_broadcast(config.output_dir, config.weight_broadcast, config.model.lora)

    if parallel_dims.cp_enabled:
        cp_group = parallel_dims.world_mesh["cp"].get_group()
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()
        substitute_hf_flash_attn(cp_group, heads_k_stride=1)
        substitute_ring_attn(cp_group, heads_k_stride=1, attn_impl=config.model.attn)
        from prime_rl.utils.cp import setup_hybrid_cp, setup_sparse_mla_cp

        setup_hybrid_cp(model, cp_group, cp_rank, parallel_dims.cp)
        setup_sparse_mla_cp(model, cp_group, cp_rank, parallel_dims.cp)

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")

    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
    )

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake, config.model.seq_len, parallel_dims.get_mesh("dp").size())
    else:
        dataloader = DataLoader(
            config.output_dir,
            progress.step,
            parallel_dims.get_mesh("dp").size(),
            config.model.seq_len,
            config.model.cp,
            tokenizer,
            config.rollout_transport,
            # D5 fix: when compaction training is on, tell the packer to
            # un-pack every sample into its own micro-batch so every
            # micro-batch routes through segmented_forward. Eliminates
            # the compaction <-> text transition that caused smoke-#4
            # OOM. See plans/phase3_training_integration.md.
            compaction_enabled=config.compaction.window_size > 0,
        )

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    is_first_step = True
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        # Broadcast weights at every step, (except step 0, because no need to broadcast the base model)
        # Also, with NCCL broadcast, we do not broadcast weights the last async level step as the orchestrator is already finished and will not initialize the receive on the inference; for filesystem broadcast, we do "broadcast" until the final step to allow to resume from the broadcast directory
        if weight_broadcast is None:
            broadcast_weights_time = 0
        else:
            last_async_level_steps = config.max_steps and progress.step >= config.max_steps - config.max_async_level
            if progress.step > 0 and (not last_async_level_steps or config.weight_broadcast.type == "filesystem"):
                broadcast_weights_start_time = time.perf_counter()
                weight_broadcast.broadcast_weights(model, step=progress.step)
                broadcast_weights_time = time.perf_counter() - broadcast_weights_start_time
                # Clean up old broadcast directories (unless at ckpt interval if using filesystem weight broadcast)
                ckpt_interval = config.ckpt and config.ckpt.interval
                interval_to_keep = ckpt_interval if config.weight_broadcast.type == "filesystem" else None
                if config.weight_broadcast.type == "filesystem":
                    weight_broadcast.maybe_clean(config.max_async_level, interval_to_keep)
            else:
                broadcast_weights_time = 0
                # Usually the broadcast will set this. If broadcast is skipped, we need to reset this here.
                for idx in multi_run_manager.used_idxs:
                    multi_run_manager.ready_to_update[idx] = False

        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            save_ckpt_time = 0

            gc.collect()
            torch.cuda.empty_cache()

            if not config.ckpt.weights_only:
                # Single-run: Save full checkpoint
                logger.info(f"Saving checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
                save_ckpt_time += time.perf_counter() - save_ckpt_start_time

            ckpt_manager.maybe_clean()

            # Save weight checkpoint
            if weight_ckpt_manager is not None:
                logger.info(f"Saving weight checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                weight_ckpt_manager.save(progress.step, model, tokenizer)
                save_ckpt_time += time.perf_counter() - save_ckpt_start_time
                weight_ckpt_manager.maybe_clean()
        elif config.max_concurrent_runs > 1:
            # Multi-run: Save per-run checkpoints (each run has its own interval from orchestrator config)
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(optimizer, scheduler)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time
            ckpt_manager.maybe_clean()
        else:
            save_ckpt_time = 0

        # Break if we have reached the maximum number of steps
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        logger.debug(f"Starting training step {progress.step}")
        step_start_time = time.perf_counter()

        # Wait for the batch to be available
        logger.debug("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.perf_counter()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.perf_counter() - wait_for_batch_start_time
        logger.debug(f"Waited for batch to arrive for {wait_for_batch_time:.2f} seconds")

        # Load the training batch
        logger.debug("Loading batch")
        load_data_start_time = time.perf_counter()
        micro_batches = dataloader.get_batch()
        load_data_time = time.perf_counter() - load_data_start_time
        logger.debug(f"Loaded batch in {load_data_time:.2f} seconds")

        # Offline-repro dump: serialize the step's raw MicroBatches (calls,
        # compaction/restore events, inference logprobs) so the flex-mask
        # replay harnesses can re-run trainer forwards deterministically
        # off-cluster. One file per (rank, step); msgspec list[MicroBatch],
        # the format compare_flexmask_per_call.py already decodes.
        _kve_dump_dir = os.environ.get("KVE_DUMP_MICRO_BATCHES", "")
        if _kve_dump_dir:
            try:
                import msgspec as _msgspec

                _rank = int(os.environ.get("RANK", "0"))
                os.makedirs(_kve_dump_dir, exist_ok=True)
                _dump_path = os.path.join(
                    _kve_dump_dir,
                    f"micro_batches_rank{_rank}_step{progress.step}.bin",
                )
                with open(_dump_path, "wb") as _fh:
                    _fh.write(_msgspec.msgpack.encode(micro_batches))
                logger.warning(
                    f"[KVE-DUMP] wrote {len(micro_batches)} micro-batches "
                    f"to {_dump_path}"
                )
            except Exception:
                logger.exception("[KVE-DUMP] micro-batch dump failed")

        batch_size = len(micro_batches)
        flex_stack_mode = config.data.micro_batch_flex_stack_mode
        stacking_enabled = int(config.data.micro_batch_stack_size) > 1
        flex_compaction_stack_enabled = False
        flex_compaction_stackable = None
        if stacking_enabled:
            flex_compaction_stack_enabled = (
                config.compaction.window_size > 0
                and config.compaction.masked_forward_dispatch == "flex_attention"
                and (
                    (
                        flex_stack_mode == "vertical"
                        and batched_flex_mask_segmented_forward is not None
                    )
                    or (
                        flex_stack_mode == "horizontal"
                        and packed_flex_mask_segmented_forward is not None
                    )
                )
            )
            dp_mesh = parallel_dims.get_mesh("dp")
            if flex_compaction_stack_enabled:
                flex_compaction_stackable = _get_dp_flex_compaction_stackable(
                    micro_batches,
                    dp_group=dp_mesh.get_group(),
                    dp_world_size=dp_mesh.size(),
                    compaction_enabled=config.compaction.window_size > 0,
                    flex_compaction_enabled=flex_compaction_stack_enabled,
                    cp_enabled=parallel_dims.cp_enabled,
                    lora_enabled=config.model.lora is not None,
                    multi_run_enabled=config.max_concurrent_runs > 1,
                )
            stack_seq_lens = None
            if config.data.micro_batch_stack_token_budget is not None:
                stack_seq_lens = _get_dp_max_micro_batch_seq_lens(
                    micro_batches,
                    dp_group=dp_mesh.get_group(),
                    dp_world_size=dp_mesh.size(),
                    compaction_enabled=config.compaction.window_size > 0,
                    flex_compaction_enabled=flex_compaction_stack_enabled,
                    cp_enabled=parallel_dims.cp_enabled,
                    lora_enabled=config.model.lora is not None,
                    multi_run_enabled=config.max_concurrent_runs > 1,
                    flex_compaction_stackable=flex_compaction_stackable,
                )
            micro_batch_groups = make_micro_batch_groups(
                micro_batches,
                stack_size=config.data.micro_batch_stack_size,
                stack_token_budget=config.data.micro_batch_stack_token_budget,
                flex_compaction_stack_mode=flex_stack_mode,
                seq_lens=stack_seq_lens,
                flex_compaction_enabled=flex_compaction_stack_enabled,
                flex_compaction_stackable=flex_compaction_stackable,
                compaction_enabled=config.compaction.window_size > 0,
                cp_enabled=parallel_dims.cp_enabled,
                lora_enabled=config.model.lora is not None,
                multi_run_enabled=config.max_concurrent_runs > 1,
            )
        else:
            micro_batch_groups = [[micro_batch] for micro_batch in micro_batches]
        memory_profiler = None
        if config.memory_profiler_path is not None:
            memory_profiler = MemoryProfiler(progress.step, config.memory_profiler_path)

        forward_backward_start_time = time.perf_counter()
        seq_len = micro_batches[0]["input_ids"].shape[1]

        # Normalize by the local number of unmasked tokens in the batch (per-batch length normalization)
        loss_scale = sum(micro_batch["loss_mask"].sum().item() for micro_batch in micro_batches)
        loss_scale = max(loss_scale, 1)

        logger.debug(f"Starting forward and backward pass ({batch_size=})")
        tensors = Tensors()  # Used to accumulate tensor statistics across micro-batches and ranks for logging
        cp_enabled = parallel_dims.cp_enabled
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
        cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
        cp_size = parallel_dims.cp

        for micro_step, micro_batch_group in enumerate(micro_batch_groups):
            flex_compaction_group = all(
                is_flex_compaction_stackable_micro_batch(
                    mb,
                    compaction_enabled=config.compaction.window_size > 0,
                    flex_compaction_enabled=flex_compaction_stack_enabled,
                    cp_enabled=parallel_dims.cp_enabled,
                    lora_enabled=config.model.lora is not None,
                    multi_run_enabled=config.max_concurrent_runs > 1,
                )
                for mb in micro_batch_group
            )
            if flex_compaction_group:
                if flex_stack_mode == "horizontal":
                    micro_batch = pack_horizontal_flex_compaction_micro_batches(
                        micro_batch_group
                    )
                else:
                    micro_batch = stack_flex_compaction_micro_batches(
                        micro_batch_group
                    )
            else:
                micro_batch = stack_standard_micro_batches(micro_batch_group)
            stacked_micro_batches = len(micro_batch_group)
            _trace_mb_start_time = None
            _stack_trace = ""
            if _KVE_MEM_TRACE or _KVE_STACK_TRACE:
                # "EVT" = sample has real compaction events (multi-
                # segment segmented_forward); "NOEVT" = sample is
                # event-less (single-segment in a compaction run, or
                # packed text in a non-compaction run). In a compaction
                # run both types take the segmented path; only the
                # segment count differs.
                _compaction_flag = (
                    "EVT" if (
                        micro_batch.get("compaction_events") is not None
                        and len(micro_batch.get("compaction_events") or []) > 0
                    ) else "NOEVT"
                )
                _seq = tuple(micro_batch["input_ids"].shape)
                _stack_trace = _stack_trace_summary(
                    micro_batch_group,
                    micro_batch,
                    flex_compaction_group=flex_compaction_group,
                )
                _trace_mb_start_time = time.perf_counter()
                if _KVE_MEM_TRACE:
                    torch.cuda.reset_peak_memory_stats()
                logger.info(
                    _trace_snap(
                        f"mb {micro_step}/{len(micro_batch_groups)} "
                        f"{_compaction_flag} stack={stacked_micro_batches} "
                        f"shape={_seq}{_stack_trace} ENTRY"
                    )
                )
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            inference_logprobs = micro_batch["inference_logprobs"].to("cuda")
            teacher_logprobs = (
                micro_batch["teacher_logprobs"].to("cuda") if micro_batch["teacher_logprobs"] is not None else None
            )
            routed_experts = (
                micro_batch["routed_experts"].to("cuda") if micro_batch["routed_experts"] is not None else None
            )

            if routed_experts is None and config.enable_router_replay:
                raise ValueError(
                    "You must set `enable_return_routed_experts=True` in the inference config or pass `--enable-return-routed-experts` to vLLM server to use router replay."
                )

            if routed_experts is not None and not config.enable_router_replay:
                # we could've gotten routed experts from the inference server, but we didn't enable router replay
                routed_experts = None

            # Multimodal fields (Qwen3-VL) - only present for VLM training
            pixel_values = (
                micro_batch["pixel_values"].to("cuda") if micro_batch.get("pixel_values") is not None else None
            )
            image_grid_thw = (
                micro_batch["image_grid_thw"].to("cuda") if micro_batch.get("image_grid_thw") is not None else None
            )

            labels = shift_tensor_left(input_ids)

            # VLM + CP is not supported: MRoPE requires global positions but CP shards the sequence
            if cp_enabled and pixel_values is not None:
                raise NotImplementedError("Context parallelism is not supported with VLM/multimodal training")

            if cp_enabled:
                input_ids, forward_position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size, cp_group)
                labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=cp_size)
                if routed_experts is not None:
                    routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_size)
            else:
                forward_position_ids = position_ids

            if config.model.lora:
                lora_num_tokens = micro_batch["lora_num_tokens"].to("cuda")
                if cp_enabled:
                    chunk_size = input_ids.shape[1]
                    # Convert to cumsum, adjust for CP chunk, convert back to num_tokens
                    cu_offsets = lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
                    adjusted_cu = torch.clip(cu_offsets - chunk_size * cp_rank, min=0, max=chunk_size)
                    lora_num_tokens = torch.diff(
                        adjusted_cu, prepend=torch.tensor([0], device=adjusted_cu.device, dtype=adjusted_cu.dtype)
                    )
                set_lora_num_tokens(lora_num_tokens)

            temperatures = micro_batch["temperatures"].to("cuda")

            # Shard temperatures for context parallelism if enabled
            if cp_enabled:
                temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=cp_size)

            # kv-eviction: in a compaction training run, route EVERY
            # micro-batch through segmented_forward, even samples whose
            # inference rollout didn't trigger a compaction event.
            # Samples with empty events run as a single-segment forward
            # (numerically equivalent to a standard text forward on the
            # same unpacked sample). Samples with events run multi-
            # segment. Every sample is un-packed by the packer (see the
            # compaction_enabled thread in trainer/batch.py).
            #
            # Pre-D5 behavior was `use_segmented = <sample has events>`,
            # which left event-less short rollouts on the packed text
            # path. The resulting compaction <-> text modality
            # transition inside a single step caused the smoke-#4 OOM
            # cascade: the allocator held ~11 GB of cached blocks sized
            # for per-segment tensor patterns that didn't fit the
            # contiguous shapes the packed text forward needed. See
            # plans/phase3_training_integration.md D5 section.
            compaction_events = micro_batch.get("compaction_events") or []
            use_segmented = config.compaction.window_size > 0
            use_per_call = False
            use_flex_mask = False
            calls_obj = micro_batch.get("calls")
            horizontal_flex_compaction = (
                micro_batch.get("flex_stack_mode") == "horizontal"
            )
            batched_flex_compaction = (
                use_segmented
                and flex_compaction_group
                and (input_ids.shape[0] > 1 or horizontal_flex_compaction)
                and isinstance(calls_obj, list)
                and len(calls_obj) > 0
                and isinstance(calls_obj[0], list)
            )

            # Defense-in-depth: verify every DP rank made the SAME branch
            # decision at this step index. prepare_batch partitions by
            # modality so this should already hold, but if a packer
            # regression, data corruption, or serialization drift caused
            # rank-level divergence, we'd otherwise silently deadlock at
            # the first intra-branch collective. A tiny all_reduce MAX/MIN
            # on the local bool catches it cheaply and turns a NCCL hang
            # into an informative assertion.
            if dist.is_initialized() and dist.get_world_size() > 1:
                bool_tensor = torch.tensor(
                    [int(use_segmented)], device="cuda", dtype=torch.int32
                )
                max_b = bool_tensor.clone()
                min_b = bool_tensor.clone()
                dist.all_reduce(max_b, op=dist.ReduceOp.MAX)
                dist.all_reduce(min_b, op=dist.ReduceOp.MIN)
                assert int(max_b.item()) == int(min_b.item()), (
                    "Rank-level divergence in the compaction/standard "
                    "dispatch branch. prepare_batch is supposed to enforce "
                    "modality uniformity across DP ranks at each step index "
                    "via per-group padding, so this should be unreachable. "
                    "A divergence here would otherwise deadlock at the next "
                    "FSDP all-gather or the segmented all_reduce."
                )

            if use_segmented:
                assert segmented_forward is not None, (
                    "Micro batch carries compaction_events but the "
                    "kv_eviction package is not importable. Install kv_eviction "
                    "or disable compaction on the inference engine."
                )
                assert config.compaction.window_size > 0, (
                    "Micro batch has compaction_events but trainer config "
                    "compaction.window_size is 0. The trainer's compaction "
                    "config must mirror the inference engine's config."
                )
                assert config.model.attn in {"flash_attention_2", "flex_attention"}, (
                    f"Compaction requires attn='flash_attention_2' for cache replay "
                    f"or 'flex_attention' for masked FlexAttention. Got "
                    f"attn={config.model.attn!r}."
                )
                assert config.model.impl == "hf", (
                    f"Compaction requires impl='hf' (the custom llama path "
                    f"asserts past_key_values is None and cannot run "
                    f"segmented_forward). Got impl={config.model.impl!r}. "
                    f"Set trainer.model.impl='hf' in the config."
                )
                assert not cp_enabled, (
                    "Compaction is incompatible with context parallelism: "
                    "CP shards input_ids along the sequence dimension, but "
                    "segmented_forward needs the full unsharded sequence to "
                    "compute segment ranges and re-feed the boundary token. "
                    "The TrainerConfig validator rejects this combination at "
                    "config load; this runtime assert is defense-in-depth."
                )
                # Legacy fallback still cannot run bptt_segments != 1
                # under multi-rank FSDP2. Per-call dispatch handles it
                # by synchronizing forward counts at each BPTT window.
                mb_prompt_len = micro_batch.get("prompt_len")
                assert mb_prompt_len is not None, (
                    "prompt_len missing on micro_batch in compaction run "
                    "(compaction.window_size > 0). prepare_sample should "
                    "set prompt_len on EVERY sample when compaction_enabled "
                    "is True; a missing value points to a wire-format or "
                    "packer regression."
                )
                # Legacy/per-call cache replay is still singleton. Batched
                # compaction is only supported by the masked FlexAttention
                # path, whose BlockMask encodes each row's KV liveness.
                if not batched_flex_compaction:
                    assert input_ids.shape[0] == 1, (
                        f"Compaction samples must be batch_size=1 outside "
                        f"batched flex replay, got "
                        f"input_ids.shape={tuple(input_ids.shape)}. Check "
                        f"prepare_batch / _is_compaction_sample partitioning."
                    )
                    bs = config.compaction.block_size
                    pp = config.compaction.protected_prefix_tokens
                    if pp == -1:
                        # Auto-detect: infer from first compaction event's
                        # position_offset_after and tokens_evicted. The first
                        # eviction starts at the block-aligned protected prefix,
                        # so offset_after - evicted gives us the eviction start.
                        # Under D5 unified dispatch, event-less samples also
                        # pass through this branch; in that case pp is unused
                        # downstream (no eviction math needed) — fall back to
                        # mb_prompt_len so the effective-prompt calc is a no-op.
                        if compaction_events:
                            first_evt = compaction_events[0]
                            pp = (
                                first_evt.position_offset_after
                                - first_evt.tokens_evicted
                            )
                        else:
                            pp = mb_prompt_len
                    effective_prompt_len = (
                        min(pp, mb_prompt_len) if pp > 0 else mb_prompt_len
                    )
                    prompt_aligned_len = (
                        (effective_prompt_len + bs - 1) // bs
                    ) * bs
                    segment_boundaries = [
                        int(e.num_output_tokens_at_compaction)
                        for e in compaction_events
                    ]
                else:
                    assert (
                        config.compaction.masked_forward_dispatch == "flex_attention"
                    )
                    if horizontal_flex_compaction:
                        assert packed_flex_mask_segmented_forward is not None
                    else:
                        assert batched_flex_mask_segmented_forward is not None
                    prompt_aligned_len = 0
                    segment_boundaries = []
                # Per-rank segment count may diverge across DP ranks if some
                # samples have more compaction events than others. All-reduce
                # MAX so every rank runs the same number of forward passes,
                # preventing NCCL all-gather deadlock. segmented_forward pads
                # with dummy passes on ranks whose actual count is below max.
                #
                # INVARIANT: all DP ranks must simultaneously take either the
                # segmented or the standard branch within a single training
                # step. Mixing (some ranks segmented, others standard) causes
                # a mismatch in total forward-pass count per step and will
                # deadlock at the next FSDP all-gather. The usual way this
                # breaks is a very short rollout that never triggered any
                # compaction event landing on one rank while other ranks see
                # long rollouts with events. For Phase 3 initial testing,
                # configure experiments so all rollouts exceed the compaction
                # window (avoid mixing long and short rollouts in the same
                # batch).
                # Per-call dispatch (plans/single_forward_pre_eviction.md, Phase 5).
                # When per_call_dispatch is enabled AND the micro-batch
                # carries a `calls` list, route through per_call_segmented_forward.
                # That function runs ONE HF forward per call against a
                # persistent DynamicCache (carried across calls, detached
                # between them). Admission events are handled inline via
                # eviction-aware position_ids — no two-phase split, no
                # cache splice. The all_reduce_MAX on per-rank forward
                # counts (= len(calls)) keeps FSDP2 collectives synced.
                #
                # When ANY call has mid-gen compaction events
                # (sliding-window block-FIFO firing during decode,
                # num_output_tokens_at_compaction > 0), fall back to the
                # legacy segmented_forward — its per-stride drop logic
                # handles mid-gen correctly. The per-call path is the
                # structural fix for ADMISSION (multi-turn KL gap) and
                # doesn't duplicate mid-gen handling.
                use_flex_mask_local = (
                    config.compaction.masked_forward_dispatch == "flex_attention"
                    and micro_batch.get("calls") is not None
                    and (
                        (
                            batched_flex_compaction
                            and (
                                (
                                    horizontal_flex_compaction
                                    and packed_flex_mask_segmented_forward is not None
                                )
                                or (
                                    not horizontal_flex_compaction
                                    and (
                                        batched_flex_mask_segmented_forward
                                        is not None
                                    )
                                )
                            )
                        )
                        or (
                            not batched_flex_compaction
                            and flex_mask_segmented_forward is not None
                        )
                    )
                )
                use_per_call_local = (
                    config.compaction.per_call_dispatch
                    and per_call_segmented_forward is not None
                    and micro_batch.get("calls") is not None
                    and not batched_flex_compaction
                    and not use_flex_mask_local
                )
                # Cross-rank agreement on the call-based dispatch BEFORE entering any
                # branch-conditional dist.all_reduce. If ANY rank cannot
                # dispatch through calls (its sample has no `calls` list — e.g.
                # a Markovian Summary sample), ALL ranks fall back to
                # legacy. Doing this MIN first avoids the deadlock where
                # one rank enters the mid-gen all_reduce below while
                # another skips it.
                if dist.is_initialized() and dist.get_world_size() > 1:
                    flex_t = torch.tensor(
                        [int(use_flex_mask_local)],
                        device="cuda", dtype=torch.int32,
                    )
                    pc_t = torch.tensor(
                        [int(use_per_call_local)],
                        device="cuda", dtype=torch.int32,
                    )
                    dist.all_reduce(flex_t, op=dist.ReduceOp.MIN)
                    dist.all_reduce(pc_t, op=dist.ReduceOp.MIN)
                    use_flex_mask = bool(flex_t.item())
                    use_per_call = bool(pc_t.item()) and not use_flex_mask
                else:
                    use_flex_mask = use_flex_mask_local
                    use_per_call = use_per_call_local

                if use_flex_mask or use_per_call:
                    calls_for_midgen = micro_batch["calls"]
                    if batched_flex_compaction:
                        calls_for_midgen = [
                            call
                            for row_calls in calls_for_midgen
                            for call in row_calls
                        ]
                    has_midgen_local = any(
                        any(
                            int(getattr(e, "num_output_tokens_at_compaction", 0)) > 0
                            for e in (call.compaction_events or [])
                        )
                        for call in calls_for_midgen
                    )
                    # If ANY rank has a mid-gen event, ALL ranks fall back
                    # to legacy (per_call_segmented_forward defers mid-gen
                    # handling to the legacy path).
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        mg_t = torch.tensor(
                            [int(has_midgen_local)],
                            device="cuda", dtype=torch.int32,
                        )
                        dist.all_reduce(mg_t, op=dist.ReduceOp.MAX)
                        has_midgen_local = bool(mg_t.item())
                    if has_midgen_local:
                        use_flex_mask = False
                        use_per_call = False

                if (
                    config.compaction.masked_forward_dispatch == "flex_attention"
                    and not use_flex_mask
                ):
                    raise ValueError(
                        "trainer.compaction.masked_forward_dispatch='flex_attention' "
                        "requires kv_eviction.flex_mask_segmented_forward, "
                        "TrainingSample.calls on every rank, and no mid-generation "
                        "compaction events."
                    )

                if (
                    not use_per_call
                    and not use_flex_mask
                    and config.compaction.bptt_segments != 1
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                ):
                    raise ValueError(
                        "trainer.compaction.bptt_segments != 1 under "
                        "multi-rank FSDP2 requires per-call compaction "
                        "dispatch for every rank in the step. This "
                        "micro-batch fell back to legacy segmented_forward "
                        "(missing calls or mid-generation compaction event). "
                        "Use bptt_segments=1 for legacy/mixed samples."
                    )

                seq_len_local = input_ids.shape[1]
                if use_flex_mask:
                    n_forwards_local = 1
                elif use_per_call:
                    n_forwards_local = compute_num_per_call_forwards(
                        micro_batch["calls"]
                    )
                else:
                    n_forwards_local = compute_num_segments(
                        seq_len_local, mb_prompt_len, segment_boundaries
                    )

                if dist.is_initialized() and dist.get_world_size() > 1:
                    n_forwards_t = torch.tensor(
                        [n_forwards_local], device="cuda", dtype=torch.int32
                    )
                    dist.all_reduce(n_forwards_t, op=dist.ReduceOp.MAX)
                    max_forwards = int(n_forwards_t.item())
                else:
                    max_forwards = n_forwards_local

                max_bptt_window_forward_passes = None
                if use_per_call and config.compaction.bptt_segments != 1:
                    assert compute_per_call_bptt_window_forward_counts is not None
                    local_window_forwards = (
                        compute_per_call_bptt_window_forward_counts(
                            micro_batch["calls"],
                            config.compaction.bptt_segments,
                        )
                    )
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        n_windows_t = torch.tensor(
                            [len(local_window_forwards)],
                            device="cuda",
                            dtype=torch.int32,
                        )
                        dist.all_reduce(n_windows_t, op=dist.ReduceOp.MAX)
                        max_windows = int(n_windows_t.item())
                        local_windows_t = torch.zeros(
                            max_windows,
                            device="cuda",
                            dtype=torch.int32,
                        )
                        if local_window_forwards:
                            local_windows_t[: len(local_window_forwards)] = (
                                torch.tensor(
                                    local_window_forwards,
                                    device="cuda",
                                    dtype=torch.int32,
                                )
                            )
                        dist.all_reduce(local_windows_t, op=dist.ReduceOp.MAX)
                        max_bptt_window_forward_passes = [
                            int(x) for x in local_windows_t.cpu().tolist()
                        ]
                    else:
                        max_bptt_window_forward_passes = local_window_forwards

                # Per-segment loss closure. segmented_forward will call
                # this after every segment's forward with the segment's
                # pre-scaled logits and the range of full-sequence
                # positions the segment owns. We slice the
                # full-sequence quantities (labels, advantages, mask,
                # inference/teacher logprobs) to match and delegate to
                # prime-rl's standard compute_loss — no duplication of
                # the loss formula, just range bookkeeping.
                #
                # Alignment detail: segment owns logit positions
                # [full_logit_start, full_logit_end), which predict
                # target tokens at positions [full_logit_start + 1,
                # full_logit_end + 1). selective_log_softmax on the
                # segment's logits against labels[:, start:end] yields
                # per-token logprobs aligned with those target
                # positions directly (no shift_tensor_right needed,
                # because unlike the standard path we never look at
                # position 0, which is a prompt token that belongs to
                # no segment's loss range).
                #
                # Scaling decomposition: compute_loss sums result.loss
                # over sequences and divides by loss_scale once at the
                # end. The per-token loss functions (default_loss_fn,
                # sft_loss_fn) return result.loss as a pure .sum() over
                # token contributions, with no internal averaging.
                # Therefore sum_over_segments(compute_loss(seg).item())
                # == compute_loss(full_seq).item() exactly, as long as
                # every segment call uses the SAME global loss_scale.
                # window_loss inside segmented_forward accumulates the
                # per-segment scaled losses (each already divided by
                # loss_scale) before backward, so the final
                # accumulated_loss equals the full-sample scaled loss.
                accumulated_loss_tensors: dict[str, list[torch.Tensor]] = {}
                # Per-segment masked entropy values, to be concat'd into
                # a single 1-D tensor and appended to tensors["entropy"]
                # once segmented_forward returns. Each segment pushes
                # only its OWNED target tokens (disjoint across segments
                # by construction — see the owned_range logic in
                # segmented_forward) so the concatenation counts every
                # completion token exactly once, matching the standard
                # path's out["entropy"][loss_mask] semantics.
                accumulated_entropy_masked: list[torch.Tensor] = []

                full_seq_len = input_ids.shape[1]
                full_seq_lens = micro_batch.get("sequence_lengths")
                if full_seq_lens is None:
                    full_seq_lens = [full_seq_len] * int(input_ids.shape[0])
                sequence_offsets = micro_batch.get("sequence_offsets")
                if sequence_offsets is None:
                    sequence_offsets = [0] * len(full_seq_lens)
                singleton_compaction_row = (
                    not batched_flex_compaction
                    and not horizontal_flex_compaction
                    and int(input_ids.shape[0]) == 1
                )

                def _row_slice(
                    tensor: torch.Tensor,
                    batch_idx: int,
                    start: int,
                    end: int,
                ) -> torch.Tensor:
                    if singleton_compaction_row:
                        return tensor[:, start:end]
                    if horizontal_flex_compaction:
                        offset = int(sequence_offsets[batch_idx])
                        return tensor[:, offset + start : offset + end]
                    return tensor[batch_idx : batch_idx + 1, start:end]

                def _row_assign(
                    tensor: torch.Tensor,
                    batch_idx: int,
                    start: int,
                    end: int,
                    values: torch.Tensor,
                ) -> None:
                    if singleton_compaction_row:
                        tensor[:, start:end] = values
                        return
                    if horizontal_flex_compaction:
                        offset = int(sequence_offsets[batch_idx])
                        tensor[:, offset + start : offset + end] = values
                    else:
                        tensor[batch_idx : batch_idx + 1, start:end] = values

                def _full_seq_len_for_row(batch_idx: int) -> int:
                    if singleton_compaction_row:
                        return full_seq_len
                    return int(full_seq_lens[batch_idx])

                distill_cfg = config.compaction.distillation
                distill_enabled = use_flex_mask and distill_cfg.enabled
                distill_adv_adjustment = (
                    torch.zeros_like(advantages) if distill_enabled else None
                )
                kve_top_mismatch_n = _kve_top_mismatch_n()
                kve_top_mismatch_scope = os.environ.get(
                    "KVE_TOP_MISMATCH_SCOPE", "keep"
                )
                kve_call_mismatch_n = _kve_call_mismatch_n()
                kve_call_mismatch_scope = os.environ.get(
                    "KVE_CALL_MISMATCH_SCOPE",
                    os.environ.get("KVE_TOP_MISMATCH_SCOPE", "keep"),
                )
                kve_top_mismatch_records: list[dict] = []
                kve_top_mismatch_counts: dict[str, int] = {}
                kve_call_mismatch_records: list[dict] = []
                kve_call_mismatch_counts: dict[str, int] = {}
                kve_call_plans = None
                if (
                    (kve_top_mismatch_n > 0 or kve_call_mismatch_n > 0)
                    and (use_per_call or use_flex_mask)
                    and not batched_flex_compaction
                    and _build_pre_trim_plan is not None
                ):
                    try:
                        kve_call_plans, _ = _build_pre_trim_plan(
                            micro_batch["calls"]
                        )
                    except Exception:
                        logger.exception("[KVE-TOP-KL] failed to build call plan")

                def _append_distill_metric(
                    name: str,
                    values: torch.Tensor,
                    mask: torch.Tensor,
                ) -> None:
                    if mask.any():
                        accumulated_loss_tensors.setdefault(name, []).append(
                            values[mask].detach().to("cpu", dtype=torch.float32)
                        )

                distill_compact_state: dict[
                    tuple[int, int, int], dict[str, torch.Tensor]
                ] = {}

                def _distill_effective_slice(
                    logits: torch.Tensor,
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> tuple[torch.Tensor, int]:
                    row_full_seq_len = _full_seq_len_for_row(batch_idx)
                    if full_logit_end >= row_full_seq_len:
                        effective_logit_end = row_full_seq_len - 1
                        logits = logits[
                            :, : effective_logit_end - full_logit_start, :
                        ]
                    else:
                        effective_logit_end = full_logit_end
                    return logits, effective_logit_end

                def _distill_labels_and_mask(
                    full_logit_start: int,
                    effective_logit_end: int,
                    batch_idx: int = 0,
                ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
                    distill_labels = _row_slice(
                        labels,
                        batch_idx,
                        full_logit_start,
                        effective_logit_end,
                    )
                    tgt_start = full_logit_start + 1
                    tgt_end = effective_logit_end + 1
                    distill_mask = _row_slice(
                        loss_mask,
                        batch_idx,
                        tgt_start,
                        tgt_end,
                    ).bool()
                    return distill_labels, distill_mask, tgt_start, tgt_end

                def _segment_distillation_prepare_fn(
                    student_logits: torch.Tensor,
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> None:
                    if distill_cfg.estimator != "rb_topk":
                        return
                    student_eff, effective_logit_end = _distill_effective_slice(
                        student_logits,
                        full_logit_start,
                        full_logit_end,
                        batch_idx,
                    )
                    if effective_logit_end <= full_logit_start:
                        return
                    _, distill_mask, _, _ = _distill_labels_and_mask(
                        full_logit_start,
                        effective_logit_end,
                        batch_idx,
                    )
                    if not distill_mask.any():
                        return
                    k = max(1, min(int(distill_cfg.top_k), student_eff.shape[-1]))
                    _, top_idx = torch.topk(student_eff.float(), k=k, dim=-1)
                    distill_compact_state[
                        (batch_idx, full_logit_start, effective_logit_end)
                    ] = {
                        "top_idx": top_idx.detach().to("cpu"),
                    }

                def _segment_distillation_teacher_fn(
                    teacher_logits: torch.Tensor,
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> None:
                    teacher_eff, effective_logit_end = _distill_effective_slice(
                        teacher_logits,
                        full_logit_start,
                        full_logit_end,
                        batch_idx,
                    )
                    if effective_logit_end <= full_logit_start:
                        return
                    distill_labels, distill_mask, _, _ = _distill_labels_and_mask(
                        full_logit_start,
                        effective_logit_end,
                        batch_idx,
                    )
                    if not distill_mask.any():
                        return

                    key = (batch_idx, full_logit_start, effective_logit_end)
                    state = distill_compact_state.setdefault(key, {})
                    teacher_float = teacher_eff.float()
                    teacher_logz = torch.logsumexp(
                        teacher_float, dim=-1, keepdim=True
                    )
                    selected_teacher = (
                        torch.gather(
                            teacher_float,
                            dim=-1,
                            index=distill_labels.unsqueeze(-1),
                        ).squeeze(-1)
                        - teacher_logz.squeeze(-1)
                    )
                    state["selected_teacher_logp"] = selected_teacher.detach().to("cpu")

                    if distill_cfg.estimator == "rb_topk":
                        top_idx_cpu = state.get("top_idx")
                        if top_idx_cpu is None:
                            raise RuntimeError(
                                "RB-topk distillation teacher hook ran before "
                                "student support preparation."
                            )
                        top_idx = top_idx_cpu.to(
                            device=teacher_eff.device, non_blocking=True
                        )
                        top_teacher_logp = (
                            torch.gather(teacher_float, dim=-1, index=top_idx)
                            - teacher_logz
                        )
                        state["top_teacher_logp"] = top_teacher_logp.detach().to("cpu")

                def _segment_distillation_loss_fn(
                    student_logits: torch.Tensor,
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> torch.Tensor:
                    student_eff, effective_logit_end = _distill_effective_slice(
                        student_logits,
                        full_logit_start,
                        full_logit_end,
                        batch_idx,
                    )
                    if effective_logit_end <= full_logit_start:
                        return student_logits.sum() * 0.0
                    key = (batch_idx, full_logit_start, effective_logit_end)
                    state = distill_compact_state.get(key)
                    if state is None:
                        return student_eff.sum() * 0.0
                    distill_labels, distill_mask, tgt_start, tgt_end = (
                        _distill_labels_and_mask(
                            full_logit_start,
                            effective_logit_end,
                            batch_idx,
                        )
                    )
                    if not distill_mask.any():
                        return student_eff.sum() * 0.0

                    selected_teacher_cpu = state.get("selected_teacher_logp")
                    if selected_teacher_cpu is None:
                        raise RuntimeError(
                            "distillation teacher selected logprobs are missing"
                        )
                    student_float = student_eff.float()
                    student_logz = torch.logsumexp(
                        student_float, dim=-1, keepdim=True
                    )
                    selected_student = (
                        torch.gather(
                            student_float,
                            dim=-1,
                            index=distill_labels.unsqueeze(-1),
                        ).squeeze(-1)
                        - student_logz.squeeze(-1)
                    )
                    selected_teacher = selected_teacher_cpu.to(
                        device=student_eff.device, non_blocking=True
                    )
                    selected_k1 = selected_student - selected_teacher

                    topk_mass: torch.Tensor | None = None
                    if distill_cfg.estimator == "k1_sample":
                        kl = selected_k1
                    elif distill_cfg.estimator == "rb_topk":
                        top_idx_cpu = state.get("top_idx")
                        top_teacher_cpu = state.get("top_teacher_logp")
                        if top_idx_cpu is None or top_teacher_cpu is None:
                            raise RuntimeError(
                                "RB-topk distillation support is incomplete"
                            )
                        top_idx = top_idx_cpu.to(
                            device=student_eff.device, non_blocking=True
                        )
                        top_teacher_logp = top_teacher_cpu.to(
                            device=student_eff.device, non_blocking=True
                        )
                        top_student_logits = torch.gather(
                            student_float,
                            dim=-1,
                            index=top_idx,
                        )
                        top_student_logp = top_student_logits - student_logz
                        top_prob = top_student_logp.exp()
                        kl = (
                            top_prob * (top_student_logp - top_teacher_logp)
                        ).sum(dim=-1)
                        topk_mass = top_prob.sum(dim=-1)
                    else:
                        raise ValueError(
                            f"compact distillation does not support "
                            f"estimator={distill_cfg.estimator!r}"
                        )

                    reward_signal = kl
                    if distill_adv_adjustment is not None and distill_cfg.reward_coef > 0:
                        shaped_reward = -distill_cfg.reward_coef * reward_signal.detach()
                        _row_assign(
                            distill_adv_adjustment,
                            batch_idx,
                            tgt_start,
                            tgt_end,
                            torch.where(
                                distill_mask,
                                shaped_reward,
                                torch.zeros_like(shaped_reward),
                            ),
                        )
                        _append_distill_metric(
                            "distill/reward",
                            shaped_reward,
                            distill_mask,
                        )

                    _append_distill_metric(
                        "distill/student_teacher_kl",
                        kl,
                        distill_mask,
                    )
                    _append_distill_metric(
                        "distill/reverse_kl",
                        kl,
                        distill_mask,
                    )
                    _append_distill_metric(
                        "distill/k1_sample",
                        selected_k1,
                        distill_mask,
                    )
                    if topk_mass is not None:
                        _append_distill_metric(
                            "distill/topk_mass",
                            topk_mass,
                            distill_mask,
                        )

                    if distill_cfg.loss_coef <= 0:
                        return student_eff.sum() * 0.0
                    return (
                        distill_cfg.loss_coef
                        * kl[distill_mask].sum()
                        / loss_scale
                    )

                def _segment_distillation_fn(
                    student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> torch.Tensor:
                    row_full_seq_len = _full_seq_len_for_row(batch_idx)
                    if full_logit_end >= row_full_seq_len:
                        effective_logit_end = row_full_seq_len - 1
                        student_eff = student_logits[
                            :, : effective_logit_end - full_logit_start, :
                        ]
                        teacher_eff = teacher_logits[
                            :, : effective_logit_end - full_logit_start, :
                        ]
                    else:
                        effective_logit_end = full_logit_end
                        student_eff = student_logits
                        teacher_eff = teacher_logits
                    if effective_logit_end <= full_logit_start:
                        return student_logits.sum() * 0.0

                    distill_labels = _row_slice(
                        labels,
                        batch_idx,
                        full_logit_start,
                        effective_logit_end,
                    )
                    tgt_start = full_logit_start + 1
                    tgt_end = effective_logit_end + 1
                    distill_mask = _row_slice(
                        loss_mask,
                        batch_idx,
                        tgt_start,
                        tgt_end,
                    ).bool()
                    terms = compute_reverse_kl_terms(
                        student_logits=student_eff,
                        teacher_logits=teacher_eff,
                        labels=distill_labels,
                        estimator=distill_cfg.estimator,
                        top_k=distill_cfg.top_k,
                    )

                    reward_signal = terms.kl
                    if distill_adv_adjustment is not None and distill_cfg.reward_coef > 0:
                        shaped_reward = -distill_cfg.reward_coef * reward_signal.detach()
                        _row_assign(
                            distill_adv_adjustment,
                            batch_idx,
                            tgt_start,
                            tgt_end,
                            torch.where(
                                distill_mask,
                                shaped_reward,
                                torch.zeros_like(shaped_reward),
                            ),
                        )
                        _append_distill_metric(
                            "distill/reward",
                            shaped_reward,
                            distill_mask,
                        )

                    _append_distill_metric(
                        "distill/student_teacher_kl",
                        terms.kl,
                        distill_mask,
                    )
                    _append_distill_metric(
                        "distill/reverse_kl",
                        terms.kl,
                        distill_mask,
                    )
                    _append_distill_metric(
                        "distill/k1_sample",
                        terms.selected_k1,
                        distill_mask,
                    )
                    if terms.topk_mass is not None:
                        _append_distill_metric(
                            "distill/topk_mass",
                            terms.topk_mass,
                            distill_mask,
                        )

                    if distill_cfg.loss_coef <= 0 or not distill_mask.any():
                        return student_eff.sum() * 0.0
                    return (
                        distill_cfg.loss_coef
                        * terms.kl[distill_mask].sum()
                        / loss_scale
                    )

                def _segment_selected_logprob_inputs_fn(
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> tuple[torch.Tensor, torch.Tensor, int]:
                    row_full_seq_len = _full_seq_len_for_row(batch_idx)
                    effective_logit_end = min(
                        int(full_logit_end),
                        row_full_seq_len - 1,
                    )
                    if effective_logit_end <= full_logit_start:
                        return (
                            _row_slice(
                                labels,
                                batch_idx,
                                full_logit_start,
                                full_logit_start,
                            ),
                            _row_slice(
                                temperatures,
                                batch_idx,
                                full_logit_start,
                                full_logit_start,
                            ),
                            effective_logit_end,
                        )
                    return (
                        _row_slice(
                            labels,
                            batch_idx,
                            full_logit_start,
                            effective_logit_end,
                        ),
                        _row_slice(
                            temperatures,
                            batch_idx,
                            full_logit_start,
                            effective_logit_end,
                        ),
                        effective_logit_end,
                    )

                def _segment_selected_logprob_loss_fn(
                    seg_raw_logprobs: torch.Tensor,
                    seg_entropy: torch.Tensor,
                    full_logit_start: int,
                    effective_logit_end: int,
                    batch_idx: int = 0,
                ) -> torch.Tensor:
                    if effective_logit_end <= full_logit_start:
                        return seg_raw_logprobs.sum() * 0.0

                    tgt_start = full_logit_start + 1
                    tgt_end = effective_logit_end + 1
                    seg_adv = _row_slice(advantages, batch_idx, tgt_start, tgt_end)
                    seg_mask = _row_slice(loss_mask, batch_idx, tgt_start, tgt_end)
                    seg_inf = _row_slice(
                        inference_logprobs,
                        batch_idx,
                        tgt_start,
                        tgt_end,
                    )
                    seg_teach = (
                        _row_slice(
                            teacher_logprobs,
                            batch_idx,
                            tgt_start,
                            tgt_end,
                        )
                        if teacher_logprobs is not None
                        else None
                    )
                    seg_loss_val, seg_metrics = compute_loss(
                        trainer_logprobs=(seg_raw_logprobs.squeeze(0),),
                        inference_logprobs=(seg_inf.squeeze(0),),
                        teacher_logprobs=(
                            (seg_teach.squeeze(0),) if seg_teach is not None else None
                        ),
                        advantages=(seg_adv.squeeze(0),),
                        loss_mask=(seg_mask.squeeze(0),),
                        loss_fn=loss_fn,
                        loss_scale=loss_scale,
                    )
                    for mk, mv in seg_metrics.items():
                        accumulated_loss_tensors.setdefault(mk, []).append(
                            mv.detach()
                        )
                    with torch.no_grad():
                        seg_loss_mask_1d = seg_mask.squeeze(0).bool()
                        if seg_loss_mask_1d.any():
                            seg_trainer_lp_1d = seg_raw_logprobs.squeeze(0)
                            seg_inf_lp_1d = seg_inf.squeeze(0)
                            seg_adv_1d = seg_adv.squeeze(0)
                            seg_log_ratio = seg_trainer_lp_1d - seg_inf_lp_1d
                            seg_token_kl = (
                                torch.exp(seg_log_ratio) - seg_log_ratio - 1.0
                            )
                            accumulated_loss_tensors.setdefault(
                                "mismatch_kl_token_weighted", []
                            ).append(
                                seg_token_kl[seg_loss_mask_1d]
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            accumulated_loss_tensors.setdefault(
                                "trainer_infer_logprob_delta_token_weighted", []
                            ).append(
                                seg_log_ratio[seg_loss_mask_1d]
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            seg_prob_delta = torch.exp(seg_trainer_lp_1d) - torch.exp(
                                seg_inf_lp_1d
                            )
                            accumulated_loss_tensors.setdefault(
                                "trainer_infer_prob_delta_token_weighted", []
                            ).append(
                                seg_prob_delta[seg_loss_mask_1d]
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )

                            if isinstance(config.loss, DefaultLossConfig):
                                seg_prob_diff = seg_prob_delta
                                seg_dppo_high = (
                                    seg_prob_diff > config.loss.dppo_mask_high
                                )
                                seg_dppo_low = (
                                    seg_prob_diff < -config.loss.dppo_mask_low
                                )
                                seg_dppo_mask = torch.where(
                                    seg_adv_1d > 0,
                                    seg_dppo_high,
                                    seg_dppo_low,
                                )
                                seg_keep_mask = seg_loss_mask_1d & ~seg_dppo_mask
                                seg_masked_mask = seg_loss_mask_1d & seg_dppo_mask
                                if seg_keep_mask.any():
                                    accumulated_loss_tensors.setdefault(
                                        "unmasked_mismatch_kl_token_weighted",
                                        [],
                                    ).append(
                                        seg_token_kl[seg_keep_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "unmasked_trainer_infer_logprob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_log_ratio[seg_keep_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "unmasked_trainer_infer_prob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_prob_delta[seg_keep_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                if seg_masked_mask.any():
                                    accumulated_loss_tensors.setdefault(
                                        "masked_mismatch_kl_token_weighted",
                                        [],
                                    ).append(
                                        seg_token_kl[seg_masked_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "masked_trainer_infer_logprob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_log_ratio[seg_masked_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "masked_trainer_infer_prob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_prob_delta[seg_masked_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )

                    accumulated_entropy_masked.append(
                        seg_entropy.squeeze(0)[seg_mask.squeeze(0).bool()]
                        .detach()
                        .to("cpu")
                    )
                    return seg_loss_val

                def _segment_loss_fn(
                    seg_logits: torch.Tensor,  # [1, seg_owned_logits, vocab]
                    full_logit_start: int,
                    full_logit_end: int,
                    batch_idx: int = 0,
                ) -> torch.Tensor:
                    # Logit at position P predicts token at P+1, so a
                    # segment's owned logit range [full_logit_start,
                    # full_logit_end) predicts target tokens at
                    # [full_logit_start+1, full_logit_end+1). For the
                    # last segment, full_logit_end == full_seq_len, and
                    # the final logit (position full_seq_len-1) would
                    # predict a nonexistent token at full_seq_len. The
                    # standard path discards that logit implicitly via
                    # shift_tensor_right; we do it explicitly here by
                    # dropping the final logit and capping tgt_end at
                    # full_seq_len. Matches the loss contribution of
                    # the standard path exactly.
                    row_full_seq_len = _full_seq_len_for_row(batch_idx)
                    if full_logit_end >= row_full_seq_len:
                        effective_logit_end = row_full_seq_len - 1
                        seg_logits_effective = seg_logits[
                            :, : effective_logit_end - full_logit_start, :
                        ]
                    else:
                        effective_logit_end = full_logit_end
                        seg_logits_effective = seg_logits
                    if effective_logit_end <= full_logit_start:
                        # Degenerate: the segment owns only the final
                        # (meaningless) logit that gets dropped by the
                        # end-of-sequence trim. This can happen when a
                        # compaction event fires at the very last
                        # completion token, producing a 1-token tail
                        # segment whose single logit is trimmed.
                        #
                        # Return a zero loss that's still tied to the
                        # autograd graph so window_loss.backward()
                        # inside segmented_forward has at least one
                        # reachable path. seg_logits is non-empty
                        # here (segmented_forward only creates segments
                        # with seg_start < seg_end), so `seg_logits.sum()
                        # * 0.0` is a scalar with a grad_fn chaining
                        # back through the forward pass. Backward
                        # contributes zero gradient, which is exactly
                        # what we want for a segment with no trainable
                        # tokens.
                        return seg_logits.sum() * 0.0

                    seg_labels = _row_slice(
                        labels,
                        batch_idx,
                        full_logit_start,
                        effective_logit_end,
                    )
                    seg_raw_logprobs = selective_log_softmax(
                        seg_logits_effective, seg_labels
                    )
                    # Target token positions for these logits:
                    #   [full_logit_start + 1, effective_logit_end + 1)
                    # which is always within [1, full_seq_len].
                    tgt_start = full_logit_start + 1
                    tgt_end = effective_logit_end + 1
                    seg_adv = _row_slice(advantages, batch_idx, tgt_start, tgt_end)
                    if distill_adv_adjustment is not None:
                        seg_adv = seg_adv + _row_slice(
                            distill_adv_adjustment,
                            batch_idx,
                            tgt_start,
                            tgt_end,
                        )
                    seg_mask = _row_slice(loss_mask, batch_idx, tgt_start, tgt_end)
                    seg_inf = _row_slice(
                        inference_logprobs,
                        batch_idx,
                        tgt_start,
                        tgt_end,
                    )

                    # ── KVE per-token (V_lp, T_lp) dump for KL scatter
                    # analysis. Each call to _segment_loss_fn corresponds
                    # to ONE call/turn in the per-call dispatch path, so
                    # we tag rows with tgt_start (= unique key within a
                    # sample; pre-eviction calls have smaller tgt_start
                    # than post-eviction calls). Set
                    # KVE_DUMP_LOGPROB_CSV=<path> to enable. By default
                    # this appends one row per loss_mask=True completion
                    # token. Set KVE_DUMP_LOGPROB_FILTER=keep to dump only
                    # tokens that survive DPPO masking and therefore
                    # contribute to the policy-gradient term.
                    _csv_path = os.environ.get("KVE_DUMP_LOGPROB_CSV", "")
                    if _csv_path:
                        _t = seg_raw_logprobs.squeeze(0).detach().float()
                        _i = seg_inf.squeeze(0).detach().float()
                        _a = seg_adv.squeeze(0).detach().float()
                        _loss_m = seg_mask.squeeze(0).bool()
                        if isinstance(config.loss, DefaultLossConfig):
                            _prob_diff = torch.exp(_t) - torch.exp(_i)
                            _dppo_high = _prob_diff > config.loss.dppo_mask_high
                            _dppo_low = _prob_diff < -config.loss.dppo_mask_low
                            _dppo_m = torch.where(_a > 0, _dppo_high, _dppo_low)
                        else:
                            _prob_diff = torch.exp(_t) - torch.exp(_i)
                            _dppo_m = torch.zeros_like(_loss_m)
                        _keep_m = _loss_m & ~_dppo_m
                        _csv_filter = os.environ.get(
                            "KVE_DUMP_LOGPROB_FILTER", "loss_mask"
                        ).lower()
                        if _csv_filter in {"keep", "unmasked", "kept"}:
                            _msk = _keep_m
                        elif _csv_filter in {"dppo_masked", "masked"}:
                            _msk = _loss_m & _dppo_m
                        else:
                            _msk = _loss_m
                        if _msk.any():
                            _t1d = _t.cpu()
                            _i1d = _i.cpu()
                            _a1d = _a.cpu()
                            _pd1d = _prob_diff.cpu()
                            _loss_cpu = _loss_m.cpu()
                            _dppo_cpu = _dppo_m.cpu()
                            _keep_cpu = _keep_m.cpu()
                            _lbl = seg_labels.squeeze(0).detach().cpu()
                            _mcpu = _msk.cpu()
                            _needs_header = (
                                not os.path.exists(_csv_path)
                                or os.path.getsize(_csv_path) == 0
                            )
                            with open(_csv_path, "a") as _f:
                                if _needs_header:
                                    _f.write(
                                        "step,micro_step,tgt_start,pos,token_id,"
                                        "trainer_lp,inference_lp,diff,k3,advantage,"
                                        "prob_diff,loss_mask,dppo_masked,keep_mask\n"
                                    )
                                for _j in range(_t1d.shape[0]):
                                    if not _mcpu[_j].item():
                                        continue
                                    _diff = float(_t1d[_j].item() - _i1d[_j].item())
                                    try:
                                        _k3 = math.exp(_diff) - _diff - 1.0
                                    except OverflowError:
                                        _k3 = float("inf")
                                    _f.write(
                                        f"{progress.step},{micro_step},{tgt_start},"
                                        f"{tgt_start + _j},"
                                        f"{int(_lbl[_j].item())},"
                                        f"{float(_t1d[_j].item()):.6f},"
                                        f"{float(_i1d[_j].item()):.6f},"
                                        f"{_diff:.6f},"
                                        f"{_k3:.6f},"
                                        f"{float(_a1d[_j].item()):.6f},"
                                        f"{float(_pd1d[_j].item()):.6f},"
                                        f"{int(_loss_cpu[_j].item())},"
                                        f"{int(_dppo_cpu[_j].item())},"
                                        f"{int(_keep_cpu[_j].item())}\n"
                                    )
                    seg_teach = (
                        _row_slice(
                            teacher_logprobs,
                            batch_idx,
                            tgt_start,
                            tgt_end,
                        )
                        if teacher_logprobs is not None
                        else None
                    )
                    seg_loss_val, seg_metrics = compute_loss(
                        trainer_logprobs=(seg_raw_logprobs.squeeze(0),),
                        inference_logprobs=(seg_inf.squeeze(0),),
                        teacher_logprobs=(
                            (seg_teach.squeeze(0),) if seg_teach is not None else None
                        ),
                        advantages=(seg_adv.squeeze(0),),
                        loss_mask=(seg_mask.squeeze(0),),
                        loss_fn=loss_fn,
                        loss_scale=loss_scale,
                    )
                    # Accumulate metric tensors across segments for
                    # later logging. compute_loss returns each metric
                    # already stacked/concat'd for its single-sequence
                    # input, so we just stash each segment's tensors
                    # into lists and stack at the end.
                    for mk, mv in seg_metrics.items():
                        accumulated_loss_tensors.setdefault(mk, []).append(
                            mv.detach()
                        )
                    # Token-weighted KL diagnostics for segmented
                    # execution. The built-in loss metrics above are
                    # per-segment means, so downstream aggregation is a
                    # mean-of-means. These values store the per-token k3
                    # KL samples directly; downstream tensor_stats then
                    # reports a true token-weighted mean.
                    with torch.no_grad():
                        seg_loss_mask_1d = seg_mask.squeeze(0).bool()
                        if seg_loss_mask_1d.any():
                            seg_trainer_lp_1d = seg_raw_logprobs.squeeze(0)
                            seg_inf_lp_1d = seg_inf.squeeze(0)
                            seg_adv_1d = seg_adv.squeeze(0)
                            seg_log_ratio = seg_trainer_lp_1d - seg_inf_lp_1d
                            seg_token_kl = (
                                torch.exp(seg_log_ratio) - seg_log_ratio - 1.0
                            )
                            accumulated_loss_tensors.setdefault(
                                "mismatch_kl_token_weighted", []
                            ).append(
                                seg_token_kl[seg_loss_mask_1d]
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            accumulated_loss_tensors.setdefault(
                                "trainer_infer_logprob_delta_token_weighted", []
                            ).append(
                                seg_log_ratio[seg_loss_mask_1d]
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            seg_prob_delta = torch.exp(seg_trainer_lp_1d) - torch.exp(
                                seg_inf_lp_1d
                            )
                            accumulated_loss_tensors.setdefault(
                                "trainer_infer_prob_delta_token_weighted", []
                            ).append(
                                seg_prob_delta[seg_loss_mask_1d]
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )

                            if isinstance(config.loss, DefaultLossConfig):
                                seg_prob_diff = seg_prob_delta
                                seg_dppo_high = (
                                    seg_prob_diff > config.loss.dppo_mask_high
                                )
                                seg_dppo_low = (
                                    seg_prob_diff < -config.loss.dppo_mask_low
                                )
                                seg_dppo_mask = torch.where(
                                    seg_adv_1d > 0,
                                    seg_dppo_high,
                                    seg_dppo_low,
                                )
                                seg_keep_mask = seg_loss_mask_1d & ~seg_dppo_mask
                                seg_masked_mask = seg_loss_mask_1d & seg_dppo_mask
                                if seg_keep_mask.any():
                                    accumulated_loss_tensors.setdefault(
                                        "unmasked_mismatch_kl_token_weighted",
                                        [],
                                    ).append(
                                        seg_token_kl[seg_keep_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "unmasked_trainer_infer_logprob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_log_ratio[seg_keep_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "unmasked_trainer_infer_prob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_prob_delta[seg_keep_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                if seg_masked_mask.any():
                                    accumulated_loss_tensors.setdefault(
                                        "masked_mismatch_kl_token_weighted",
                                        [],
                                    ).append(
                                        seg_token_kl[seg_masked_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "masked_trainer_infer_logprob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_log_ratio[seg_masked_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )
                                    accumulated_loss_tensors.setdefault(
                                        "masked_trainer_infer_prob_delta_token_weighted",
                                        [],
                                    ).append(
                                        seg_prob_delta[seg_masked_mask]
                                        .detach()
                                        .to("cpu", dtype=torch.float32)
                                    )

                    # Per-segment entropy over OWNED target tokens masked
                    # by loss_mask. seg_logits_effective is already
                    # temperature-scaled by segmented_forward and
                    # boundary-trimmed, so it matches exactly the set
                    # of logits whose target tokens this segment owns.
                    #
                    # We INLINE the entropy math instead of calling
                    # prime_rl.trainer.rl.loss.compute_entropy, which
                    # has @torch.compile(dynamic=True). Each segment
                    # produces a different [1, N, vocab] shape, and
                    # torch.compile's dynamic-shape handling does not
                    # generalize across the sizes segmented_forward
                    # produces — first smoke attempt with the compiled
                    # variant stalled all 4 DP ranks for 25+ minutes
                    # on step 0 (Inductor recompile thrash). The
                    # inlined math runs under torch.no_grad so there
                    # is no autograd overhead.
                    with torch.no_grad():
                        lse = torch.logsumexp(seg_logits_effective, dim=-1)
                        pd = torch.nn.functional.softmax(
                            seg_logits_effective, dim=-1
                        )
                        seg_entropy = lse - torch.sum(
                            pd * seg_logits_effective, dim=-1
                        )
                    if kve_top_mismatch_n > 0:
                        call_plan = _kve_find_call_plan(
                            kve_call_plans,
                            int(full_logit_start),
                            int(full_logit_end),
                        )
                        _kve_collect_top_mismatch_records(
                            records=kve_top_mismatch_records,
                            counts=kve_top_mismatch_counts,
                            top_n=kve_top_mismatch_n,
                            scope=kve_top_mismatch_scope,
                            step=progress.step,
                            micro_step=micro_step,
                            full_logit_start=full_logit_start,
                            full_logit_end=effective_logit_end,
                            tgt_start=tgt_start,
                            tgt_end=tgt_end,
                            trainer_logprobs=seg_raw_logprobs,
                            inference_logprobs=seg_inf,
                            labels=seg_labels,
                            advantages=seg_adv,
                            loss_mask=seg_mask,
                            entropy=seg_entropy,
                            loss_config=config.loss,
                            call_plan=call_plan,
                        )
                    if kve_call_mismatch_n > 0:
                        call_plan = _kve_find_call_plan(
                            kve_call_plans,
                            int(full_logit_start),
                            int(full_logit_end),
                        )
                        _kve_collect_call_mismatch_record(
                            records=kve_call_mismatch_records,
                            counts=kve_call_mismatch_counts,
                            scope=kve_call_mismatch_scope,
                            step=progress.step,
                            micro_step=micro_step,
                            full_logit_start=full_logit_start,
                            full_logit_end=effective_logit_end,
                            tgt_start=tgt_start,
                            tgt_end=tgt_end,
                            trainer_logprobs=seg_raw_logprobs,
                            inference_logprobs=seg_inf,
                            labels=seg_labels,
                            advantages=seg_adv,
                            loss_mask=seg_mask,
                            entropy=seg_entropy,
                            loss_config=config.loss,
                            call_plan=call_plan,
                        )
                    accumulated_entropy_masked.append(
                        seg_entropy.squeeze(0)[seg_mask.squeeze(0).bool()]
                        .detach()
                        .to("cpu")
                    )
                    return seg_loss_val

                selected_logprob_flex = False
                if (
                    use_flex_mask
                    and not distill_enabled
                    and isinstance(
                        config.model.fused_lm_head_token_chunk_size,
                        int,
                    )
                    and not os.environ.get("KVE_DUMP_LOGPROB_CSV", "")
                    and kve_top_mismatch_n <= 0
                    and kve_call_mismatch_n <= 0
                ):
                    if horizontal_flex_compaction:
                        selected_logprob_flex = (
                            selected_logprob_packed_flex_mask_segmented_forward
                            is not None
                        )
                    elif batched_flex_compaction:
                        selected_logprob_flex = (
                            selected_logprob_batched_flex_mask_segmented_forward
                            is not None
                        )
                    else:
                        selected_logprob_flex = (
                            selected_logprob_flex_mask_segmented_forward
                            is not None
                        )
                    if selected_logprob_flex:
                        # The historical flex path ignored per-token
                        # temperature. Keep this optimization constrained to
                        # the production-equivalent TextWorld setting we
                        # validated, where temperatures are all 1.0.
                        selected_logprob_flex = bool(
                            torch.all(temperatures == 1).item()
                        )
                if dist.is_initialized() and dist.get_world_size() > 1:
                    selected_t = torch.tensor(
                        [int(selected_logprob_flex)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    dist.all_reduce(selected_t, op=dist.ReduceOp.MIN)
                    selected_logprob_flex = bool(selected_t.item())

                with (
                    maybe_record_function("forward"),
                    maybe_activation_offloading(config.model.ac_offloading),
                ):
                    if use_flex_mask:
                        if selected_logprob_flex:
                            selected_kwargs = dict(
                                model=model,
                                merged_input_ids=input_ids,
                                merged_position_ids=forward_position_ids,
                                inputs_fn=_segment_selected_logprob_inputs_fn,
                                loss_fn=_segment_selected_logprob_loss_fn,
                                max_forward_passes=max_forwards,
                                device=input_ids.device,
                            )
                            if batched_flex_compaction:
                                if horizontal_flex_compaction:
                                    out = (
                                        selected_logprob_packed_flex_mask_segmented_forward(
                                            calls_batch=micro_batch["calls"],
                                            **selected_kwargs,
                                        )
                                    )
                                else:
                                    out = (
                                        selected_logprob_batched_flex_mask_segmented_forward(
                                            calls_batch=micro_batch["calls"],
                                            **selected_kwargs,
                                        )
                                    )
                            else:
                                out = selected_logprob_flex_mask_segmented_forward(
                                    calls=micro_batch["calls"],
                                    **selected_kwargs,
                                )
                        else:
                            flex_kwargs = dict(
                                model=model,
                                merged_input_ids=input_ids,
                                merged_position_ids=forward_position_ids,
                                loss_fn=_segment_loss_fn,
                                max_forward_passes=max_forwards,
                                distillation_fn=(
                                    _segment_distillation_fn
                                    if distill_enabled
                                    and distill_cfg.estimator == "rb_full"
                                    else None
                                ),
                                distillation_prepare_fn=(
                                    _segment_distillation_prepare_fn
                                    if distill_enabled
                                    and distill_cfg.estimator == "rb_topk"
                                    else None
                                ),
                                distillation_teacher_fn=(
                                    _segment_distillation_teacher_fn
                                    if distill_enabled
                                    and distill_cfg.estimator != "rb_full"
                                    else None
                                ),
                                distillation_loss_fn=(
                                    _segment_distillation_loss_fn
                                    if distill_enabled
                                    and distill_cfg.estimator != "rb_full"
                                    else None
                                ),
                                device=input_ids.device,
                            )
                            if batched_flex_compaction:
                                if horizontal_flex_compaction:
                                    out = packed_flex_mask_segmented_forward(
                                        calls_batch=micro_batch["calls"],
                                        **flex_kwargs,
                                    )
                                else:
                                    out = batched_flex_mask_segmented_forward(
                                        calls_batch=micro_batch["calls"],
                                        **flex_kwargs,
                                    )
                            else:
                                out = flex_mask_segmented_forward(
                                    calls=micro_batch["calls"],
                                    **flex_kwargs,
                                )
                    elif use_per_call:
                        out = per_call_segmented_forward(
                            model=model,
                            calls=micro_batch["calls"],
                            merged_input_ids=input_ids,
                            merged_position_ids=forward_position_ids,
                            loss_fn=_segment_loss_fn,
                            max_forward_passes=max_forwards,
                            max_bptt_window_forward_passes=(
                                max_bptt_window_forward_passes
                            ),
                            bptt_segments=config.compaction.bptt_segments,
                            device=input_ids.device,
                        )
                    else:
                        out = segmented_forward(
                            model=model,
                            input_ids=input_ids,
                            position_ids=forward_position_ids,
                            segment_boundaries=segment_boundaries,
                            prompt_len=mb_prompt_len,
                            prompt_aligned_len=prompt_aligned_len,
                            stride=config.compaction.stride,
                            temperature=temperatures,
                            max_forward_passes=max_forwards,
                            loss_fn=_segment_loss_fn,
                            bptt_segments=config.compaction.bptt_segments,
                        )

                if kve_top_mismatch_n > 0:
                    _kve_log_top_mismatch_records(
                        logger,
                        tokenizer,
                        kve_top_mismatch_records,
                        kve_top_mismatch_counts,
                        kve_top_mismatch_n,
                    )
                if kve_call_mismatch_n > 0:
                    _kve_log_call_mismatch_records(
                        logger,
                        tokenizer,
                        kve_call_mismatch_records,
                        kve_call_mismatch_counts,
                        kve_call_mismatch_n,
                    )

                # segmented_forward already ran backward per BPTT
                # window; accumulated_loss is a detached scalar for
                # logging. Aggregate per-segment metrics with
                # torch.cat to match the shape convention the
                # standard path produces.
                #
                # Shape derivation: compute_loss is called once per
                # segment with a SINGLE-sequence one-element tuple
                # input. Inside compute_loss (loss.py:250-255) the
                # per-sequence scalar metrics (_safe_mean output, 0-dim)
                # are wrapped via `torch.stack([scalar_0d])` → 1-D
                # tensor of length 1. So `accumulated_loss_tensors[k]`
                # is a list of shape-[1] 1-D tensors, one per segment.
                # `torch.cat(v)` along dim 0 gives a 1-D tensor of
                # length n_segments — same shape the standard path
                # produces for n_sequences of packed compute_loss
                # input, which is what downstream `compute_stats`
                # (utils.py) expects (it asserts ndim==1 and cats
                # across micro-steps).
                #
                # Known limitation: the built-in loss functions
                # (default_loss_fn, sft_loss_fn) only emit metrics
                # via _safe_mean, which is a per-segment average.
                # Downstream mean-of-stacks gives a mean-of-means,
                # not a token-weighted mean. For a sample whose
                # segments are token-count-skewed the logged metric
                # value diverges from what the standard path would
                # report. The logged LOSS is still correct (sum of
                # per-segment scaled losses == full-sample scaled
                # loss); only the derived metrics like mismatch_kl
                # drift. TODO: switch to weighted (sum, denom)
                # accumulation if this metric skew matters for
                # analysis. We additionally emit
                # *_token_weighted metrics above by storing per-token
                # KL samples directly.
                loss = out["loss"]
                loss_tensors: dict[str, torch.Tensor] = {}
                for k, v in accumulated_loss_tensors.items():
                    # Defensive: prime-rl's compute_loss produces 1-D
                    # metric tensors for its single-sequence input.
                    # A custom loss_fn returning a non-1-D metric
                    # would need a different aggregation path; fail
                    # loudly here instead of silently producing a
                    # shape downstream compute_stats rejects.
                    assert v[0].dim() == 1, (
                        f"Segmented per-segment metrics must be 1-D "
                        f"tensors (got shape {tuple(v[0].shape)} for "
                        f"metric {k!r}); compute_loss wraps scalar "
                        "per-sequence metrics via torch.stack so the "
                        "1-dim is the per-sequence axis. Custom "
                        "loss_fns that break this convention need a "
                        "different segmented aggregation path."
                    )
                    loss_tensors[k] = torch.cat(v)
            else:
                # Forward pass with per-token temperatures (standard path)
                with maybe_record_function("forward"), maybe_activation_offloading(config.model.ac_offloading):
                    out = forward(
                        model,
                        input_ids,
                        forward_position_ids,
                        labels=labels,
                        temperature=temperatures,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        routed_experts=routed_experts,
                    )

                if out.get("logprobs") is None:
                    # VanillaOutputLinear path - compute logprobs externally.
                    assert out.get("logits") is not None, "Logits must be provided to compute logprobs"
                    logits = out["logits"]
                    # Per-token temperature scaling: temperatures is [batch, seq], logits is [batch, seq, vocab]
                    scaled_logits = logits / temperatures.unsqueeze(-1)
                    out["logprobs"] = selective_log_softmax(scaled_logits, labels)
                    out["entropy"] = compute_entropy(scaled_logits)
                # else: FusedOutputLinear was used - logprobs already computed with per-token temperatures

                if cp_enabled:
                    out["logprobs"] = gather_for_cp(out["logprobs"], cp_group)
                    out["entropy"] = gather_for_cp_wo_grad(out["entropy"], cp_size, cp_group)

                vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
                # This is not really necessary as the first token should be masked out, but we do it anyway to be sure
                out["logprobs"] = shift_tensor_right(
                    out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
                )
                out["entropy"] = shift_tensor_right(
                    out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
                )

                # Compute loss. Standard micro-batch stacking introduces a
                # real batch dimension, so flatten and split by the existing
                # packed-position convention instead of using squeeze().
                loss, loss_tensors = compute_loss(
                    trainer_logprobs=split_packed_batch(out["logprobs"], position_ids),
                    inference_logprobs=split_packed_batch(inference_logprobs, position_ids),
                    teacher_logprobs=(
                        split_packed_batch(teacher_logprobs, position_ids)
                        if teacher_logprobs is not None
                        else None
                    ),
                    advantages=split_packed_batch(advantages, position_ids),
                    loss_mask=split_packed_batch(loss_mask, position_ids),
                    loss_fn=loss_fn,
                    loss_scale=loss_scale,
                )

                # Backward pass
                with maybe_record_function("backward"):
                    loss.backward()

            # Add relevant tensors to tensor dict for logging purposes.
            # For the standard path, out["entropy"][loss_mask] is a
            # 1-D tensor of masked per-token entropies. For the
            # segmented path, _segment_loss_fn has already computed
            # per-segment masked entropies and concat'ing them yields
            # the same shape and semantics: each completion token
            # counted exactly once (disjoint owned ranges across
            # segments guarantee this; see the accumulated_entropy_masked
            # init comment above).
            if use_segmented:
                if accumulated_entropy_masked:
                    tensors["entropy"].append(
                        torch.cat(accumulated_entropy_masked)
                    )
                else:
                    # Degenerate sample: every segment short-circuited
                    # to the zero-loss tail branch (all final logits
                    # trimmed, no owned tokens). Append an empty 1-D
                    # tensor with the same dtype the standard path
                    # produces (bfloat16, since compute_entropy
                    # preserves the model's output dtype and compaction
                    # requires flash_attention_2 → half precision).
                    # Dtype consistency matters because
                    # Tensors.compute_stats torch.cat's the entire list
                    # across micro-batches and will fail on a mismatch.
                    tensors["entropy"].append(
                        torch.zeros(0, dtype=torch.bfloat16)
                    )
            else:
                tensors["entropy"].append(out["entropy"][loss_mask].detach().to("cpu"))
            tensors["loss"].append(loss.detach().to("cpu").unsqueeze(0))

            if is_tt_moe_model(model):
                load_balance_stats = get_load_balance_stats(model)
                for k, v in load_balance_stats.items():
                    if v is not None:
                        tensors[k].append(v)

            # Add loss tensors to tensor dict for logging purposes
            for key, loss_tensor in loss_tensors.items():
                loss_tensor = loss_tensor.detach().to("cpu")
                tensors[key].append(loss_tensor)

            # Compaction metrics: track event counts and eviction volume.
            # Under per-call dispatch the admission events live INSIDE
            # call.compaction_events (the outer sample.compaction_events
            # gets emptied by _apply_admission_trim). Aggregate from both
            # sources so the metric reflects all events that fired.
            metric_event_rows = compaction_metric_events_by_row(
                compaction_events,
                micro_batch.get("calls"),
                include_call_events=(use_per_call or use_flex_mask),
            )
            metric_counts = [len(events) for events in metric_event_rows]
            if any(metric_counts) or use_segmented:
                evicted_counts = [
                    sum(getattr(event, "tokens_evicted", 0) for event in events)
                    for events in metric_event_rows
                ]
                tensors["compaction/num_events"].append(
                    torch.tensor(metric_counts, dtype=torch.float32)
                )
                tensors["compaction/tokens_evicted"].append(
                    torch.tensor(evicted_counts, dtype=torch.float32)
                )

            # Debug log with *local, micro step* stats. Entropy is now
            # computed for both the standard path (out["entropy"][loss_mask])
            # and the segmented path (per-segment compute_entropy over
            # owned target tokens, concatenated). Both paths append a
            # 1-D tensor per micro-batch, so indexing [-1] is always
            # safe as long as the list is non-empty. On the rare
            # degenerate segmented case (all segments had zero owned
            # tokens), the appended tensor is zero-length and .mean()
            # returns NaN — acceptable for a debug log.
            micro_step_message = (
                f"Micro Step {micro_step}/{len(micro_batch_groups)} "
                f"(stack={stacked_micro_batches}) | "
                f"Loss: {tensors['loss'][-1].mean().item():.4f}"
            )
            if len(tensors["entropy"]) > 0 and tensors["entropy"][-1].numel() > 0:
                micro_step_message += (
                    f" | Entropy: {tensors['entropy'][-1].mean().item():.4f}"
                )
            if "mismatch_kl" in tensors and len(tensors["mismatch_kl"]) > 0:
                micro_step_message += f" | Mismatch KL: {tensors['mismatch_kl'][-1].mean().item():.4f}"
            if (
                "mismatch_kl_token_weighted" in tensors
                and len(tensors["mismatch_kl_token_weighted"]) > 0
            ):
                micro_step_message += (
                    " | Token KL: "
                    f"{tensors['mismatch_kl_token_weighted'][-1].mean().item():.4f}"
                )
            if "max_vio" in tensors and len(tensors["max_vio"]) > 0:
                micro_step_message += f" | Max Vio: {tensors['max_vio'][-1].mean().item():.4f}"
            logger.debug(micro_step_message)

            if _KVE_MEM_TRACE or _KVE_STACK_TRACE:
                _dt_part = (
                    f" dt={time.perf_counter() - _trace_mb_start_time:.2f}s"
                    if _trace_mb_start_time is not None
                    else ""
                )
                logger.info(
                    _trace_snap(
                        f"mb {micro_step}/{len(micro_batch_groups)} "
                        f"{_compaction_flag} stack={stacked_micro_batches} "
                        f"shape={tuple(micro_batch['input_ids'].shape)}"
                        f"{_stack_trace} EXIT{_dt_part}"
                    )
                )

        if _KVE_MEM_TRACE:
            logger.info(_mem_snap(f"step {progress.step} PRE-OPTIM"))

        # Optionally, clip the gradients
        grad_norm: torch.Tensor | None = None
        if config.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(
                model.parameters(), max_norm=config.optim.max_norm, ep_enabled=parallel_dims.ep_enabled
            )
            if grad_norm.device.type == "cpu":
                grad_norm = grad_norm.to(torch.device("cuda"))

        zero_grad_ratio = get_zero_gradient_ratio(model.parameters(), parallel_dims.dp_replicate)

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()
        if _KVE_MEM_TRACE:
            logger.info(_mem_snap(f"step {progress.step} POST-OPTIM"))

        # Update learning rate scheduler
        scheduler.step()

        if config.max_concurrent_runs == 1:
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.get_current_lr()
        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        tensor_stats = tensors.compute_stats()

        # Compute step metrics
        num_local_tokens = sum(int(micro_batch["input_ids"].numel()) for micro_batch in micro_batches)
        num_tokens = parallel_dims.get_mesh("dp").size() * num_local_tokens
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        perf_counter = get_perf_counter(model, seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3  # GiB

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {tensor_stats['loss/mean']:.4f} | Entropy: {tensor_stats['entropy/mean']:.4f}"
        if "mismatch_kl/mean" in tensor_stats:
            step_message += f" | Mismatch KL: {tensor_stats['mismatch_kl/mean']:.4f}"
        if "unmasked_mismatch_kl/mean" in tensor_stats:
            step_message += f" | Unmasked KL: {tensor_stats['unmasked_mismatch_kl/mean']:.4f}"
        if "masked_mismatch_kl/mean" in tensor_stats:
            step_message += f" | Masked KL: {tensor_stats['masked_mismatch_kl/mean']:.4f}"
        if "is_masked/mean" in tensor_stats:
            step_message += f" | Masked Frac: {tensor_stats['is_masked/mean']:.3f}"
        if "unmasked_trainer_infer_logprob_delta/mean" in tensor_stats:
            step_message += (
                " | Unmasked dLP: "
                f"{tensor_stats['unmasked_trainer_infer_logprob_delta/mean']:.4f}"
            )
        if "mismatch_kl_token_weighted/mean" in tensor_stats:
            step_message += f" | Token KL: {tensor_stats['mismatch_kl_token_weighted/mean']:.4f}"
        if "unmasked_mismatch_kl_token_weighted/mean" in tensor_stats:
            step_message += (
                " | Token Unmasked KL: "
                f"{tensor_stats['unmasked_mismatch_kl_token_weighted/mean']:.4f}"
            )
        if "masked_mismatch_kl_token_weighted/mean" in tensor_stats:
            step_message += (
                " | Token Masked KL: "
                f"{tensor_stats['masked_mismatch_kl_token_weighted/mean']:.4f}"
            )
        if "unmasked_trainer_infer_logprob_delta_token_weighted/mean" in tensor_stats:
            step_message += (
                " | Token Unmasked dLP: "
                f"{tensor_stats['unmasked_trainer_infer_logprob_delta_token_weighted/mean']:.4f}"
            )
        if "masked_trainer_infer_logprob_delta_token_weighted/mean" in tensor_stats:
            step_message += (
                " | Token Masked dLP: "
                f"{tensor_stats['masked_trainer_infer_logprob_delta_token_weighted/mean']:.4f}"
            )
        if "distill/student_teacher_kl/mean" in tensor_stats:
            step_message += (
                " | Student-Teacher KL: "
                f"{tensor_stats['distill/student_teacher_kl/mean']:.4f}"
            )
        if "distill/k1_sample/mean" in tensor_stats:
            step_message += (
                " | ST K1: "
                f"{tensor_stats['distill/k1_sample/mean']:.4f}"
            )
        if "distill/topk_mass/mean" in tensor_stats:
            step_message += (
                " | ST TopK Mass: "
                f"{tensor_stats['distill/topk_mass/mean']:.3f}"
            )
        if grad_norm is not None:
            step_message += f" | Grad. Norm: {grad_norm:.4f}"
        step_message += f" | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f} GiB"
        if "max_vio/mean" in tensor_stats:
            step_message += f" | Max Vio: {tensor_stats['max_vio/mean']:.4f}"
        if "compaction/num_events/mean" in tensor_stats:
            step_message += (
                f" | Compactions: {tensor_stats['compaction/num_events/mean']:.1f} avg"
                f" / {tensor_stats['compaction/num_events/max']:.0f} max"
                f" | Evicted: {tensor_stats['compaction/tokens_evicted/mean']:.0f} avg"
            )
        logger.success(step_message)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "perf/peak_memory": peak_memory,
            "step": progress.step,
        }
        monitor.log(perf_metrics, step=progress.step)

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "optim/zero_grad_ratio": zero_grad_ratio,
            "step": progress.step,
        }
        if grad_norm is not None:
            optim_metrics["optim/grad_norm"] = grad_norm.item()
        monitor.log(optim_metrics, step=progress.step)

        # Compute derived metrics
        entropy_mean = tensor_stats.get("entropy/mean", 0.0)
        mismatch_kl_mean = tensor_stats.get("mismatch_kl/mean")
        if mismatch_kl_mean is not None and entropy_mean > 0:
            tensor_stats["kl_ent_ratio/mean"] = mismatch_kl_mean / entropy_mean

        tensor_stats["step"] = progress.step
        monitor.log(filter_rl_trainer_tensor_stats_for_wandb(tensor_stats), step=progress.step)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/wait_for_batch": wait_for_batch_time,
            "time/load_data": load_data_time,
            "time/broadcast_weights": broadcast_weights_time,
            "time/save_ckpt": save_ckpt_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics, step=progress.step)

        # Log disk metrics
        disk_metrics = get_ckpt_disk_metrics(config.output_dir)
        disk_metrics["step"] = progress.step
        monitor.log(disk_metrics, step=progress.step)

        # Update Prometheus metrics if configured
        if metrics_server is not None:
            metrics_server.update(
                step=progress.step,
                loss=tensor_stats["loss/mean"],
                throughput=throughput,
                grad_norm=grad_norm.item() if grad_norm is not None else None,
                peak_memory_gib=peak_memory,
                learning_rate=current_lr,
                mfu=mfu,
                entropy=tensor_stats.get("entropy/mean", 0.0),
                mismatch_kl=tensor_stats.get("mismatch_kl/mean", 0.0),
                zero_grad_ratio=zero_grad_ratio,
            )
            # Update run/LoRA metrics
            multi_run_manager = get_multi_run_manager()
            runs_discovered = len(list(config.output_dir.glob("run_*")))
            run_stats = []
            for idx in multi_run_manager.used_idxs:
                run_id = multi_run_manager.idx_2_id[idx]
                run_progress = multi_run_manager.progress[idx]
                if config.max_concurrent_runs == 1:
                    lr = optimizer.param_groups[0]["lr"]
                else:
                    lr = optimizer.get_current_lr(idx) if optimizer.optimizers[idx] else 0.0
                run_stats.append(
                    RunStats(
                        run_id=run_id,
                        step=run_progress.step,
                        total_tokens=run_progress.total_tokens,
                        learning_rate=lr,
                        ready=multi_run_manager.ready_to_update[idx],
                    )
                )
            metrics_server.update_runs(
                runs_discovered=runs_discovered,
                runs_max=multi_run_manager.max_runs,
                run_stats=run_stats,
            )

        progress.step += 1
        is_first_step = False

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    # Write final checkpoint (only for single-run mode; multi-run checkpoints are managed by MultiCheckpointManager)
    if config.max_concurrent_runs == 1 and ckpt_manager is not None:
        gc.collect()
        torch.cuda.empty_cache()
        if not (config.ckpt and config.ckpt.weights_only):
            logger.info("Writing final checkpoint")
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
        ckpt_manager.maybe_clean()

    if config.max_concurrent_runs == 1 and weight_ckpt_manager is not None:
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(progress.step, model, tokenizer)
        weight_ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("RL trainer finished!")

    # Stop metrics/health server if configured
    if metrics_server is not None:
        metrics_server.stop()
    if health_server is not None:
        health_server.stop()

    # Optionally, print benchmark table and export JSON
    if config.bench is not None and world.is_master:
        history = to_col_format(monitor.history)
        print_benchmark(history)
        if config.bench.output_json:
            export_benchmark_json(history, config.bench.output_json)
            logger.info(f"Benchmark results written to {config.bench.output_json}")


def main():
    """Main entry-point for RL trainer. Run using `uv run trainer`.

    Debugging: set ``PRIME_RL_TRAINER_DEBUGPY=<port>`` (default 5679) to
    wait for a VS Code debugger to attach, or ``PRIME_RL_TRAINER_RPDB=<port>``
    (default 4445) to drop into an rpdb gate (connect via ``nc 127.0.0.1
    <port>``). Different default ports from the orchestrator's gate so
    both can run simultaneously. The /opt/toolkit/libpretend LD_PRELOAD
    shim blocks debugpy's adapter — we clear it temporarily during
    debugpy.listen.
    """
    set_proc_title("Trainer")
    import os as _os
    _dp = _os.environ.get("PRIME_RL_TRAINER_DEBUGPY", "").strip()
    if _dp:
        _orig_ld = _os.environ.pop("LD_PRELOAD", None)
        import debugpy  # type: ignore[import-not-found]
        port = int(_dp) if _dp.isdigit() else 5679
        host = _os.environ.get("PRIME_RL_DEBUGPY_HOST", "127.0.0.1")
        debugpy.listen((host, port))
        if _orig_ld is not None:
            _os.environ["LD_PRELOAD"] = _orig_ld
        print(
            f"[trainer] debugpy listening on {host}:{port} — "
            "attach from VS Code / Cursor (Python: Remote Attach).",
            flush=True,
        )
        debugpy.wait_for_client()
        print("[trainer] debugger attached, continuing.", flush=True)
    _rp = _os.environ.get("PRIME_RL_TRAINER_RPDB", "").strip()
    if _rp:
        import rpdb  # type: ignore[import-not-found]
        port = int(_rp) if _rp.isdigit() else 4445
        print(
            f"[trainer] rpdb gate armed on 127.0.0.1:{port} — "
            f"connect with: nc 127.0.0.1 {port}",
            flush=True,
        )
        rpdb.set_trace(port=port)
    train(cli(TrainerConfig))


if __name__ == "__main__":
    main()
