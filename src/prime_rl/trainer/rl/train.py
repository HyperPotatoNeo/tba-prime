import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

from contextlib import nullcontext
import os
import time

import torch

# D5 debug: memory trace at every micro-batch boundary. Guarded by env
# var so non-debug runs are unaffected. When KVE_MEM_TRACE=1, each
# micro-batch logs its path, pre/post-forward/backward memory_allocated
# and max_memory_allocated, so we can see exactly where memory grows
# across the compaction -> text transition on step 1 (see D5
# investigation notes in plans/phase3_training_integration.md).
_KVE_MEM_TRACE = os.environ.get("KVE_MEM_TRACE") == "1"
_SEGMENTED_LOGPROB_CHUNK_TOKENS = max(
    1, int(os.environ.get("PRIME_RL_SEGMENTED_LOGPROB_CHUNK_TOKENS", "512"))
)
_SEGMENTED_ENTROPY_CHUNK_TOKENS = max(
    1, int(os.environ.get("PRIME_RL_SEGMENTED_ENTROPY_CHUNK_TOKENS", "128"))
)
_KVE_DEBUG_KL_BY_EVENT = os.environ.get("KVE_DEBUG_KL_BY_EVENT") == "1"


def _mem_snap(tag: str) -> str:
    if not _KVE_MEM_TRACE:
        return ""
    import torch as _t
    alloc = _t.cuda.memory_allocated() / 1e9
    peak = _t.cuda.max_memory_allocated() / 1e9
    reserved = _t.cuda.memory_reserved() / 1e9
    return f"[MEM] {tag} alloc={alloc:.3f}GB peak={peak:.3f}GB reserved={reserved:.3f}GB"


def _selective_log_softmax_seq_chunked(
    logits: torch.Tensor,
    index: torch.Tensor,
    *,
    chunk_tokens: int = _SEGMENTED_LOGPROB_CHUNK_TOKENS,
) -> torch.Tensor:
    def _selective_fp32(
        chunk_logits: torch.Tensor,
        chunk_index: torch.Tensor,
    ) -> torch.Tensor:
        # vLLM reports raw logprobs by converting BF16 logits to FP32 before
        # log-softmax. Match that numerics here; BF16 log_softmax inflates the
        # train-vs-inference KL even when the KV replay is correct.
        chunk_logprobs = chunk_logits.float().log_softmax(dim=-1)
        return torch.gather(
            chunk_logprobs,
            dim=-1,
            index=chunk_index.unsqueeze(-1),
        ).squeeze(-1)

    if logits.shape[1] <= chunk_tokens:
        return _selective_fp32(logits, index)

    pieces = []
    for start in range(0, logits.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, logits.shape[1])
        pieces.append(_selective_fp32(logits[:, start:end, :], index[:, start:end]))
    return torch.cat(pieces, dim=1)


def _masked_entropy_seq_chunked(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    chunk_tokens: int = _SEGMENTED_ENTROPY_CHUNK_TOKENS,
) -> torch.Tensor:
    pieces = []
    with torch.no_grad():
        for start in range(0, logits.shape[1], chunk_tokens):
            end = min(start + chunk_tokens, logits.shape[1])
            chunk_logits = logits[:, start:end, :]
            chunk_mask = mask[:, start:end].bool()
            if not chunk_mask.any():
                continue
            lse = torch.logsumexp(chunk_logits, dim=-1)
            pd = torch.nn.functional.softmax(chunk_logits, dim=-1)
            chunk_entropy = lse - torch.sum(pd * chunk_logits, dim=-1)
            pieces.append(chunk_entropy[chunk_mask])

    if pieces:
        return torch.cat(pieces, dim=0)
    return torch.empty(0, device=logits.device, dtype=logits.dtype)
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.trainer.models.layers.attn import substitute_ring_attn
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.multi_ckpt import setup_multi_checkpoint_manager
from prime_rl.trainer.optim import setup_optimizer, setup_multi_optimizer
from prime_rl.trainer.scheduler import setup_scheduler, setup_multi_scheduler
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader
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
from prime_rl.trainer.model import (
    forward,
    setup_tokenizer,
    setup_model,
    is_tt_moe_model,
    get_load_balance_stats,
)

try:
    from kv_eviction.segmented_forward import (
        compute_num_segments,
        compute_num_per_call_forwards,
        per_call_segmented_forward,
        segmented_forward,
    )
except ImportError:
    # kv_eviction is optional; only required when TrainerConfig.compaction
    # is enabled and TrainingSamples carry compaction_events. Import failure
    # is tolerated at module load; the assertion at the dispatch site will
    # raise with a clearer error if compaction is actually used.
    segmented_forward = None  # type: ignore[assignment]
    compute_num_segments = None  # type: ignore[assignment]
    per_call_segmented_forward = None  # type: ignore[assignment]
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
    get_response_lengths,
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

    # Early check: reject finite TBPTT(K>1) under multi-rank FSDP2.
    # Compaction's per-segment backward mode issues one backward per
    # BPTT window, and the number of real backwards per rank is
    # ceil(per_rank_segments / K). Under DP > 1 this count varies
    # across ranks and the dummy-pass padding inside segmented_forward
    # can't compensate, which would deadlock on reduce-scatter at the
    # first compaction micro-batch. Fail at training init (before any
    # model load or data fetch) so a misconfigured run crashes
    # immediately instead of after a few mixed-sample steps.
    #
    # Full BPTT (bptt_segments=None) is allowed: segmented_forward pads
    # shorter ranks with zero-loss dummy forward graphs before the single
    # trajectory backward, so both FSDP forward and backward hook counts match.
    # TODO: implement max_backwards all-reduce + mixed dummy padding for
    # finite K > 1 TBPTT under FSDP2.
    if (
        config.compaction.window_size > 0
        and config.compaction.bptt_segments not in (1, None)
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        raise ValueError(
            "Compaction with finite trainer.compaction.bptt_segments > 1 is not "
            "yet supported under multi-rank FSDP2 data parallelism due to "
            "a reduce-scatter count mismatch in the dummy-pass padding "
            "inside segmented_forward. Got world_size="
            f"{dist.get_world_size()}, bptt_segments="
            f"{config.compaction.bptt_segments}. Set "
            "trainer.compaction.bptt_segments=1 (the default) or "
            "bptt_segments='full' for multi-rank compaction runs."
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

        batch_size = len(micro_batches)
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

        for micro_step, micro_batch in enumerate(micro_batches):
            if _KVE_MEM_TRACE:
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
                _seq = micro_batch["input_ids"].shape[1]
                torch.cuda.reset_peak_memory_stats()
                logger.info(
                    _mem_snap(
                        f"mb {micro_step}/{len(micro_batches)} "
                        f"{_compaction_flag} seq={_seq} ENTRY"
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

            lora_num_tokens = None
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
            calls = micro_batch.get("calls") or []
            # Per-call replay is intentionally conservative here: use it for
            # full-context multi-turn traces, and leave compaction-bearing AM
            # samples on the established segmented replay path. That isolates
            # DiscoveryWorld full-context KL without changing AM semantics.
            use_per_call = bool(calls) and not bool(compaction_events)
            use_segmented = config.compaction.window_size > 0
            dispatch_kind = 2 if use_per_call else (1 if use_segmented else 0)

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
                    [dispatch_kind], device="cuda", dtype=torch.int32
                )
                max_b = bool_tensor.clone()
                min_b = bool_tensor.clone()
                dist.all_reduce(max_b, op=dist.ReduceOp.MAX)
                dist.all_reduce(min_b, op=dist.ReduceOp.MIN)
                assert int(max_b.item()) == int(min_b.item()), (
                    "Rank-level divergence in the replay/standard "
                    "dispatch branch. prepare_batch is supposed to enforce "
                    "modality uniformity across DP ranks at each step index "
                    "via per-group padding, so this should be unreachable. "
                    "A divergence here would otherwise deadlock at the next "
                    "FSDP all-gather or the segmented all_reduce."
                )

            if use_segmented or use_per_call:
                assert segmented_forward is not None, (
                    "Micro batch carries compaction_events but the "
                    "kv_eviction package is not importable. Install kv_eviction "
                    "or disable compaction on the inference engine."
                )
                if use_per_call:
                    assert per_call_segmented_forward is not None, (
                        "Micro batch carries per-call replay metadata but the "
                        "kv_eviction package is not importable."
                    )
                    assert compute_num_per_call_forwards is not None, (
                        "per-call replay needs compute_num_per_call_forwards."
                    )
                if use_segmented:
                    assert config.compaction.window_size > 0, (
                        "Micro batch has compaction_events but trainer config "
                        "compaction.window_size is 0. The trainer's compaction "
                        "config must mirror the inference engine's config."
                    )
                assert config.model.attn == "flash_attention_2", (
                    f"Compaction requires attn='flash_attention_2' to match "
                    f"vLLM's inference kernel numerics and avoid spurious "
                    f"train-vs-inference KL. Got attn={config.model.attn!r}. "
                    f"Set trainer.model.attn='flash_attention_2' in the config."
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
                # FSDP2 + multi-rank bptt_segments != 1 constraint is
                # already rejected at training init (see the early
                # check after get_parallel_dims); no per-micro-batch
                # re-check needed here.
                mb_prompt_len = micro_batch.get("prompt_len")
                if use_segmented:
                    assert mb_prompt_len is not None, (
                        "prompt_len missing on micro_batch in compaction run "
                        "(compaction.window_size > 0). prepare_sample should "
                        "set prompt_len on EVERY sample when compaction_enabled "
                        "is True; a missing value points to a wire-format or "
                        "packer regression."
                    )
                # Compaction samples MUST be batch-1. The packer
                # (prepare_batch in trainer/batch.py) isolates each
                # compaction sample into its own micro-batch precisely
                # so segment-range bookkeeping and per-sample KV
                # eviction don't have to handle bin-packing. The
                # _segment_loss_fn closure below uses .squeeze(0) on
                # [1, seq] slices which would silently misinterpret a
                # batch-2+ tensor as a single sequence; this assert
                # catches any packer regression before it corrupts
                # training.
                assert input_ids.shape[0] == 1, (
                    f"Compaction samples must be batch_size=1 (packer "
                    f"invariant), got input_ids.shape={tuple(input_ids.shape)}. "
                    f"Check prepare_batch / _is_compaction_sample partitioning."
                )
                seq_len_local = input_ids.shape[1]
                if use_per_call:
                    n_forwards_local = compute_num_per_call_forwards(calls)
                else:
                    bs = config.compaction.block_size
                    prompt_aligned_len = ((mb_prompt_len + bs - 1) // bs) * bs
                    segment_boundaries = [
                        int(e.num_output_tokens_at_compaction)
                        for e in compaction_events
                    ]
                    n_forwards_local = compute_num_segments(
                        seq_len_local,
                        mb_prompt_len,
                        segment_boundaries,
                        compaction_strategy=config.compaction.strategy,
                        compaction_events=compaction_events,
                        attention_matching_decode_chunk_size=(
                            config.compaction.attention_matching_decode_chunk_size
                        ),
                    )
                if dist.is_initialized() and dist.get_world_size() > 1:
                    n_forwards_t = torch.tensor(
                        [n_forwards_local], device="cuda", dtype=torch.int32
                    )
                    dist.all_reduce(n_forwards_t, op=dist.ReduceOp.MAX)
                    max_forwards = int(n_forwards_t.item())
                else:
                    max_forwards = n_forwards_local

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

                def _segment_loss_fn(
                    seg_logits: torch.Tensor,  # [1, seg_owned_logits, vocab]
                    full_logit_start: int,
                    full_logit_end: int,
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
                    if full_logit_end >= full_seq_len:
                        effective_logit_end = full_seq_len - 1
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

                    seg_labels = labels[:, full_logit_start:effective_logit_end]
                    seg_raw_logprobs = _selective_log_softmax_seq_chunked(
                        seg_logits_effective, seg_labels
                    )
                    # Target token positions for these logits:
                    #   [full_logit_start + 1, effective_logit_end + 1)
                    # which is always within [1, full_seq_len].
                    tgt_start = full_logit_start + 1
                    tgt_end = effective_logit_end + 1
                    seg_adv = advantages[:, tgt_start:tgt_end]
                    seg_mask = loss_mask[:, tgt_start:tgt_end]
                    seg_inf = inference_logprobs[:, tgt_start:tgt_end]
                    seg_teach = (
                        teacher_logprobs[:, tgt_start:tgt_end]
                        if teacher_logprobs is not None
                        else None
                    )
                    # For segmented replay, report the primary train-vs-inference
                    # KL as a token-weighted distribution. compute_loss returns
                    # per-segment scalar means, and averaging those would
                    # overweight tiny AM replay segments.
                    seg_log_importance_ratio = seg_raw_logprobs - seg_inf
                    seg_mismatch_kl = (
                        torch.exp(seg_log_importance_ratio)
                        - seg_log_importance_ratio
                        - 1
                    )
                    if _KVE_DEBUG_KL_BY_EVENT:
                        seg_mask_count = int(seg_mask.sum().item())
                        if seg_mask_count > 0:
                            masked_kl = seg_mismatch_kl[seg_mask]
                            seg_kl_mean = float(masked_kl.mean().item())
                            seg_kl_max = float(masked_kl.max().item())
                            masked_positions = torch.nonzero(
                                seg_mask.squeeze(0), as_tuple=False
                            ).flatten()
                            max_local_idx = int(masked_kl.argmax().item())
                            max_mask_offset = int(
                                masked_positions[max_local_idx].item()
                            )
                            max_target_pos = int(
                                tgt_start + max_mask_offset
                            )
                            max_token_id = int(input_ids[0, max_target_pos].item())
                            max_old_logp = float(
                                seg_inf[0, max_mask_offset].detach().item()
                            )
                            max_new_logp = float(
                                seg_raw_logprobs[0, max_mask_offset]
                                .detach()
                                .item()
                            )
                            first_mask_offset = int(masked_positions[0].item())
                            first_target_pos = int(
                                tgt_start + first_mask_offset
                            )
                            first_token_id = int(input_ids[0, first_target_pos].item())
                            first_old_logp = float(
                                seg_inf[0, first_mask_offset].detach().item()
                            )
                            first_new_logp = float(
                                seg_raw_logprobs[0, first_mask_offset]
                                .detach()
                                .item()
                            )
                            first_kl = float(masked_kl[0].item())
                        else:
                            seg_kl_mean = 0.0
                            seg_kl_max = 0.0
                            max_target_pos = -1
                            max_token_id = -1
                            max_old_logp = 0.0
                            max_new_logp = 0.0
                            first_target_pos = -1
                            first_token_id = -1
                            first_old_logp = 0.0
                            first_new_logp = 0.0
                            first_kl = 0.0
                        logger.info(
                            f"[KLSEG] logits=({full_logit_start},{full_logit_end}) "
                            f"targets=({tgt_start},{tgt_end}) "
                            f"mask_tokens={seg_mask_count} "
                            f"mismatch_kl={seg_kl_mean:.6f} "
                            f"max_kl={seg_kl_max:.6f}@{max_target_pos} "
                            f"max_tok={max_token_id} "
                            f"max_old={max_old_logp:.6f} "
                            f"max_new={max_new_logp:.6f} "
                            f"max_delta={max_new_logp - max_old_logp:.6f} "
                            f"first_kl={first_kl:.6f}@{first_target_pos} "
                            f"first_tok={first_token_id} "
                            f"first_old={first_old_logp:.6f} "
                            f"first_new={first_new_logp:.6f} "
                            f"first_delta={first_new_logp - first_old_logp:.6f}"
                        )
                    accumulated_loss_tensors.setdefault("mismatch_kl", []).append(
                        seg_mismatch_kl[seg_mask].detach()
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
                        if mk == "mismatch_kl":
                            continue
                        accumulated_loss_tensors.setdefault(mk, []).append(
                            mv.detach()
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
                    accumulated_entropy_masked.append(
                        _masked_entropy_seq_chunked(seg_logits_effective, seg_mask)
                        .detach()
                        .to("cpu")
                    )
                    return seg_loss_val

                with (
                    maybe_record_function("forward"),
                    maybe_activation_offloading(config.model.ac_offloading),
                ):
                    if use_per_call:
                        out = per_call_segmented_forward(
                            model=model,
                            calls=calls,
                            merged_input_ids=input_ids,
                            merged_position_ids=forward_position_ids,
                            max_forward_passes=max_forwards,
                            loss_fn=_segment_loss_fn,
                            bptt_segments=config.compaction.bptt_segments,
                            device=torch.device("cuda"),
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
                            compaction_events=compaction_events,
                            compaction_strategy=config.compaction.strategy,
                            attention_matching_query_source=(
                                config.compaction.attention_matching_query_source
                            ),
                            attention_matching_max_queries_per_kv_head=(
                                config.compaction.attention_matching_max_queries_per_kv_head
                            ),
                            attention_matching_zerobeta=(
                                config.compaction.attention_matching_zerobeta
                            ),
                            attention_matching_forget_gate_enabled=(
                                config.compaction.attention_matching_forget_gate_enabled
                            ),
                            attention_matching_forget_gate_alpha=(
                                config.compaction.attention_matching_forget_gate_alpha
                            ),
                            attention_matching_gradient_mode=(
                                config.compaction.attention_matching_gradient_mode
                            ),
                            attention_matching_decode_chunk_size=(
                                config.compaction.attention_matching_decode_chunk_size
                            ),
                            lora_num_tokens=lora_num_tokens,
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
                # analysis.
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
                if _KVE_DEBUG_KL_BY_EVENT and "mismatch_kl" in loss_tensors:
                    kl_values = loss_tensors["mismatch_kl"]
                    hit_depths: list[int] = []
                    hit_gaps: list[int] = []
                    event_boundaries: list[int] = []
                    event_summaries: list[str] = []
                    for event in compaction_events:
                        boundary = int(
                            getattr(event, "num_output_tokens_at_compaction", 0)
                            or 0
                        )
                        event_boundaries.append(boundary)
                        cache_hit_tokens = int(
                            getattr(event, "attention_matching_cache_hit_tokens", 0)
                            or 0
                        )
                        if cache_hit_tokens > 0:
                            replay_steps = (
                                getattr(
                                    event,
                                    "attention_matching_replay_steps",
                                    None,
                                )
                                or []
                            )
                            target_len = int(
                                getattr(event, "target_len", 0) or 0
                            )
                            hit_depths.append(len(replay_steps))
                            hit_gaps.append(target_len - cache_hit_tokens)
                        replay_steps = (
                            getattr(event, "attention_matching_replay_steps", None)
                            or []
                        )
                        step_summary = ""
                        if replay_steps:
                            last_step = replay_steps[-1]
                            step_summary = (
                                f";last={int(last_step.get('source_len', 0) or 0)}"
                                f"->{int(last_step.get('target_len', 0) or 0)}"
                            )
                        event_summaries.append(
                            "b="
                            f"{boundary}:src={int(getattr(event, 'source_len', 0) or 0)}"
                            f"->tgt={int(getattr(event, 'target_len', 0) or 0)}"
                            f":hit={cache_hit_tokens}"
                            f":steps={len(replay_steps)}"
                            f"{step_summary}"
                        )
                    kl_mean = float(
                        kl_values.mean().item() if kl_values.numel() > 0 else 0.0
                    )
                    prompt_for_log = (
                        int(mb_prompt_len)
                        if mb_prompt_len is not None
                        else int(len(calls[0].submitted_prompt_ids))
                        if calls
                        else -1
                    )
                    logger.info(
                        f"[KLDEBUG] mb={micro_step}/{len(micro_batches)} "
                        f"seq={int(input_ids.shape[1])} prompt={prompt_for_log} "
                        f"loss_tokens={int(loss_mask.sum().item())} "
                        f"events={len(compaction_events)} boundaries={event_boundaries} "
                        f"hit_depths={hit_depths} hit_gaps={hit_gaps} "
                        f"event_details={event_summaries} "
                        f"mismatch_kl={kl_mean:.6f}"
                    )
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

                # Compute loss
                response_lengths = get_response_lengths(position_ids)
                loss, loss_tensors = compute_loss(
                    trainer_logprobs=out["logprobs"].squeeze().split(response_lengths),
                    inference_logprobs=inference_logprobs.squeeze().split(response_lengths),
                    teacher_logprobs=teacher_logprobs.squeeze().split(response_lengths)
                    if teacher_logprobs is not None
                    else None,
                    advantages=advantages.squeeze().split(response_lengths),
                    loss_mask=loss_mask.squeeze().split(response_lengths),
                    loss_fn=loss_fn,
                    loss_scale=loss_scale,
                )

                if _KVE_DEBUG_KL_BY_EVENT:
                    std_log_importance_ratio = out["logprobs"] - inference_logprobs
                    std_mismatch_kl = (
                        torch.exp(std_log_importance_ratio)
                        - std_log_importance_ratio
                        - 1
                    )
                    std_mask_count = int(loss_mask.sum().item())
                    if std_mask_count > 0:
                        masked_positions = torch.nonzero(
                            loss_mask, as_tuple=False
                        )
                        masked_kl = std_mismatch_kl[loss_mask]
                        max_local_idx = int(masked_kl.argmax().item())
                        max_row = int(masked_positions[max_local_idx, 0].item())
                        max_pos = int(masked_positions[max_local_idx, 1].item())
                        first_row = int(masked_positions[0, 0].item())
                        first_pos = int(masked_positions[0, 1].item())
                        max_old_logp = float(
                            inference_logprobs[max_row, max_pos].detach().item()
                        )
                        max_new_logp = float(
                            out["logprobs"][max_row, max_pos].detach().item()
                        )
                        first_old_logp = float(
                            inference_logprobs[first_row, first_pos].detach().item()
                        )
                        first_new_logp = float(
                            out["logprobs"][first_row, first_pos].detach().item()
                        )
                        logger.info(
                            f"[KLSTD] mb={micro_step}/{len(micro_batches)} "
                            f"seq={int(input_ids.shape[1])} "
                            f"loss_tokens={std_mask_count} "
                            f"mismatch_kl={float(masked_kl.mean().item()):.6f} "
                            f"max_kl={float(masked_kl.max().item()):.6f}@{max_pos} "
                            f"max_tok={int(input_ids[max_row, max_pos].item())} "
                            f"max_old={max_old_logp:.6f} "
                            f"max_new={max_new_logp:.6f} "
                            f"max_delta={max_new_logp - max_old_logp:.6f} "
                            f"first_kl={float(masked_kl[0].item()):.6f}@{first_pos} "
                            f"first_tok={int(input_ids[first_row, first_pos].item())} "
                            f"first_old={first_old_logp:.6f} "
                            f"first_new={first_new_logp:.6f} "
                            f"first_delta={first_new_logp - first_old_logp:.6f}"
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
            if use_segmented or use_per_call:
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

            # Debug log with *local, micro step* stats. Entropy is now
            # computed for both the standard path (out["entropy"][loss_mask])
            # and the segmented path (per-segment compute_entropy over
            # owned target tokens, concatenated). Both paths append a
            # 1-D tensor per micro-batch, so indexing [-1] is always
            # safe as long as the list is non-empty. On the rare
            # degenerate segmented case (all segments had zero owned
            # tokens), the appended tensor is zero-length and .mean()
            # returns NaN — acceptable for a debug log.
            micro_step_loss = tensors["loss"][-1].mean().item()
            micro_step_message = (
                f"Micro Step {micro_step}/{len(micro_batches)} | "
                f"Loss: {micro_step_loss:.8f} ({micro_step_loss:.3e})"
            )
            if len(tensors["entropy"]) > 0 and tensors["entropy"][-1].numel() > 0:
                micro_step_message += (
                    f" | Entropy: {tensors['entropy'][-1].mean().item():.4f}"
                )
            if "mismatch_kl" in tensors and len(tensors["mismatch_kl"]) > 0:
                micro_step_message += f" | Mismatch KL: {tensors['mismatch_kl'][-1].mean().item():.4f}"
            if "max_vio" in tensors and len(tensors["max_vio"]) > 0:
                micro_step_message += f" | Max Vio: {tensors['max_vio'][-1].mean().item():.4f}"
            logger.debug(micro_step_message)

            if _KVE_MEM_TRACE:
                logger.info(
                    _mem_snap(
                        f"mb {micro_step}/{len(micro_batches)} "
                        f"{_compaction_flag} EXIT"
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
        num_local_tokens = seq_len * batch_size
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
        loss_mean = tensor_stats["loss/mean"]
        step_message = (
            f"Step {progress.step} | Time: {step_time:.2f}s | "
            f"Loss: {loss_mean:.8f} ({loss_mean:.3e}) | "
            f"Entropy: {tensor_stats['entropy/mean']:.4f}"
        )
        if "mismatch_kl/mean" in tensor_stats:
            step_message += f" | Mismatch KL: {tensor_stats['mismatch_kl/mean']:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm: {grad_norm:.4f}"
        step_message += f" | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f} GiB"
        if "max_vio/mean" in tensor_stats:
            step_message += f" | Max Vio: {tensor_stats['max_vio/mean']:.4f}"
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
    """Main entry-point for RL trainer. Run using `uv run trainer`"""
    set_proc_title("Trainer")
    train(cli(TrainerConfig))


if __name__ == "__main__":
    main()
