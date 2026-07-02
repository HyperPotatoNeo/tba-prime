from __future__ import annotations

import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torchtitan.distributed.utils import clip_grad_norm_

import prime_rl._compat  # noqa: F401
from experiments.static_value_diagnostics.common import (
    RolloutRecord,
    action_indices,
    clipped_record_arrays,
    load_rollout_records,
    write_json,
)
from prime_rl.configs.trainer import ModelConfig, ValueFunctionConfig
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.ckpt import load_value_checkpoint, save_value_checkpoint
from prime_rl.trainer.model import predict_value, setup_value_model
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.rl.data import TensorMicroBatch, _torch_dtype
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import Tensors, build_bin_cost, get_zero_gradient_ratio, setup_torch_distributed
from prime_rl.trainer.value import (
    ValueTargets,
    ValueUpdateStats,
    align_value_logits,
    compute_gae,
    compute_value_loss,
    predict_values,
    value_head_output_size,
)
from prime_rl.trainer.world import get_world
from prime_rl.transport import MicroBatch, TrainingSample
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.cp import gather_for_cp, setup_cp_params, shard_for_cp
from prime_rl.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a static-policy value function on collected rollouts.")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--predict-rollouts", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--updates-per-batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-range", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--n-bins", type=int, default=1)
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--attn", type=str, default="flash_attention_2")
    parser.add_argument("--impl", type=str, default="auto")
    parser.add_argument("--optimization-dtype", type=str, default="bfloat16")
    parser.add_argument("--reduce-dtype", type=str, default="bfloat16")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-ac", action="store_true")
    parser.add_argument("--disable-ac-offloading", action="store_true")
    parser.add_argument("--disable-reshard-after-forward", action="store_true")
    parser.add_argument("--disable-optim-cpu-offload", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dp-replicate", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=str, default="auto")
    parser.add_argument("--dist-timeout-seconds", type=int, default=1800)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--predict-splits", type=str, nargs="+", default=["val", "test"])
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--load-value-checkpoint", type=Path, default=None)
    parser.add_argument("--stream", action="store_true", help="Reload the rollout file while collection is still running.")
    parser.add_argument("--reload-records-interval", type=int, default=5)
    parser.add_argument("--min-train-records", type=int, default=None)
    parser.add_argument("--expected-val-records", type=int, default=0)
    parser.add_argument("--expected-test-records", type=int, default=0)
    parser.add_argument("--wait-timeout-seconds", type=int, default=7200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--mfu-model-params", type=float, default=4.0e9)
    parser.add_argument("--mfu-peak-tflops-per-gpu", type=float, default=312.0)
    return parser.parse_args()


def model_config(args: argparse.Namespace) -> ModelConfig:
    ep: int | str = int(args.ep) if str(args.ep).isdigit() else args.ep
    return ModelConfig.model_validate(
        {
            "name": args.model,
            "seq_len": args.seq_len,
            "attn": args.attn,
            "impl": args.impl,
            "optimization_dtype": args.optimization_dtype,
            "reduce_dtype": args.reduce_dtype,
            "compile": None if args.disable_compile else {},
            "ac": None if args.disable_ac else {},
            "ac_offloading": None if args.disable_ac_offloading else {},
            "reshard_after_forward": not args.disable_reshard_after_forward,
            "optim_cpu_offload": not args.disable_optim_cpu_offload,
            "trust_remote_code": args.trust_remote_code,
            "dp_replicate": args.dp_replicate,
            "cp": args.cp,
            "ep": ep,
        }
    )


def value_config(args: argparse.Namespace) -> ValueFunctionConfig:
    return ValueFunctionConfig.model_validate(
        {
            "loss": {
                "type": "classification",
                "reward_range": tuple(float(x) for x in args.reward_range),
                "n_bins": args.n_bins,
            },
            "optim": {
                "type": "adamw",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_norm": args.max_norm,
            },
            "scheduler": {
                "type": "linear",
                "warmup_steps": args.warmup_steps,
                "decay_steps": 0,
                "min_lr": args.min_lr,
            },
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "warmup_updates_per_batch": args.updates_per_batch,
            "loss_weight": args.loss_weight,
        }
    )


def args_json(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def usable_records(
    records: list[RolloutRecord],
    split: str,
    seq_len: int,
    *,
    require: bool = True,
) -> list[RolloutRecord]:
    out = []
    for record in records:
        if record.split != split or not record.usable:
            continue
        _, mask, _ = clipped_record_arrays(record, seq_len)
        if any(idx > 0 for idx in action_indices(mask)):
            out.append(record)
    if require and not out:
        raise ValueError(f"no usable records for split={split!r}")
    return out


def validate_rollout_records(
    records: list[RolloutRecord],
    *,
    path: Path,
    vconfig: ValueFunctionConfig,
    group_size: int | None,
) -> None:
    seen: set[tuple[str, str, int]] = set()
    group_counts: dict[tuple[str, str], int] = {}
    reward_range: tuple[float, float] | None = None
    if hasattr(vconfig.loss, "reward_range"):
        low, high = vconfig.loss.reward_range
        reward_range = (float(low), float(high))

    for record in records:
        lengths = (len(record.token_ids), len(record.mask), len(record.logprobs))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{path}: rollout {record.split}:{record.prompt_id}:{record.rollout_id} has mismatched "
                f"token/mask/logprob lengths {lengths}"
            )
        if int(sum(record.mask)) != int(record.num_output_tokens):
            raise ValueError(
                f"{path}: rollout {record.split}:{record.prompt_id}:{record.rollout_id} has "
                f"num_output_tokens={record.num_output_tokens}, but mask sum={int(sum(record.mask))}"
            )
        if not np.isfinite(record.reward):
            raise ValueError(
                f"{path}: rollout {record.split}:{record.prompt_id}:{record.rollout_id} has non-finite reward"
            )
        if reward_range is not None:
            low, high = reward_range
            tol = 1e-5 * max(high - low, 1.0)
            if record.reward < low - tol or record.reward > high + tol:
                raise ValueError(
                    f"{path}: rollout {record.split}:{record.prompt_id}:{record.rollout_id} reward={record.reward} "
                    f"is outside reward_range={reward_range}"
                )
        key = (record.split, record.group_id, record.rollout_id)
        if key in seen:
            raise ValueError(f"{path}: duplicate rollout key {key}")
        seen.add(key)
        if not record.has_error:
            group_counts[(record.split, record.group_id)] = group_counts.get((record.split, record.group_id), 0) + 1

    if group_size is not None:
        bad_groups = [(key, count) for key, count in group_counts.items() if count != group_size]
        if bad_groups:
            key, count = bad_groups[0]
            raise ValueError(f"{path}: group {key} has {count} usable rollouts, expected {group_size}")


def load_validated_rollouts(path: Path, args: argparse.Namespace, vconfig: ValueFunctionConfig) -> list[RolloutRecord]:
    records = load_rollout_records(path)
    validate_rollout_records(records, path=path, vconfig=vconfig, group_size=args.group_size)
    return records


def wait_for_usable_records(
    path: Path,
    *,
    split: str,
    seq_len: int,
    min_count: int,
    timeout_seconds: int,
    logger,
) -> tuple[list[RolloutRecord], list[RolloutRecord]]:
    deadline = time.time() + timeout_seconds
    last_count = -1
    while True:
        all_records = load_rollout_records(path) if path.exists() else []
        split_records = usable_records(all_records, split, seq_len, require=False)
        if len(split_records) >= min_count:
            return all_records, split_records
        if len(split_records) != last_count:
            logger.info(f"Waiting for {split} records: {len(split_records)}/{min_count}")
            last_count = len(split_records)
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {min_count} usable {split} records in {path}")
        time.sleep(10)


def training_sample_from_record(record: RolloutRecord) -> TrainingSample:
    """Convert one static rollout into the same trainer wire type used by RL warmup."""
    token_ids, mask, logprobs = clipped_record_arrays(record, len(record.token_ids))
    action_idxs = action_indices(mask, len(token_ids))
    if not action_idxs:
        raise ValueError(f"record prompt_id={record.prompt_id} rollout_id={record.rollout_id} has no actions")

    rewards = [0.0] * len(token_ids)
    dones = [False] * len(token_ids)
    rewards[action_idxs[-1]] = float(record.reward)
    dones[action_idxs[-1]] = True

    return TrainingSample(
        token_ids=token_ids,
        mask=mask,
        logprobs=logprobs,
        temperatures=[1.0] * len(token_ids),
        env_name="rg_mix",
        rl_weights=[0.0] * len(token_ids),
        value_rewards=rewards,
        value_dones=dones,
    )


def _cycle_batch_indices(records: list[RolloutRecord], batch_size: int, step: int, rng: random.Random) -> list[int]:
    if batch_size <= len(records):
        start = (step * batch_size) % len(records)
        if start + batch_size <= len(records):
            return list(range(start, start + batch_size))
        return list(range(start, len(records))) + list(range(0, batch_size - (len(records) - start)))
    return [rng.randrange(len(records)) for _ in range(batch_size)]


def micro_batch_to_tensor(micro_batch: MicroBatch, max_runs: int = 1) -> TensorMicroBatch:
    """Local copy of the trainer DataLoader MicroBatch -> tensor conversion."""
    if micro_batch.lora_num_tokens is None:
        micro_batch.lora_num_tokens = [0] * max_runs
        micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
    mm_kwargs: dict[str, torch.Tensor] | None = None
    if micro_batch.mm_kwargs:
        mm_kwargs = {
            key: torch.frombuffer(bytearray(payload.data), dtype=_torch_dtype(payload.dtype)).reshape(payload.shape)
            for key, payload in micro_batch.mm_kwargs.items()
        }
    routed_experts = None
    packed_routed_experts = micro_batch.routed_experts
    if packed_routed_experts is not None:
        routed_experts = (
            torch.frombuffer(packed_routed_experts.data, dtype=_torch_dtype(packed_routed_experts.dtype))
            .reshape(packed_routed_experts.shape)
            .to(torch.int32)
            .unsqueeze(0)
        )
    return TensorMicroBatch(
        input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
        position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
        advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
        inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
        ref_logprobs=torch.tensor(micro_batch.ref_logprobs, dtype=torch.float).unsqueeze(0)
        if micro_batch.ref_logprobs is not None
        else None,
        loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
        temperatures=torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
        env_names=micro_batch.env_names,
        sequence_lengths=micro_batch.sequence_lengths,
        lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
        routed_experts=routed_experts,
        mm_kwargs=mm_kwargs,
        mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
        if micro_batch.mm_token_type_ids is not None
        else None,
        rl_weights=torch.tensor(micro_batch.rl_weights, dtype=torch.float).unsqueeze(0)
        if micro_batch.rl_weights is not None
        else None,
        ce_weights=torch.tensor(micro_batch.ce_weights, dtype=torch.float).unsqueeze(0)
        if micro_batch.ce_weights is not None
        else None,
        ref_kl_weights=torch.tensor(micro_batch.ref_kl_weights, dtype=torch.float).unsqueeze(0)
        if micro_batch.ref_kl_weights is not None
        else None,
        value_rewards=torch.tensor(micro_batch.value_rewards, dtype=torch.float).unsqueeze(0)
        if micro_batch.value_rewards is not None
        else None,
        value_dones=torch.tensor(micro_batch.value_dones, dtype=torch.bool).unsqueeze(0)
        if micro_batch.value_dones is not None
        else None,
        run_id=micro_batch.run_id,
        run_step=micro_batch.run_step,
        phase=micro_batch.phase,
        save_checkpoint=micro_batch.save_checkpoint,
    )


def prepare_value_micro_batches(
    records: list[RolloutRecord],
    *,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    bin_cost,
    pad_to_multiple_of: int,
) -> list[TensorMicroBatch]:
    samples = [training_sample_from_record(record) for record in records]
    micro_batch_grid = prepare_batch(
        rollouts=samples,
        seq_len=seq_len,
        num_train_workers=dp_world_size,
        idxs=[0] * len(samples),
        num_loras=1,
        bin_cost=bin_cost,
        pad_to_multiple_of=pad_to_multiple_of,
        phase="value_warmup",
    )
    return [micro_batch_to_tensor(micro_batch) for micro_batch in micro_batch_grid[dp_rank]]


def data_parallel_rank(global_rank: int, world_size: int, dp_world_size: int) -> int:
    non_dp_world_size = world_size // dp_world_size
    return global_rank // non_dp_world_size


def train(args: argparse.Namespace) -> None:
    world = get_world()
    logger = setup_logger(args.log_level)
    setup_torch_distributed(timeout=timedelta(seconds=args.dist_timeout_seconds), enable_gloo=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", world.local_rank)
    if args.global_batch_size < 1:
        raise ValueError(f"global_batch_size must be >= 1, got {args.global_batch_size}")
    if args.updates_per_batch < 1:
        raise ValueError(f"updates_per_batch must be >= 1, got {args.updates_per_batch}")
    if args.steps < 0:
        raise ValueError(f"steps must be non-negative, got {args.steps}")
    if args.log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {args.log_every}")
    if args.predict_only and args.load_value_checkpoint is None:
        raise ValueError("--predict-only requires --load-value-checkpoint")
    if args.stream and args.predict_rollouts is not None:
        raise ValueError("--stream is only supported when training and prediction use --rollouts")

    mconfig = model_config(args)
    vconfig = value_config(args)
    resolve_ep(mconfig)
    parallel_dims = get_parallel_dims(mconfig, seq_len=args.seq_len)
    dp_world_size = parallel_dims.get_mesh("dp").size()
    dp_rank = data_parallel_rank(world.rank, world.world_size, dp_world_size)
    logger.info(f"Initializing static value model ({mconfig})")
    value_model = setup_value_model(
        mconfig,
        parallel_dims,
        loading_from_checkpoint_later=args.load_value_checkpoint is not None,
        head_output_size=value_head_output_size(vconfig.loss),
    )
    optimizer = setup_optimizer(
        vconfig.optim,
        list(value_model.named_parameters()),
        parallel_dims,
        cpu_offload=mconfig.optim_cpu_offload,
    )
    scheduler = setup_scheduler(optimizer, vconfig.scheduler, max(args.steps, 1), vconfig.optim.lr)
    bin_cost = build_bin_cost(value_model.config)
    cp_enabled = parallel_dims.cp_enabled
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
    cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
    cp_size = parallel_dims.cp

    def prepare_value_inputs(micro_batch: TensorMicroBatch):
        if micro_batch["value_rewards"] is None or micro_batch["value_dones"] is None:
            raise ValueError("Value functions require value_rewards and value_dones in every training batch.")

        input_ids = micro_batch["input_ids"].to(device)
        position_ids = micro_batch["position_ids"].to(device)
        loss_mask = micro_batch["loss_mask"].to(device)
        rewards = micro_batch["value_rewards"].to(device)
        dones = micro_batch["value_dones"].to(device)
        routed_experts = micro_batch["routed_experts"].to(device) if micro_batch["routed_experts"] is not None else None
        mm_kwargs_raw = micro_batch.get("mm_kwargs")
        mm_kwargs = {k: v.to(device) for k, v in mm_kwargs_raw.items()} if mm_kwargs_raw else None
        mm_token_type_ids = (
            micro_batch["mm_token_type_ids"].to(device) if micro_batch.get("mm_token_type_ids") is not None else None
        )
        if cp_enabled and mm_kwargs is not None:
            raise NotImplementedError("Context parallelism is not supported with VLM/multimodal training")

        if cp_enabled:
            input_ids, forward_position_ids = setup_cp_params(
                input_ids, position_ids, cp_rank, cp_size, cp_group, cp_style=mconfig.cp_style
            )
            if routed_experts is not None:
                routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_size)
        else:
            forward_position_ids = position_ids

        return {
            "input_ids": input_ids,
            "position_ids": forward_position_ids,
            "routed_experts": routed_experts,
            "mm_kwargs": mm_kwargs,
            "mm_token_type_ids": mm_token_type_ids,
            "mask": loss_mask,
            "rewards": rewards,
            "dones": dones,
        }

    def forward_value_logits(value_inputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None]) -> torch.Tensor:
        with maybe_activation_offloading(mconfig.ac_offloading):
            value_logits = predict_value(
                value_model,
                value_inputs["input_ids"],
                value_inputs["position_ids"],
                mm_kwargs=value_inputs["mm_kwargs"],
                mm_token_type_ids=value_inputs["mm_token_type_ids"],
                routed_experts=value_inputs["routed_experts"],
            )
        if cp_enabled:
            value_logits = gather_for_cp(value_logits, cp_group)
        return align_value_logits(value_logits)

    def build_value_targets(micro_batches: list[TensorMicroBatch]) -> dict[int, ValueTargets]:
        targets = {}
        for micro_step, micro_batch in enumerate(micro_batches):
            value_inputs = prepare_value_inputs(micro_batch)
            with torch.no_grad():
                value_logits = forward_value_logits(value_inputs)
                values = predict_values(value_logits, vconfig.loss)
                advantages, returns = compute_gae(
                    rewards=value_inputs["rewards"],
                    dones=value_inputs["dones"],
                    values=values,
                    mask=value_inputs["mask"],
                    sequence_lengths=micro_batch["sequence_lengths"],
                    gamma=vconfig.gamma,
                    gae_lambda=vconfig.gae_lambda,
                )
            targets[micro_step] = ValueTargets(
                advantages=advantages.detach(),
                returns=returns.detach(),
                mask=value_inputs["mask"].detach(),
            )
        return targets

    def run_value_updates(
        micro_batches: list[TensorMicroBatch],
        value_targets: dict[int, ValueTargets],
        value_scale: int,
        tensors: Tensors,
    ) -> ValueUpdateStats:
        value_grad_norm: torch.Tensor | None = None
        value_zero_grad_ratio: float | None = None
        for _ in range(vconfig.warmup_updates_per_batch):
            optimizer.zero_grad()
            for micro_step, micro_batch in enumerate(micro_batches):
                value_inputs = prepare_value_inputs(micro_batch)
                value_logits = forward_value_logits(value_inputs)
                value_loss, value_tensors = compute_value_loss(
                    value_logits,
                    targets=value_targets[micro_step].returns,
                    mask=value_targets[micro_step].mask,
                    config=vconfig,
                    scale=value_scale,
                )
                value_loss.backward()
                tensors["value/scaled_loss"].append(value_loss.detach().to("cpu").unsqueeze(0))
                for key, value_tensor in value_tensors.items():
                    tensors[key].append(value_tensor.detach().to("cpu"))

            for param in value_model.parameters():
                if param.grad is not None:
                    param.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)

            if vconfig.optim.max_norm is not None:
                value_grad_norm = clip_grad_norm_(
                    value_model.parameters(),
                    max_norm=vconfig.optim.max_norm,
                    ep_enabled=parallel_dims.ep_enabled,
                )
                if value_grad_norm.device.type == "cpu":
                    value_grad_norm = value_grad_norm.to(device)

            value_zero_grad_ratio = get_zero_gradient_ratio(value_model.parameters(), parallel_dims.dp_replicate)
            optimizer.step()
            scheduler.step()

        optimizer.zero_grad()
        return ValueUpdateStats(grad_norm=value_grad_norm, zero_grad_ratio=value_zero_grad_ratio)

    if args.load_value_checkpoint is not None:
        if args.predict_only:
            load_value_checkpoint(args.load_value_checkpoint, value_model)
        else:
            load_value_checkpoint(args.load_value_checkpoint, value_model, [optimizer], scheduler)
        logger.info(f"Loaded value model checkpoint from {args.load_value_checkpoint}")

    all_records: list[RolloutRecord] = []
    train_records: list[RolloutRecord] = []
    rng = random.Random(args.seed)
    if not args.predict_only:
        min_train_records = args.min_train_records or args.global_batch_size
        if args.stream:
            all_records, train_records = wait_for_usable_records(
                args.rollouts,
                split="train",
                seq_len=args.seq_len,
                min_count=min_train_records,
                timeout_seconds=args.wait_timeout_seconds,
                logger=logger,
            )
        else:
            all_records = load_validated_rollouts(args.rollouts, args, vconfig)
            train_records = usable_records(all_records, "train", args.seq_len)
        rng.shuffle(train_records)
        logger.info(f"Loaded {len(train_records)} train records")

    wandb = None
    if world.is_master and args.wandb_project:
        import wandb as wandb_module

        wandb = wandb_module.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if world.is_master:
        write_json(
            args.output_dir / "train_config.json",
            {
                "args": args_json(args),
                "model_config": mconfig.model_dump(mode="json"),
                "value_config": vconfig.model_dump(mode="json"),
            },
        )

    for step in range(0 if args.predict_only else args.steps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        if args.stream and step > 0 and step % args.reload_records_interval == 0:
            _, refreshed_records = wait_for_usable_records(
                args.rollouts,
                split="train",
                seq_len=args.seq_len,
                min_count=min_train_records,
                timeout_seconds=args.wait_timeout_seconds,
                logger=logger,
            )
            if len(refreshed_records) != len(train_records):
                train_records = refreshed_records
                rng.shuffle(train_records)
                logger.info(f"Reloaded {len(train_records)} train records")
        value_model.train()
        batch_indices = _cycle_batch_indices(train_records, args.global_batch_size, step, rng)
        batch_records = [train_records[idx] for idx in batch_indices]
        micro_batches = prepare_value_micro_batches(
            batch_records,
            seq_len=args.seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            bin_cost=bin_cost,
            pad_to_multiple_of=mconfig.cp,
        )
        local_value_scale = sum(int(micro_batch["loss_mask"].sum()) for micro_batch in micro_batches)
        local_forward_tokens = sum(int(micro_batch["input_ids"].numel()) for micro_batch in micro_batches)
        global_counts = torch.tensor([local_value_scale, local_forward_tokens], dtype=torch.float32, device=device)
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp_cp").get_group())
        value_scale = max(int(global_counts[0].item()), 1)
        forward_tokens = float(global_counts[1].item())

        if world.is_master and step == 0:
            micro_batch_tokens = [int(micro_batch["input_ids"].numel()) for micro_batch in micro_batches]
            sequence_lengths = [
                length for micro_batch in micro_batches for length in micro_batch["sequence_lengths"] if length > 0
            ]
            logger.info(
                json.dumps(
                    {
                        "value_train/dp_world_size": dp_world_size,
                        "value_train/local_micro_batches": len(micro_batches),
                        "value_train/local_microbatch_max_tokens": max(micro_batch_tokens, default=0),
                        "value_train/local_microbatch_mean_tokens": sum(micro_batch_tokens)
                        / max(len(micro_batch_tokens), 1),
                        "value_train/local_max_sequence_length": max(sequence_lengths, default=0),
                        "value_train/mean_sequences_per_local_microbatch": len(sequence_lengths)
                        / max(len(micro_batches), 1),
                    },
                    sort_keys=True,
                )
            )

        tensors = Tensors()
        value_targets = build_value_targets(micro_batches)
        value_update_stats = run_value_updates(
            micro_batches,
            value_targets,
            value_scale,
            tensors,
        )
        tensor_stats = tensors.compute_stats()
        if world.is_master and (step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps):
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_start
            peak_flops = args.mfu_peak_tflops_per_gpu * 1e12 * world.world_size
            estimated_flops = (2.0 + 6.0 * vconfig.warmup_updates_per_batch) * args.mfu_model_params * forward_tokens
            metrics = {
                "value_train/step": step + 1,
                "value_train/loss": tensor_stats.get("value/loss/mean", float("nan")),
                "value_train/lr": float(scheduler.get_last_lr()[0]),
                "value_train/tokens": int(global_counts[0].item()),
                "value_train/forward_tokens": int(forward_tokens),
                "value_train/step_seconds": step_seconds,
                "value_train/forward_tokens_per_second": forward_tokens / max(step_seconds, 1e-6),
                "value_train/estimated_mfu": estimated_flops / max(step_seconds * peak_flops, 1e-6),
                "value_train/updates_per_batch": vconfig.warmup_updates_per_batch,
            }
            for key in ("value/accuracy/mean", "value/prediction/mean", "value/target/mean", "value/abs_error/mean"):
                if key in tensor_stats:
                    metrics[f"value_train/{key}"] = tensor_stats[key]
            if value_update_stats.zero_grad_ratio is not None:
                metrics["value_train/zero_grad_ratio"] = value_update_stats.zero_grad_ratio
            if value_update_stats.grad_norm is not None:
                metrics["value_train/grad_norm"] = float(value_update_stats.grad_norm.item())
            logger.info(json.dumps(metrics, sort_keys=True))
            if wandb is not None:
                wandb.log(metrics, step=step + 1)

    if not args.predict_only:
        ckpt_dir = args.output_dir / "value_checkpoint"
        save_value_checkpoint(ckpt_dir, value_model, [optimizer], scheduler)
        logger.info(f"Saved value checkpoint to {ckpt_dir}")
        dist.barrier()

    if not args.skip_predict:
        predict_rollouts = args.predict_rollouts or args.rollouts
        expected_by_split = {"val": args.expected_val_records, "test": args.expected_test_records}
        for split in args.predict_splits:
            expected = expected_by_split.get(split, 0)
            if args.stream and expected > 0:
                all_records, _ = wait_for_usable_records(
                    predict_rollouts,
                    split=split,
                    seq_len=args.seq_len,
                    min_count=expected,
                    timeout_seconds=args.wait_timeout_seconds,
                    logger=logger,
                )
            else:
                all_records = load_validated_rollouts(predict_rollouts, args, vconfig)
            predict_split(value_model, vconfig, all_records, split, args, device)
        dist.barrier()
    if wandb is not None:
        wandb.finish()
    dist.destroy_process_group()


def _classification_values_and_logits(logits: torch.Tensor, loss_config: Any) -> tuple[np.ndarray, np.ndarray]:
    values = predict_values(logits, loss_config).detach().float().cpu().numpy().reshape(-1)
    raw = logits.detach().float().cpu().numpy()
    if raw.shape[-1] == 2:
        odds_logits = raw[..., 1] - raw[..., 0]
    else:
        odds_logits = np.log(np.clip(values, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - values, 1e-8, 1.0))
    return values.astype(np.float32), odds_logits.reshape(-1).astype(np.float32)


@torch.inference_mode()
def predict_split(
    value_model: torch.nn.Module,
    vconfig: ValueFunctionConfig,
    all_records: list[RolloutRecord],
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    world = get_world()
    logger = setup_logger(args.log_level)
    records = usable_records(all_records, split, args.seq_len)
    records = records[world.rank :: world.world_size]
    value_model.eval()
    prompt_ids: list[int] = []
    rollout_ids: list[int] = []
    rewards: list[float] = []
    gen_lengths: list[int] = []
    initial_values: list[float] = []
    initial_logits: list[float] = []
    offsets = [0]
    flat_values: list[np.ndarray] = []
    flat_logits: list[np.ndarray] = []
    flat_positions: list[np.ndarray] = []

    for record in records:
        ids, mask_list, _ = clipped_record_arrays(record, args.seq_len)
        mask_idxs = [idx for idx in action_indices(mask_list, len(ids)) if idx > 0]
        if not mask_idxs:
            continue
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)
        raw_logits = predict_value(value_model, input_ids, position_ids)
        action_prefix = raw_logits[:, [idx - 1 for idx in mask_idxs], :]
        init_prefix = raw_logits[:, [mask_idxs[0] - 1], :]
        vals, odds_logits = _classification_values_and_logits(action_prefix, vconfig.loss)
        init_vals, init_odds = _classification_values_and_logits(init_prefix, vconfig.loss)
        prompt_ids.append(record.prompt_id)
        rollout_ids.append(record.rollout_id)
        rewards.append(record.reward)
        gen_lengths.append(len(mask_idxs))
        initial_values.append(float(init_vals[0]))
        initial_logits.append(float(init_odds[0]))
        flat_values.append(vals)
        flat_logits.append(odds_logits)
        flat_positions.append(np.arange(len(mask_idxs), dtype=np.int32))
        offsets.append(offsets[-1] + len(mask_idxs))

    path = args.output_dir / f"predictions_{split}_rank{world.rank:03d}.npz"
    np.savez_compressed(
        path,
        prompt_id=np.asarray(prompt_ids, dtype=np.int64),
        rollout_id=np.asarray(rollout_ids, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        values=np.concatenate(flat_values).astype(np.float32) if flat_values else np.empty(0, dtype=np.float32),
        logits=np.concatenate(flat_logits).astype(np.float32) if flat_logits else np.empty(0, dtype=np.float32),
        positions=np.concatenate(flat_positions).astype(np.int32) if flat_positions else np.empty(0, dtype=np.int32),
        gen_lengths=np.asarray(gen_lengths, dtype=np.int32),
        initial_value=np.asarray(initial_values, dtype=np.float32),
        initial_logit=np.asarray(initial_logits, dtype=np.float32),
    )
    logger.info(f"Wrote {len(prompt_ids)} {split} predictions to {path}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
