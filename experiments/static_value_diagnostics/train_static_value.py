from __future__ import annotations

import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
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
from prime_rl.trainer.model import predict_value, setup_value_model
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import get_zero_gradient_ratio, setup_torch_distributed
from prime_rl.trainer.value import align_value_logits, compute_value_loss, predict_values, value_head_output_size
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a static-policy value function on collected rollouts.")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-range", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--n-bins", type=int, default=1)
    parser.add_argument("--attn", type=str, default="flash_attention_2")
    parser.add_argument("--impl", type=str, default="auto")
    parser.add_argument("--optimization-dtype", type=str, default="bfloat16")
    parser.add_argument("--reduce-dtype", type=str, default="bfloat16")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-ac", action="store_true")
    parser.add_argument("--disable-ac-offloading", action="store_true")
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
    parser.add_argument("--stream", action="store_true", help="Reload the rollout file while collection is still running.")
    parser.add_argument("--reload-records-interval", type=int, default=5)
    parser.add_argument("--min-train-records", type=int, default=None)
    parser.add_argument("--expected-val-records", type=int, default=0)
    parser.add_argument("--expected-test-records", type=int, default=0)
    parser.add_argument("--wait-timeout-seconds", type=int, default=7200)
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
            "optim": {"type": "adamw", "lr": args.lr, "weight_decay": 0.01, "max_norm": 1.0},
            "scheduler": {"type": "linear", "warmup_steps": args.warmup_steps, "decay_steps": 0, "min_lr": 0.0},
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


def forward_records(
    model: torch.nn.Module,
    records: list[RolloutRecord],
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clipped = [clipped_record_arrays(record, seq_len) for record in records]
    total_len = sum(len(ids) for ids, _, _ in clipped)
    if total_len == 0:
        raise ValueError("cannot forward an empty value microbatch")

    input_ids: list[int] = []
    position_ids: list[int] = []
    targets: list[float] = []
    mask: list[bool] = []

    for record, (ids, mask_list, _) in zip(records, clipped, strict=True):
        mask_idxs = [idx for idx in action_indices(mask_list, len(ids)) if idx > 0]
        if not mask_idxs:
            raise ValueError(f"record prompt_id={record.prompt_id} rollout_id={record.rollout_id} has no aligned actions")
        input_ids.extend(ids)
        position_ids.extend(range(len(ids)))
        targets.extend([float(record.reward)] * len(ids))
        mask.extend(idx in mask_idxs for idx in range(len(ids)))

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    position_tensor = torch.tensor(position_ids, dtype=torch.long, device=device).unsqueeze(0)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    return align_value_logits(predict_value(model, input_tensor, position_tensor)), target_tensor, mask_tensor


def forward_record(
    model: torch.nn.Module,
    record: RolloutRecord,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    logits, targets, mask = forward_records(model, [record], seq_len, device)
    mask_idxs = mask[0].nonzero(as_tuple=False).flatten().tolist()
    return logits, targets, mask, mask_idxs


def _cycle_batch_indices(records: list[RolloutRecord], batch_size: int, step: int, rng: random.Random) -> list[int]:
    if batch_size <= len(records):
        start = (step * batch_size) % len(records)
        if start + batch_size <= len(records):
            return list(range(start, start + batch_size))
        return list(range(start, len(records))) + list(range(0, batch_size - (len(records) - start)))
    return [rng.randrange(len(records)) for _ in range(batch_size)]


def iter_record_microbatches(
    records: list[RolloutRecord],
    *,
    seq_len: int,
    max_records: int,
) -> Iterable[list[RolloutRecord]]:
    chunk: list[RolloutRecord] = []
    chunk_tokens = 0
    for record in records:
        record_tokens = len(clipped_record_arrays(record, seq_len)[0])
        if chunk and (len(chunk) >= max_records or chunk_tokens + record_tokens > seq_len):
            yield chunk
            chunk = []
            chunk_tokens = 0
        chunk.append(record)
        chunk_tokens += record_tokens
    if chunk:
        yield chunk


def train(args: argparse.Namespace) -> None:
    world = get_world()
    logger = setup_logger(args.log_level)
    setup_torch_distributed(timeout=timedelta(seconds=args.dist_timeout_seconds), enable_gloo=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", world.local_rank)
    if args.global_batch_size % world.world_size != 0:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be divisible by world_size={world.world_size} "
            "so every FSDP rank executes the same number of forwards."
        )
    if args.micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {args.micro_batch_size}")

    mconfig = model_config(args)
    vconfig = value_config(args)
    resolve_ep(mconfig)
    parallel_dims = get_parallel_dims(mconfig, seq_len=args.seq_len)
    logger.info(f"Initializing static value model ({mconfig})")
    value_model = setup_value_model(
        mconfig,
        parallel_dims,
        loading_from_checkpoint_later=False,
        head_output_size=value_head_output_size(vconfig.loss),
    )
    optimizer = setup_optimizer(
        vconfig.optim,
        list(value_model.named_parameters()),
        parallel_dims,
        cpu_offload=mconfig.optim_cpu_offload,
    )
    scheduler = setup_scheduler(optimizer, vconfig.scheduler, args.steps, vconfig.optim.lr)

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
        all_records = load_rollout_records(args.rollouts)
        train_records = usable_records(all_records, "train", args.seq_len)
    rng = random.Random(args.seed)
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

    for step in range(args.steps):
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
        optimizer.zero_grad(set_to_none=True)
        batch_indices = _cycle_batch_indices(train_records, args.global_batch_size, step, rng)
        local_indices = batch_indices[world.rank :: world.world_size]
        local_tokens = sum(len([i for i in action_indices(train_records[idx].mask, args.seq_len) if i > 0]) for idx in local_indices)
        scale_tensor = torch.tensor(local_tokens, dtype=torch.float32, device=device)
        dist.all_reduce(scale_tensor, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp_cp").get_group())
        global_scale = max(int(scale_tensor.item()), 1)
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        metric_tokens = torch.zeros((), dtype=torch.float32, device=device)

        local_records = [train_records[idx] for idx in local_indices]
        for micro_records in iter_record_microbatches(
            local_records,
            seq_len=args.seq_len,
            max_records=args.micro_batch_size,
        ):
            logits, targets, mask = forward_records(
                value_model,
                micro_records,
                args.seq_len,
                device,
            )
            loss, metrics = compute_value_loss(logits, targets, mask, vconfig, scale=global_scale)
            loss.backward()
            if metrics["value/loss"].numel():
                loss_sum += metrics["value/loss"].sum().to(device)
                metric_tokens += metrics["value/loss"].numel()

        for param in value_model.parameters():
            if param.grad is not None:
                param.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)

        grad_norm = None
        if vconfig.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(
                value_model.parameters(),
                max_norm=vconfig.optim.max_norm,
                ep_enabled=parallel_dims.ep_enabled,
            )
            if grad_norm.device.type == "cpu":
                grad_norm = grad_norm.to(device)
        zero_grad_ratio = get_zero_gradient_ratio(value_model.parameters(), parallel_dims.dp_replicate)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp_cp").get_group())
        dist.all_reduce(metric_tokens, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp_cp").get_group())
        if world.is_master and (step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps):
            metrics = {
                "value_train/step": step + 1,
                "value_train/loss": float(loss_sum.item() / max(metric_tokens.item(), 1.0)),
                "value_train/lr": float(scheduler.get_last_lr()[0]),
                "value_train/tokens": int(scale_tensor.item()),
                "value_train/zero_grad_ratio": zero_grad_ratio,
            }
            if grad_norm is not None:
                metrics["value_train/grad_norm"] = float(grad_norm.item())
            logger.info(json.dumps(metrics, sort_keys=True))
            if wandb is not None:
                wandb.log(metrics, step=step + 1)

    ckpt_dir = args.output_dir / "value_checkpoint"
    state_dict = {"value_model": value_model.state_dict(), "scheduler": scheduler.state_dict()}
    dcp_save(state_dict, checkpoint_id=ckpt_dir)
    dist.barrier()

    expected_by_split = {"val": args.expected_val_records, "test": args.expected_test_records}
    for split in args.predict_splits:
        expected = expected_by_split.get(split, 0)
        if args.stream and expected > 0:
            all_records, _ = wait_for_usable_records(
                args.rollouts,
                split=split,
                seq_len=args.seq_len,
                min_count=expected,
                timeout_seconds=args.wait_timeout_seconds,
                logger=logger,
            )
        else:
            all_records = load_rollout_records(args.rollouts)
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
