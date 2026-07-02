from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from renderers import AutoRendererConfig
from verifiers.types import ClientConfig
from verifiers.v1.legacy import rollout_output_to_trace

from experiments.static_value_diagnostics.common import (
    RolloutRecord,
    append_jsonl,
    rollout_to_dict,
    split_prompt_ids,
    write_json,
)
from experiments.static_value_diagnostics.rg_mix_env import load_environment
from prime_rl.orchestrator.trajectories import trace_to_samples


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(getattr(part, "text", "")) for part in content)
    return str(content or "")


def _last_assistant_text(trace: Any) -> str:
    for node in reversed(trace.nodes):
        message = node.message
        if node.sampled and getattr(message, "role", None) == "assistant":
            return _content_text(getattr(message, "content", ""))
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect static-policy RGMix rollouts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=str, default="/pscratch/sd/s/siddart2/datasets/rg_mix_7500")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-var", type=str, default="VLLM_API_KEY")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--num-train-prompts", type=int, default=1024)
    parser.add_argument("--num-val-prompts", type=int, default=128)
    parser.add_argument("--num-test-prompts", type=int, default=128)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=None)
    parser.add_argument("--test-offset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--renderer-pool-size", type=int, default=128)
    parser.add_argument("--max-concurrent-groups", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--group-max-attempts", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _client_config(args: argparse.Namespace) -> ClientConfig:
    os.environ.setdefault(args.api_key_var, "EMPTY")
    # Warm transformers' lazy import in the main thread before renderer workers
    # construct tokenizers concurrently.
    from transformers import AutoTokenizer as _AutoTokenizer  # noqa: F401

    return ClientConfig(
        client_type="renderer",
        api_base_url=args.base_url,
        api_key_var=args.api_key_var,
        renderer_model_name=args.model,
        renderer_config=AutoRendererConfig(),
        renderer_pool_size=args.renderer_pool_size,
    )


def _sampling_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "extra_body": {"top_k": args.top_k, "min_p": args.min_p},
        "logprobs": True,
        "max_completion_tokens": args.max_completion_tokens,
    }


def _records_from_output(
    out: dict[str, Any],
    *,
    env: Any,
    split: str,
    prompt_id: int,
    rollout_id: int,
) -> list[RolloutRecord]:
    trace = rollout_output_to_trace(out, prompt_id)
    reward = float(trace.rewards.get("reward", out.get("reward", 0.0)))
    if reward < 0.0 or reward > 1.0:
        raise ValueError(f"RGMix reward must be in [0,1], got {reward} for prompt_id={prompt_id}")
    if not trace.has_error:
        answer_idx = int(getattr(trace.task, "answer", out.get("answer")))
        rescored = float(env.score_completion(answer_idx, _last_assistant_text(trace)))
        if abs(reward - rescored) > 1e-6:
            raise ValueError(
                f"rubric reward mismatch for prompt_id={prompt_id}: rubric={reward} rescored={rescored}"
            )
    samples = trace_to_samples(trace, env_name="rg_mix")
    records: list[RolloutRecord] = []
    for branch_id, sample in enumerate(samples):
        mask = list(sample.mask)
        num_output_tokens = int(sum(mask))
        records.append(
            RolloutRecord(
                split=split,
                prompt_id=prompt_id,
                rollout_id=rollout_id + branch_id,
                group_id=f"{split}:{prompt_id}",
                reward=reward,
                token_ids=list(sample.token_ids),
                mask=mask,
                logprobs=list(sample.logprobs),
                num_output_tokens=num_output_tokens,
                stop_reason=trace.stop_condition,
                has_error=bool(trace.has_error),
                error=trace.errors[0].message if trace.errors else None,
            )
        )
    return records


async def collect(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rollouts.jsonl"
    error_path = output_dir / "collection_errors.jsonl"
    metadata_path = output_dir / "metadata.json"
    splits = split_prompt_ids(
        num_train_prompts=args.num_train_prompts,
        num_val_prompts=args.num_val_prompts,
        num_test_prompts=args.num_test_prompts,
        train_offset=args.train_offset,
        val_offset=args.val_offset,
        test_offset=args.test_offset,
    )
    max_prompt_id = max(max(ids, default=0) for ids in splits.values())
    env = load_environment(
        num_train_examples=max(7500, max_prompt_id + 1),
        num_eval_examples=100,
        seed=args.seed,
        dataset_path=args.dataset_path,
    )
    dataset = env.get_dataset()
    if max_prompt_id >= len(dataset):
        raise ValueError(f"requested prompt id {max_prompt_id}, but dataset only has {len(dataset)} rows")

    client = _client_config(args)
    sampling_args = _sampling_args(args)
    write_json(
        metadata_path,
        {
            "dataset_path": args.dataset_path,
            "model": args.model,
            "base_url": args.base_url,
            "group_size": args.group_size,
            "splits": {k: [min(v), max(v)] if v else [] for k, v in splits.items()},
            "sampling": sampling_args,
            "created_at": time.time(),
        },
    )

    seen_counts: dict[tuple[str, int], int] = {}
    if args.skip_existing and output_path.exists():
        from experiments.static_value_diagnostics.common import iter_jsonl

        for row in iter_jsonl(output_path):
            key = (row["split"], int(row["prompt_id"]))
            seen_counts[key] = seen_counts.get(key, 0) + int(not row.get("has_error", False))

    wandb = None
    if args.wandb_project:
        import wandb as wandb_module

        wandb = wandb_module.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.max_concurrent_groups)
    completed = 0

    async def collect_one(split: str, prompt_id: int) -> None:
        nonlocal completed
        if seen_counts.get((split, prompt_id), 0) >= args.group_size:
            return
        async with sem:
            task = dict(dataset[int(prompt_id)])
            task["example_id"] = prompt_id
            good_records: list[RolloutRecord] = []
            error_rows: list[dict[str, Any]] = []
            for attempt in range(args.group_max_attempts):
                needed = args.group_size - len(good_records)
                if needed <= 0:
                    break
                group_inputs = [task for _ in range(needed)]
                outs = await env.run_group(
                    group_inputs,
                    client,
                    args.model,
                    sampling_args,
                    max_retries=args.max_retries,
                    state_columns=["trajectory"],
                )
                for out in outs:
                    records = _records_from_output(
                        out,
                        env=env,
                        split=split,
                        prompt_id=prompt_id,
                        rollout_id=len(good_records),
                    )
                    for record in records:
                        if record.usable:
                            good_records.append(replace(record, rollout_id=len(good_records), has_error=False))
                        else:
                            error_rows.append({**rollout_to_dict(record), "attempt": attempt + 1})
                    if len(good_records) >= args.group_size:
                        break
            if len(good_records) < args.group_size:
                if error_rows:
                    async with lock:
                        append_jsonl(error_path, error_rows)
                raise RuntimeError(
                    f"prompt {split}:{prompt_id} produced {len(good_records)}/{args.group_size} usable rollouts "
                    f"after {args.group_max_attempts} attempts"
                )
            rows = [rollout_to_dict(record) for record in good_records[: args.group_size]]
            async with lock:
                append_jsonl(output_path, rows)
                if error_rows:
                    append_jsonl(error_path, error_rows)
                completed += 1
                if wandb is not None:
                    rewards = [row["reward"] for row in rows]
                    wandb.log(
                        {
                            "collect/completed_groups": completed,
                            "collect/reward_mean_last_group": sum(rewards) / len(rewards) if rewards else 0.0,
                            "collect/usable_rollouts_last_group": len(rewards),
                            "collect/error_attempts_last_group": len(error_rows),
                        }
                    )
                print(f"collected {split} prompt_id={prompt_id} rows={len(rows)}")

    tasks = [collect_one(split, prompt_id) for split, ids in splits.items() for prompt_id in ids]
    await asyncio.gather(*tasks)
    if wandb is not None:
        wandb.finish()


def main() -> None:
    asyncio.run(collect(parse_args()))


if __name__ == "__main__":
    main()
