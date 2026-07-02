from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from prime_rl.configs.shared import EnvVars, WandbConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ClassificationValueLossConfig,
    LinearSchedulerConfig,
    ModelConfig,
    ValueFunctionConfig,
)
from prime_rl.utils.config import BaseConfig


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_static_value_model() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "name": "Qwen/Qwen3-4B-Instruct-2507",
            "seq_len": 8192,
            "attn": "flash_attention_2",
            "impl": "auto",
            "optimization_dtype": "bfloat16",
            "reduce_dtype": "bfloat16",
            "compile": None,
            "ac": {},
            "ac_offloading": None,
            "reshard_after_forward": False,
            "optim_cpu_offload": False,
            "dp_replicate": 4,
        }
    )


def default_static_value_function() -> ValueFunctionConfig:
    return ValueFunctionConfig(
        loss=ClassificationValueLossConfig(reward_range=(0.0, 1.0), n_bins=1),
        optim=AdamWConfig(lr=5e-5, weight_decay=0.01, max_norm=1.0),
        scheduler=LinearSchedulerConfig(warmup_steps=50, decay_steps=0, min_lr=0.0),
        gamma=1.0,
        gae_lambda=1.0,
        warmup_updates_per_batch=1,
    )


class RGMixStaticDataConfig(BaseConfig):
    dataset_path: Path = Path("/pscratch/sd/s/siddart2/datasets/rg_mix_7500")
    """Existing saved RGMix dataset path. Static diagnostics do not regenerate data."""

    train_episodes: int = Field(10_000, ge=1)
    """Static-policy train episodes to collect before value training."""

    eval_episodes: int = Field(1_024, ge=2)
    """Held-out episodes to collect after value training, split evenly into val/test."""

    group_size: int = Field(8, ge=2)
    """Rollouts per prompt group."""

    train_offset: int = Field(0, ge=0)
    """First prompt id for train collection."""

    eval_val_offset: int = Field(7000, ge=0)
    """First prompt id for held-out validation collection."""

    max_dataset_prompts: int = Field(7500, ge=1)
    """Dataset prompt count used to guard offset ranges."""

    seed: int = 42

    @property
    def train_groups(self) -> int:
        return self.train_episodes // self.group_size

    @property
    def eval_groups(self) -> int:
        return self.eval_episodes // self.group_size

    @property
    def val_groups(self) -> int:
        return self.eval_groups // 2

    @property
    def test_groups(self) -> int:
        return self.eval_groups - self.val_groups

    @property
    def eval_test_offset(self) -> int:
        return self.eval_val_offset + self.val_groups

    @model_validator(mode="after")
    def validate_episode_counts(self):
        if self.train_episodes % self.group_size != 0:
            raise ValueError("data.train_episodes must be divisible by data.group_size")
        if self.eval_episodes % self.group_size != 0:
            raise ValueError("data.eval_episodes must be divisible by data.group_size")
        if self.eval_groups % 2 != 0:
            raise ValueError("data.eval_episodes / data.group_size must be even for equal val/test groups")
        if self.train_offset + self.train_groups > self.eval_val_offset:
            raise ValueError("train prompt range overlaps held-out validation prompt range")
        if self.eval_test_offset + self.test_groups > self.max_dataset_prompts:
            raise ValueError("held-out prompt range exceeds data.max_dataset_prompts")
        return self


class StaticSamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_completion_tokens: int | None = None
    renderer_pool_size: int = Field(128, ge=1)
    max_concurrent_groups: int = Field(32, ge=1)
    max_retries: int = Field(3, ge=0)
    group_max_attempts: int = Field(4, ge=1)


class StaticInferenceConfig(BaseConfig):
    gpus_per_node: int = Field(4, ge=1)
    port: int = Field(8000, ge=1, le=65535)
    tp: int = Field(1, ge=1)
    dp: int = Field(4, ge=1)
    gpu_memory_utilization: float = Field(0.9, gt=0.0, le=1.0)
    ready_timeout_seconds: int = Field(1800, ge=1)

    @model_validator(mode="after")
    def validate_gpu_layout(self):
        if self.tp * self.dp != self.gpus_per_node:
            raise ValueError("inference.tp * inference.dp must equal inference.gpus_per_node")
        return self


class StaticValueTrainConfig(BaseConfig):
    steps: int = Field(100, ge=0)
    global_batch_size: int = Field(256, ge=1)
    log_every: int = Field(1, ge=1)
    dist_timeout_seconds: int = Field(1800, ge=1)
    mfu_model_params: float = Field(4.0e9, gt=0)
    mfu_peak_tflops_per_gpu: float = Field(312.0, gt=0)


class StaticDiagnosticsConfig(BaseConfig):
    group_sizes: list[int] = Field(default_factory=lambda: [2, 4, 8])
    rho_step: float = Field(0.05, gt=0, le=1)
    sensitivity_draws: int = Field(200, ge=1)
    seed: int = 0


class StaticValueDeploymentConfig(BaseConfig):
    num_nodes: int = Field(2, ge=2)
    train_node_index: int = Field(0, ge=0)

    @model_validator(mode="after")
    def validate_train_node(self):
        if self.num_nodes != 2:
            raise ValueError("deployment.num_nodes must be exactly 2 for staged static-value diagnostics")
        if self.train_node_index >= self.num_nodes:
            raise ValueError("deployment.train_node_index must be less than deployment.num_nodes")
        return self


class StaticValueConfig(BaseConfig):
    model: ModelConfig = Field(default_factory=default_static_value_model)

    value_function: ValueFunctionConfig = Field(default_factory=default_static_value_function)

    data: RGMixStaticDataConfig = RGMixStaticDataConfig()

    sampling: StaticSamplingConfig = StaticSamplingConfig()

    inference: StaticInferenceConfig = StaticInferenceConfig()

    train: StaticValueTrainConfig = StaticValueTrainConfig()

    diagnostics: StaticDiagnosticsConfig = StaticDiagnosticsConfig()

    deployment: StaticValueDeploymentConfig = StaticValueDeploymentConfig()

    env_vars: EnvVars = {}
    """Extra environment variables forwarded to remote collector/trainer processes."""

    wandb: WandbConfig | None = WandbConfig(project="prime-rl-static-value-diagnostics")

    output_dir: Path = Path("outputs/static_value")
    """Run directory containing data, logs, value checkpoints, and diagnostics."""

    repo_dir: Path = Path(".")
    """Repository root visible from every Slurm node."""

    stage: Literal["all", "collect_train", "train_value", "collect_eval", "predict", "diagnostics"] = "all"

    dry_run: bool = False

    @model_validator(mode="before")
    @classmethod
    def preserve_static_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if isinstance(data.get("model"), dict):
            defaults = default_static_value_model().model_dump(mode="python")
            data["model"] = _deep_merge(defaults, data["model"])
        if isinstance(data.get("value_function"), dict):
            defaults = default_static_value_function().model_dump(mode="python")
            data["value_function"] = _deep_merge(defaults, data["value_function"])
        return data

    @model_validator(mode="after")
    def validate_static_value_config(self):
        if not isinstance(self.value_function.loss, ClassificationValueLossConfig):
            raise ValueError("static value diagnostics currently require value_function.loss.type='classification'")
        if not isinstance(self.value_function.scheduler, LinearSchedulerConfig):
            raise ValueError("static value diagnostics currently require a linear value scheduler")
        if self.value_function.scheduler.decay_steps != 0:
            raise ValueError("static value diagnostics expect value_function.scheduler.decay_steps=0")
        if self.value_function.optim.max_norm is None:
            raise ValueError("static value diagnostics require value_function.optim.max_norm to be set")
        if not self.value_function.use_gae:
            raise ValueError("static value diagnostics always use GAE/return targets; value_function.use_gae must be true")
        if self.value_function.updates_per_step != 1:
            raise ValueError("static value diagnostics use value_function.warmup_updates_per_batch, not updates_per_step")
        if self.value_function.resolved_warmup_batches != 0:
            raise ValueError("value_function.resolved_warmup_batches is managed only by RL training")
        if self.model.lora is not None:
            raise ValueError("static value diagnostics do not support LoRA value functions")
        if self.model.vlm is not None:
            raise ValueError("static value diagnostics do not support VLM training")
        if self.model.compile is not None:
            raise ValueError("static value diagnostics currently require model.compile=null")
        if self.model.ac_offloading is not None:
            raise ValueError("static value diagnostics currently require model.ac_offloading=null")
        if self.model.fsdp_cpu_offload:
            raise ValueError("static value diagnostics do not support model.fsdp_cpu_offload")
        default_ac = default_static_value_model().ac
        if self.model.ac is not None and default_ac is not None:
            if self.model.ac.model_dump(mode="python") != default_ac.model_dump(mode="python"):
                raise ValueError("static value diagnostics only support default activation checkpointing or model.ac=null")
        bad_group_sizes = [size for size in self.diagnostics.group_sizes if size > self.data.group_size or size < 2]
        if bad_group_sizes:
            raise ValueError("diagnostics.group_sizes must be between 2 and data.group_size")
        return self
