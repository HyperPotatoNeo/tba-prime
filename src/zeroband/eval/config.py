from pathlib import Path
from typing import Annotated, Literal

# Import environment before any other imports
# ruff: noqa
from zeroband.inference import envs

from pydantic import Field

from zeroband.inference.config import ModelConfig, EvalConfig, LogConfig, SamplingConfig, DataConfig
from zeroband.utils.config import MultiMonitorConfig
from zeroband.utils.pydantic_config import BaseConfig, BaseSettings


class ParallelConfig(BaseConfig):
    """Configures multi-node and multi-GPU setups through different types of parallelism (TP, DP, PP)."""

    # Note(Mika): Currently, we do not support DP and PP for evaluation.

    tp: Annotated[
        int | Literal["auto"],
        Field(
            default=1,
            description="Number of local GPUs to use for tensor parallelism. It is directly passed to vLLM. If 'auto', will be set to all available local GPUs.",
        ),
    ]


class Config(BaseSettings):
    """Configures evaluation."""

    # The model configuration
    model: Annotated[ModelConfig, Field(default=ModelConfig())]

    # The sampling configuration
    sampling: Annotated[SamplingConfig, Field(default=SamplingConfig())]

    # The parallel configuration
    parallel: Annotated[ParallelConfig, Field(default=ParallelConfig())]

    # The monitor configuration
    monitor: Annotated[MultiMonitorConfig, Field(default=MultiMonitorConfig())]

    # The data configuration
    data: Annotated[DataConfig, Field(default=DataConfig())]

    # The logging configuration
    log: Annotated[LogConfig, Field(default=LogConfig())]

    # The evaluation configuration
    eval: Annotated[EvalConfig, Field(default=EvalConfig())]
    
    ckpt_path: Annotated[Path, Field(default=Path("checkpoints"), description="Path to read new checkpoints from.")]

    use_tqdm: Annotated[bool, Field(default=False, description="Whether to use tqdm to display progress bars during model generation.")]

    seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Random seed used across evaluation components. If None, no seeding is used.",
        ),
    ]
