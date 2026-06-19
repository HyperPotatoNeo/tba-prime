from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class."""
    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            cli(config_cls, args=["@", config_file.as_posix()])
            could_parse.append(True)
        except (ValidationError, ConfigFileError, SystemExit):
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class NestedConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.01
    name: str = "default"


class VariantA(BaseModel):
    type: Literal["a"] = "a"
    alpha: float = 0.1
    shared: int = 1


class VariantB(BaseModel):
    type: Literal["b"] = "b"
    beta: float = 0.2
    shared: int = 1


VariantType = Annotated[VariantA | VariantB, Field(discriminator="type")]


class DummyConfig(BaseConfig):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults():
    """All defaults are applied when no TOML or CLI args are given."""
    config = cli(DummyConfig, args=[])
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults():
    """CLI args override defaults."""
    config = cli(DummyConfig, args=["--name", "my-run", "--seed", "7"])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_defaults(tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"])
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01


def test_removed_fused_lm_head_chunk_size_field_is_rejected():
    with pytest.raises(ValidationError, match="fused_lm_head_chunk_size"):
        TrainerModelConfig.model_validate({"fused_lm_head_chunk_size": "auto"})


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def _am_trainer_config(**compaction_overrides):
    compaction = {
        "window_size": 1024,
        "strategy": "attention_matching",
        "stride": 16,
        "block_size": 16,
        "attention_matching_zerobeta": True,
    }
    compaction.update(compaction_overrides)
    return TrainerConfig.model_validate(
        {
            "model": {
                "impl": "hf",
                "attn": "flash_attention_2",
                "ac": None,
            },
            "compaction": compaction,
        }
    )


def test_am_full_bptt_requires_straight_through_gradient_mode():
    with pytest.raises(ValidationError, match="attention_matching_gradient_mode='detached'"):
        _am_trainer_config(bptt_segments="full")


def test_am_straight_through_accepts_full_bptt_aliases():
    cfg = _am_trainer_config(
        attention_matching_gradient_mode="straight_through",
        bptt_segments="full",
    )
    assert cfg.compaction.bptt_segments is None

    cfg_zero = _am_trainer_config(
        attention_matching_gradient_mode="straight_through",
        bptt_segments=0,
    )
    assert cfg_zero.compaction.bptt_segments is None


def _am_rl_config(*, logprobs_mode: str | None):
    vllm_extra = {
        "block_size": 16,
        "attention_backend": "FLASH_ATTN",
        "compaction_strategy": "attention_matching",
        "compaction_window_size": 512,
        "compaction_stride": 16,
        "attention_matching_zerobeta": True,
    }
    if logprobs_mode is not None:
        vllm_extra["logprobs_mode"] = logprobs_mode
    return {
        "max_steps": 1,
        "seq_len": 4096,
        "model": {"name": "Qwen/Qwen3-4B-Instruct-2507"},
        "trainer": {
            "model": {
                "name": "Qwen/Qwen3-4B-Instruct-2507",
                "impl": "hf",
                "attn": "flash_attention_2",
            },
            "compaction": {
                "window_size": 512,
                "strategy": "attention_matching",
                "stride": 16,
                "block_size": 16,
                "attention_matching_zerobeta": True,
            },
        },
        "orchestrator": {
            "batch_size": 2,
            "rollouts_per_example": 2,
            "client": {"base_url": ["http://localhost:8000/v1"]},
            "train": {"env": [{"id": "dummy-env"}]},
        },
        "inference": {
            "model": {
                "name": "Qwen/Qwen3-4B-Instruct-2507",
                "max_model_len": 4096,
            },
            "vllm_extra": vllm_extra,
        },
    }


def test_attention_matching_rl_requires_raw_vllm_logprobs():
    with pytest.raises(ValidationError, match="logprobs_mode"):
        RLConfig.model_validate(_am_rl_config(logprobs_mode=None))

    with pytest.raises(ValidationError, match="logprobs_mode"):
        RLConfig.model_validate(_am_rl_config(logprobs_mode="processed_logprobs"))

    cfg = RLConfig.model_validate(_am_rl_config(logprobs_mode="raw_logprobs"))
    assert cfg.inference is not None
    assert cfg.inference.vllm_extra["logprobs_mode"] == "raw_logprobs"
