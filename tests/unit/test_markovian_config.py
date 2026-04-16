# SPDX-License-Identifier: Apache-2.0
"""Validator tests for orchestrator.markovian_thinker.

Each rejection case exercises one of the four mutex constraints
enforced by `validate_markovian_thinker` on `RLConfig`:

- vLLM-side compaction (compaction_window_size / compaction_max_turns)
- trainer-side compaction (compaction.window_size)
- block-aligned padding (compaction_padding.enabled)
- token-in-token-out client (use_token_client)
"""

import pytest
from pydantic import ValidationError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import (
    MarkovianThinkerConfig,
    OrchestratorConfig,
)
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig


def _base_rl_config(**overrides) -> dict:
    """Build a minimal RLConfig kwargs dict with Markovian enabled.

    `use_token_client` defaults to True for most orchestrator configs
    but must be False for Markovian; we set it here unless the caller
    wants to test the TITO-mutex rejection.
    """
    orch = overrides.pop("orchestrator", None) or OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(enabled=True, max_turns=4),
        use_token_client=False,
    )
    trainer = overrides.pop("trainer", None) or TrainerConfig()
    return {"trainer": trainer, "orchestrator": orch, **overrides}


def test_markovian_off_default_config_validates():
    """Happy path: default config (markovian disabled) validates fine."""
    RLConfig(trainer=TrainerConfig(), orchestrator=OrchestratorConfig())


def test_markovian_on_valid_config_accepted():
    """All constraints satisfied -> config loads."""
    RLConfig(**_base_rl_config())


def test_markovian_rejects_vllm_compaction_window_size():
    inference = InferenceConfig()
    inference.vllm_extra = {"compaction_window_size": 1024}
    with pytest.raises(ValidationError, match="compaction_window_size"):
        RLConfig(**_base_rl_config(inference=inference))


def test_markovian_rejects_vllm_compaction_max_turns():
    inference = InferenceConfig()
    inference.vllm_extra = {"compaction_max_turns": 8}
    with pytest.raises(ValidationError, match="compaction_max_turns"):
        RLConfig(**_base_rl_config(inference=inference))


def test_markovian_rejects_trainer_compaction():
    trainer = TrainerConfig()
    trainer.compaction.window_size = 512
    with pytest.raises(ValidationError, match="trainer.compaction.window_size"):
        RLConfig(**_base_rl_config(trainer=trainer))


def test_markovian_rejects_compaction_padding():
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(enabled=True),
    )
    orch.compaction_padding.enabled = True
    with pytest.raises(ValidationError, match="compaction_padding"):
        RLConfig(**_base_rl_config(orchestrator=orch))


def test_markovian_rejects_use_token_client():
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(enabled=True),
        use_token_client=True,
    )
    with pytest.raises(ValidationError, match="use_token_client"):
        RLConfig(**_base_rl_config(orchestrator=orch))


def test_markovian_max_turns_minimum_one():
    """max_turns=0 is rejected by the Pydantic ge=1 constraint."""
    with pytest.raises(ValidationError):
        MarkovianThinkerConfig(enabled=True, max_turns=0)
