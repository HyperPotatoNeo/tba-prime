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


def test_markovian_kv_eviction_expands_coupled_flags():
    """kv_eviction=true is the single high-level switch for the full
    trainer/orchestrator/vLLM eviction stack."""
    inference = InferenceConfig()
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(
            enabled=True,
            kv_eviction=True,
            max_turns=2,
            stride=1,
        ),
    )
    config = RLConfig(
        trainer=TrainerConfig(),
        orchestrator=orch,
        inference=inference,
    )

    assert config.orchestrator.use_token_client is False
    assert config.orchestrator.compaction_padding.enabled is True
    assert config.orchestrator.compaction_padding.phase4_enabled is True

    assert config.trainer.model.impl == "hf"
    assert config.trainer.model.attn == "flex_attention"
    assert config.trainer.compaction.window_size == 4096
    assert config.trainer.compaction.stride == 512
    assert config.trainer.compaction.block_size == 16
    assert config.trainer.compaction.protected_prefix_tokens == -1
    assert config.trainer.compaction.per_call_dispatch is True
    assert config.trainer.compaction.masked_forward_dispatch == "flex_attention"
    assert config.trainer.compaction.bptt_segments == -1

    assert config.inference is not None
    assert config.inference.enable_prefix_caching is True
    assert config.inference.vllm_extra["async_scheduling"] is False
    assert config.inference.vllm_extra["compaction_window_size"] == 4096
    assert config.inference.vllm_extra["compaction_stride"] == 512
    assert config.inference.vllm_extra["block_size"] == 16
    assert config.inference.vllm_extra["compaction_protected_prefix_tokens"] == -1
    assert config.inference.vllm_extra["compaction_max_turns"] == 2
    assert config.inference.vllm_extra["compaction_eviction_turn_stride"] == 1
    assert (
        config.inference.vllm_extra["compaction_assume_aligned_turn_boundaries"]
        is True
    )
    assert config.inference.vllm_extra["compaction_block_aligned_finish"] is True
    assert config.inference.vllm_extra["compaction_filler_token_id"] == 151643


def test_markovian_kv_eviction_eval_concurrency_defaults_and_opt_out():
    config = RLConfig(
        trainer=TrainerConfig(),
        orchestrator={
            "markovian_thinker": {
                "enabled": True,
                "kv_eviction": True,
                "max_turns": 2,
            },
            "eval": {},
        },
        inference=InferenceConfig(),
    )

    assert config.orchestrator.eval is not None
    assert config.orchestrator.eval.max_concurrent == 32
    assert config.orchestrator.eval.env[0].max_concurrent == 32

    uncapped = RLConfig(
        trainer=TrainerConfig(),
        orchestrator={
            "markovian_thinker": {
                "enabled": True,
                "kv_eviction": True,
                "max_turns": 2,
            },
            "eval": {"max_concurrent": None},
        },
        inference=InferenceConfig(),
    )

    assert uncapped.orchestrator.eval is not None
    assert uncapped.orchestrator.eval.max_concurrent is None
    assert uncapped.orchestrator.eval.env[0].max_concurrent is None


def test_markovian_kv_eviction_respects_explicit_flash_replay():
    inference = InferenceConfig()
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(
            enabled=True,
            kv_eviction=True,
            max_turns=2,
            stride=1,
        ),
    )
    config = RLConfig(
        trainer={"model": {"impl": "hf", "attn": "flash_attention_2"}},
        orchestrator=orch,
        inference=inference,
    )

    assert config.trainer.model.attn == "flash_attention_2"
    assert config.trainer.compaction.masked_forward_dispatch == "off"
    assert config.trainer.compaction.bptt_segments == 1


def test_markovian_kv_eviction_respects_window_override():
    inference = InferenceConfig()
    inference.vllm_extra = {"compaction_window_size": 8192}
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(
            enabled=True,
            kv_eviction=True,
            max_turns=2,
        ),
    )
    config = RLConfig(
        trainer=TrainerConfig(),
        orchestrator=orch,
        inference=inference,
    )

    assert config.trainer.compaction.window_size == 8192
    assert config.inference is not None
    assert config.inference.vllm_extra["compaction_window_size"] == 8192


def test_markovian_kv_eviction_rejects_explicit_tito():
    inference = InferenceConfig()
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(
            enabled=True,
            kv_eviction=True,
            max_turns=2,
        ),
        use_token_client=True,
    )
    with pytest.raises(ValidationError, match="use_token_client=false"):
        RLConfig(
            trainer=TrainerConfig(),
            orchestrator=orch,
            inference=inference,
        )


def test_markovian_kv_eviction_rejects_async_scheduling():
    inference = InferenceConfig()
    inference.vllm_extra = {"async_scheduling": True}
    orch = OrchestratorConfig(
        markovian_thinker=MarkovianThinkerConfig(
            enabled=True,
            kv_eviction=True,
            max_turns=2,
        ),
    )
    with pytest.raises(ValidationError, match="async_scheduling=false"):
        RLConfig(
            trainer=TrainerConfig(),
            orchestrator=orch,
            inference=inference,
        )


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
