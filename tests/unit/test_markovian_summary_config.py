# SPDX-License-Identifier: Apache-2.0
"""Validator tests for orchestrator.markovian_thinker.summary.

Covers the rules enforced by `validate_markovian_summary` on `RLConfig`:

- summary.enabled requires markovian_thinker.enabled
- compaction_max_turns > 0 required
- teacher_rollout_model rejected
- max_len_summary must fit under inference.model.max_model_len + 2048
- mode="markovian" requires vLLM-side compaction OFF
- mode="eviction" requires at least one vLLM-side compaction knob ON
- mode="eviction" relaxes the existing Markovian mutex vs. trainer.compaction
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import (
    MarkovianSummaryConfig,
    MarkovianThinkerConfig,
    OrchestratorConfig,
)
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import TrainerConfig


def _mt_config(**summary_kwargs):
    """Build a MarkovianThinkerConfig with enabled=true and a summary
    config with sensible defaults for the test cases."""
    base_summary_kwargs = {
        "enabled": True,
        "mode": "markovian",
        "compaction_max_turns": 4,
        "max_len_summary": 512,
    }
    base_summary_kwargs.update(summary_kwargs)
    return MarkovianThinkerConfig(
        enabled=True,
        max_turns=4,
        summary=MarkovianSummaryConfig(**base_summary_kwargs),
    )


def _orch(**overrides) -> OrchestratorConfig:
    """Orchestrator with use_token_client=False (Markovian requires)."""
    kw = dict(overrides)
    kw.setdefault("markovian_thinker", _mt_config())
    kw.setdefault("use_token_client", False)
    return OrchestratorConfig(**kw)


def _rl_config(**overrides) -> dict:
    return {
        "trainer": overrides.pop("trainer", TrainerConfig()),
        "orchestrator": overrides.pop("orchestrator", _orch()),
        **overrides,
    }


# ─── Happy paths ───


def test_summary_default_off_validates():
    """Default MarkovianThinkerConfig has summary.enabled=False: no-op."""
    RLConfig(trainer=TrainerConfig(), orchestrator=OrchestratorConfig())


def test_summary_markovian_mode_accepted():
    RLConfig(**_rl_config())


def test_summary_eviction_mode_accepted_with_vllm_window():
    inf = InferenceConfig()
    inf.vllm_extra = {"compaction_window_size": 4096, "compaction_stride": 512}
    trainer = TrainerConfig()
    trainer.model.impl = "hf"  # segmented_forward requires hf impl
    trainer.compaction.window_size = 4096
    trainer.compaction.stride = 512
    orch = _orch(
        markovian_thinker=_mt_config(mode="eviction", compaction_max_turns=4)
    )
    RLConfig(**_rl_config(inference=inf, trainer=trainer, orchestrator=orch))


def test_summary_eviction_mode_accepted_with_vllm_max_turns():
    inf = InferenceConfig()
    inf.vllm_extra = {"compaction_max_turns": 8}
    orch = _orch(
        markovian_thinker=_mt_config(mode="eviction", compaction_max_turns=4)
    )
    RLConfig(**_rl_config(inference=inf, orchestrator=orch))


# ─── Rejections: common constraints ───


def test_summary_requires_markovian_enabled():
    """summary.enabled=true with markovian_thinker.enabled=false must fail."""
    mt = MarkovianThinkerConfig(
        enabled=False,
        summary=MarkovianSummaryConfig(
            enabled=True, mode="markovian", compaction_max_turns=4
        ),
    )
    orch = OrchestratorConfig(markovian_thinker=mt, use_token_client=False)
    with pytest.raises(ValidationError, match="markovian_thinker.enabled=true"):
        RLConfig(**_rl_config(orchestrator=orch))


def test_summary_requires_compaction_max_turns_positive():
    """compaction_max_turns=0 is rejected (no trigger would ever fire)."""
    orch = _orch(
        markovian_thinker=_mt_config(mode="markovian", compaction_max_turns=0)
    )
    with pytest.raises(ValidationError, match="compaction_max_turns"):
        RLConfig(**_rl_config(orchestrator=orch))


def test_summary_max_len_out_of_pydantic_range():
    """max_len_summary<16 is rejected at Pydantic layer (ge=16)."""
    with pytest.raises(ValidationError):
        MarkovianSummaryConfig(
            enabled=True, mode="markovian", compaction_max_turns=4, max_len_summary=8
        )


# ─── Rejections: mode-specific ───


def test_markovian_mode_rejects_vllm_compaction_window():
    inf = InferenceConfig()
    inf.vllm_extra = {"compaction_window_size": 4096}
    with pytest.raises(ValidationError, match="compaction_window_size"):
        RLConfig(**_rl_config(inference=inf))


def test_markovian_mode_rejects_vllm_compaction_max_turns():
    inf = InferenceConfig()
    inf.vllm_extra = {"compaction_max_turns": 8}
    with pytest.raises(ValidationError, match="compaction_max_turns"):
        RLConfig(**_rl_config(inference=inf))


def test_eviction_mode_requires_some_vllm_compaction():
    """eviction mode with NO vLLM-side compaction configured must fail."""
    orch = _orch(
        markovian_thinker=_mt_config(mode="eviction", compaction_max_turns=4)
    )
    inf = InferenceConfig()
    inf.vllm_extra = {}
    with pytest.raises(ValidationError, match="compaction_window_size"):
        RLConfig(**_rl_config(inference=inf, orchestrator=orch))


def test_eviction_mode_relaxes_trainer_compaction_mutex():
    """The pre-existing Markovian mutex rejects trainer.compaction.window_size>0
    but eviction mode requires it to match vLLM-side block compaction.
    Ensure the relaxation works."""
    inf = InferenceConfig()
    inf.vllm_extra = {
        "compaction_window_size": 4096,
        "compaction_stride": 512,
        "block_size": 16,
    }
    trainer = TrainerConfig()
    trainer.model.impl = "hf"
    trainer.compaction.window_size = 4096
    trainer.compaction.stride = 512
    trainer.compaction.block_size = 16
    orch = _orch(
        markovian_thinker=_mt_config(mode="eviction", compaction_max_turns=4)
    )
    # Should not raise — the pre-existing Markovian mutex is relaxed.
    RLConfig(**_rl_config(inference=inf, trainer=trainer, orchestrator=orch))


def test_eviction_mode_still_rejects_use_token_client():
    """use_token_client stays forbidden in both modes."""
    inf = InferenceConfig()
    inf.vllm_extra = {"compaction_max_turns": 8}
    mt = _mt_config(mode="eviction", compaction_max_turns=4)
    orch = OrchestratorConfig(
        markovian_thinker=mt, use_token_client=True
    )
    with pytest.raises(ValidationError, match="use_token_client"):
        RLConfig(**_rl_config(inference=inf, orchestrator=orch))


# ─── Pydantic field-level constraint tests ───


def test_mode_literal_rejects_unknown_value():
    with pytest.raises(ValidationError):
        MarkovianSummaryConfig(
            enabled=True, mode="nope", compaction_max_turns=4  # type: ignore
        )


def test_on_error_literal_rejects_unknown_value():
    with pytest.raises(ValidationError):
        MarkovianSummaryConfig(
            enabled=True,
            mode="markovian",
            compaction_max_turns=4,
            on_error="panic",  # type: ignore
        )


def test_temperature_range():
    with pytest.raises(ValidationError):
        MarkovianSummaryConfig(
            enabled=True,
            mode="markovian",
            compaction_max_turns=4,
            temperature=2.5,
        )


def test_top_p_range():
    with pytest.raises(ValidationError):
        MarkovianSummaryConfig(
            enabled=True,
            mode="markovian",
            compaction_max_turns=4,
            top_p=0.0,
        )
