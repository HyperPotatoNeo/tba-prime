"""RLConfig kv_mode: single-flag expansion into the 4 context-management modes."""

import pytest

from prime_rl.configs.rl import RLConfig


def _base_data(kv_mode: str | None) -> dict:
    data: dict = {
        "orchestrator": {
            "model": {"name": "Qwen/Qwen3-4B"},
            "markovian_thinker": {"max_turns": 40, "stride": 30},
        },
        "trainer": {"model": {"name": "Qwen/Qwen3-4B"}},
        "inference": {"model": {"name": "Qwen/Qwen3-4B"}},
    }
    if kv_mode is not None:
        data["kv_mode"] = kv_mode
    return data


def test_kv_mode_none_is_inert():
    config = RLConfig.model_validate(_base_data(None))
    assert config.kv_mode is None
    assert not config.orchestrator.markovian_thinker.enabled
    assert config.orchestrator.kv_mode is None
    assert config.inference.kv_mode is None


def test_kv_mode_markovian():
    config = RLConfig.model_validate(_base_data("markovian"))
    mt = config.orchestrator.markovian_thinker
    assert mt.enabled and not mt.kv_eviction
    assert not config.orchestrator.compaction_padding.enabled
    assert config.orchestrator.kv_mode is None
    assert config.inference.kv_mode is None


def test_kv_mode_kv_eviction():
    config = RLConfig.model_validate(_base_data("kv-eviction"))
    mt = config.orchestrator.markovian_thinker
    assert mt.enabled and mt.kv_eviction
    assert config.orchestrator.compaction_padding.enabled
    assert config.orchestrator.compaction_padding.phase4_enabled
    assert config.orchestrator.kv_mode is None
    assert config.inference.kv_mode is None
    assert config.inference.vllm_extra["compaction_max_turns"] == 40
    assert config.inference.vllm_extra["compaction_eviction_turn_stride"] == 30


@pytest.mark.parametrize("mode", ["kv-recall", "markovian-recall"])
def test_kv_mode_recall(mode):
    config = RLConfig.model_validate(_base_data(mode))
    mt = config.orchestrator.markovian_thinker
    assert mt.enabled and mt.kv_eviction
    assert config.orchestrator.compaction_padding.enabled
    assert config.orchestrator.kv_mode == mode
    assert config.inference.kv_mode == mode
    assert config.orchestrator.kv_recall_max_spans == 5
    assert config.inference.kv_recall_max_spans == 5


def test_kv_mode_invalid_rejected():
    with pytest.raises(ValueError):
        RLConfig.model_validate(_base_data("full-context"))


def test_padding_kwargs_for_mode_restore_modes():
    from kv_eviction.modes import padding_kwargs_for_mode

    kv = padding_kwargs_for_mode("kv-recall", max_turns=40, stride=30)
    assert kv["managed_context_restore_mode"] == "kv"
    assert kv["managed_context_turns_last_kept"] == 10
    mk = padding_kwargs_for_mode("markovian-recall", max_turns=40, stride=30)
    assert mk["managed_context_restore_mode"] == "visible_prefill"
    assert padding_kwargs_for_mode("kv-eviction", max_turns=40, stride=30) == {}


def test_engine_env_only_for_recall_modes():
    from kv_eviction.modes import engine_env_for_mode

    assert engine_env_for_mode("kv-eviction") == {}
    assert engine_env_for_mode("markovian") == {}
    env = engine_env_for_mode("kv-recall", recall_max_spans=3)
    assert env["KVE_SOFT_PIN"] == "1"
    assert env["KVE_MANAGED_CONTEXT_RECALL_MAX_SPANS"] == "3"
