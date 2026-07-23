import multiprocessing
from types import SimpleNamespace
from typing import Any

from prime_rl.inference.vllm.cache_salt import (
    POLICY_VERSION_STATE_ATTR,
    apply_prime_rl_policy_cache_salt,
    create_shared_policy_cache_version,
    get_policy_cache_version,
    set_policy_cache_version,
)


def _set_shared_policy_version(version_state: Any, policy_version: int) -> None:
    state = SimpleNamespace(**{POLICY_VERSION_STATE_ATTR: version_state})
    set_policy_cache_version(state, policy_version)


def _read_shared_policy_version(version_state: Any, result_queue: Any) -> None:
    state = SimpleNamespace(**{POLICY_VERSION_STATE_ATTR: version_state})
    result_queue.put(get_policy_cache_version(state))


def test_policy_cache_salt_is_added_once():
    request = SimpleNamespace(cache_salt=None, request_id="request-1")

    apply_prime_rl_policy_cache_salt(request, policy_version=8)
    apply_prime_rl_policy_cache_salt(request, policy_version=8)

    assert request.cache_salt == "prime-rl-policy-step:8"


def test_policy_cache_salt_preserves_caller_salt():
    request = SimpleNamespace(cache_salt="caller-salt", request_id="request-1")

    apply_prime_rl_policy_cache_salt(request, policy_version=8)

    assert request.cache_salt == "caller-salt|prime-rl-policy-step:8"


def test_policy_cache_version_is_shared_across_spawned_processes():
    context = multiprocessing.get_context("spawn")
    version_state = create_shared_policy_cache_version()
    state = SimpleNamespace(**{POLICY_VERSION_STATE_ATTR: version_state})
    writer = context.Process(
        target=_set_shared_policy_version,
        args=(version_state, 8),
    )

    writer.start()
    writer.join(timeout=10)
    if writer.is_alive():
        writer.terminate()
        writer.join()

    assert writer.exitcode == 0
    assert get_policy_cache_version(state) == 8

    result_queue = context.Queue()
    reader = context.Process(
        target=_read_shared_policy_version,
        args=(version_state, result_queue),
    )
    reader.start()
    reader.join(timeout=10)
    if reader.is_alive():
        reader.terminate()
        reader.join()

    assert reader.exitcode == 0
    assert result_queue.get(timeout=1) == 8
    result_queue.close()
    result_queue.join_thread()
