from types import SimpleNamespace

from prime_rl.inference.vllm.cache_salt import apply_prime_rl_policy_cache_salt


def test_policy_cache_salt_is_added_once():
    request = SimpleNamespace(cache_salt=None, request_id="request-1")

    apply_prime_rl_policy_cache_salt(request, policy_version=8)
    apply_prime_rl_policy_cache_salt(request, policy_version=8)

    assert request.cache_salt == "prime-rl-policy-step:8"


def test_policy_cache_salt_preserves_caller_salt():
    request = SimpleNamespace(cache_salt="caller-salt", request_id="request-1")

    apply_prime_rl_policy_cache_salt(request, policy_version=8)

    assert request.cache_salt == "caller-salt|prime-rl-policy-step:8"
