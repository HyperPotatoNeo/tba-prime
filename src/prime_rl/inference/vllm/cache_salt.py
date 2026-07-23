import logging
import multiprocessing
from typing import Any

logger = logging.getLogger(__name__)

POLICY_VERSION_STATE_ATTR = "prime_rl_prefix_cache_policy_version"


def create_shared_policy_cache_version(initial_version: int = 0) -> Any:
    """Create policy-version state shared by vLLM's spawned API processes."""
    return multiprocessing.get_context("spawn").Value("q", int(initial_version))


def get_policy_cache_version(state: Any) -> int:
    version_state = getattr(state, POLICY_VERSION_STATE_ATTR, 0)
    if hasattr(version_state, "value"):
        return int(version_state.value)
    return int(version_state)


def set_policy_cache_version(state: Any, policy_version: int) -> None:
    version_state = getattr(state, POLICY_VERSION_STATE_ATTR, None)
    if hasattr(version_state, "value"):
        version_state.value = int(policy_version)
    else:
        logger.warning(
            "%s is not shared across API processes; falling back to process-local policy-version state",
            POLICY_VERSION_STATE_ATTR,
        )
        setattr(state, POLICY_VERSION_STATE_ATTR, int(policy_version))


def increment_policy_cache_version(state: Any) -> int:
    version_state = getattr(state, POLICY_VERSION_STATE_ATTR, None)
    if hasattr(version_state, "get_lock"):
        with version_state.get_lock():
            policy_version = int(version_state.value) + 1
            version_state.value = policy_version
            return policy_version

    policy_version = get_policy_cache_version(state) + 1
    set_policy_cache_version(state, policy_version)
    return policy_version


def apply_prime_rl_policy_cache_salt(
    request: Any,
    *,
    policy_version: int,
) -> None:
    """Salt vLLM prefix-cache keys by the active RL policy version."""
    version_salt = f"prime-rl-policy-step:{int(policy_version)}"
    if request.cache_salt:
        salt_parts = request.cache_salt.split("|")
        if version_salt in salt_parts:
            return
        request.cache_salt = request.cache_salt + "|" + version_salt
    else:
        request.cache_salt = version_salt
    logger.info(
        "Applied PrimeRL prefix-cache salt %s to request %s",
        version_salt,
        getattr(request, "request_id", None),
    )
