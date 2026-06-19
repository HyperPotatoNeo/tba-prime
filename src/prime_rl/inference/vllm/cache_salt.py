import logging
from typing import Any

logger = logging.getLogger(__name__)


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
