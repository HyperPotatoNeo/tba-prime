import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import pytest

from prime_rl.utils.client import (
    _is_retryable_lora_error,
    load_lora_adapter,
    update_weights,
)


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_load_lora_adapter_retries_on_404_then_succeeds():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 404
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=error_response)
        return success_response

    mock_client.post = mock_post

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert call_count == 2


def test_load_lora_adapter_raises_non_retryable_error_immediately():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 400
    mock_client.post.side_effect = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=error_response)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert exc_info.value.response.status_code == 400
    assert mock_client.post.call_count == 1


def test_update_weights_preserve_kv_keeps_requests_and_skips_cache_reset():
    mock_client = AsyncMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    mock_client.post.return_value = response

    asyncio.run(update_weights([mock_client], None, step=8, preserve_kv=True))

    assert mock_client.post.await_args_list == [
        call("/pause", params={"mode": "keep", "clear_cache": "false"}),
        call(
            "/update_weights",
            json={
                "weight_dir": None,
                "step": 8,
                "reset_prefix_cache": False,
            },
        ),
        call("/resume"),
    ]


def test_update_weights_preserve_kv_rejects_lora_updates():
    with pytest.raises(ValueError, match="not supported for LoRA"):
        asyncio.run(
            update_weights(
                [AsyncMock()],
                Path("/test/path"),
                lora_name="test-lora",
                preserve_kv=True,
            )
        )


def test_update_weights_resumes_all_engines_after_partial_pause_failure():
    clients = [AsyncMock(), AsyncMock()]
    response = MagicMock()
    response.raise_for_status = MagicMock()

    async def first_post(*args, **kwargs):
        return response

    async def second_post(path, **kwargs):
        if path == "/pause":
            raise RuntimeError("pause failed")
        return response

    clients[0].post.side_effect = first_post
    clients[1].post.side_effect = second_post

    with pytest.raises(RuntimeError, match="Failed to pause all inference engines"):
        asyncio.run(update_weights(clients, None, preserve_kv=True))

    for client in clients:
        assert call("/resume") in client.post.await_args_list
