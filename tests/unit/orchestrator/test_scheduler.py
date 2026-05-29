import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.orchestrator.scheduler import GroupState, InflightRequest, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_async_level = 1
    scheduler.strict_async_level = False
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(
        output_dir=Path("/tmp/prime-rl-test"),
        compaction_padding=SimpleNamespace(enabled=False, phase4_enabled=False),
    )
    scheduler.logger = MagicMock()
    scheduler.checkpoint_ready = asyncio.Event()
    scheduler.checkpoint_ready.set()
    scheduler.lora_name = None
    scheduler.model_name = "test-model"
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.enable_policy_updates = True
    scheduler._checked_initial_broadcast_state = False
    return scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRequest(off_policy_steps=1, client_config=client, env_name="test", group_id=1),
            survivor_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRequest(
                    off_policy_steps=0,
                    client_config=client,
                    env_name="test",
                    group_id=3,
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_restart_inflight_rollouts_for_weight_sync_requeues_active_phase4_tasks():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.compaction_padding.enabled = True
        scheduler.config.compaction_padding.phase4_enabled = True
        scheduler.weight_sync_restarted_rollouts_count = 0

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        active_task = asyncio.create_task(asyncio.sleep(60))
        done_task = asyncio.create_task(asyncio.sleep(0))
        await done_task

        scheduler.groups = {
            1: GroupState(
                example={"env_name": "test"},
                rollouts_to_schedule=0,
                completed_rollouts=[{"reward": 1.0}],
            ),
            2: GroupState(example={"env_name": "test"}, rollouts_to_schedule=0),
        }
        scheduler.inflight_requests = {
            active_task: InflightRequest(
                off_policy_steps=0,
                client_config=client,
                env_name="test",
                group_id=1,
                rollout_count=2,
            ),
            done_task: InflightRequest(
                off_policy_steps=0,
                client_config=client,
                env_name="test",
                group_id=2,
                rollout_count=1,
            ),
        }

        restarted = await scheduler.restart_inflight_rollouts_for_weight_sync(8)

        assert restarted == 2
        assert active_task not in scheduler.inflight_requests
        assert done_task in scheduler.inflight_requests
        assert scheduler.groups[1].rollouts_to_schedule == 2
        assert scheduler.groups[1].completed_rollouts == [{"reward": 1.0}]
        assert scheduler.cancelled_rollouts_count == 2
        assert scheduler.weight_sync_restarted_rollouts_count == 2
        assert active_task.cancelled()

    asyncio.run(run())


def test_maybe_update_policy_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(scheduler.maybe_update_policy())
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert scheduler.ckpt_step == 8

    asyncio.run(run())


def test_maybe_update_policy_refuses_stale_broadcast_when_starting_from_checkpoint():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.step = 50
        scheduler.ckpt_step = 50
        scheduler.inference_pool = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=51),
            pytest.raises(RuntimeError, match="stale broadcast weights"),
        ):
            await scheduler.maybe_update_policy()

        scheduler.inference_pool.update_weights.assert_not_called()

    asyncio.run(run())


def test_pause_policy_updates_cancels_loop_and_waits_for_active_sync():
    async def run() -> None:
        scheduler = make_scheduler()
        loop_cancelled = asyncio.Event()
        sync_started = asyncio.Event()
        sync_finished = asyncio.Event()

        async def update_loop() -> None:
            try:
                await asyncio.Future()
            finally:
                loop_cancelled.set()

        async def active_sync() -> None:
            sync_started.set()
            await asyncio.sleep(0.01)
            sync_finished.set()

        scheduler.update_policy_task = asyncio.create_task(update_loop())
        scheduler.inflight_policy_update_task = asyncio.create_task(active_sync())

        await sync_started.wait()
        await scheduler.pause_policy_updates()

        assert loop_cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert sync_finished.is_set()
        assert scheduler.inflight_policy_update_task.done()
        assert not scheduler.inflight_policy_update_task.cancelled()

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler.update_policy_task = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())
