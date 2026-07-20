from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


@pytest.mark.parametrize("is_master", [False, True])
def test_every_trainer_rank_waits_for_inference_readiness(tmp_path: Path, is_master: bool) -> None:
    broadcaster = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcaster.logger = MagicMock()
    broadcaster.world = SimpleNamespace(is_master=is_master)
    broadcaster.multi_run_manager = SimpleNamespace(
        used_idxs=[0],
        ready_to_update={0: True},
        progress={0: SimpleNamespace(step=3)},
        get_run_dir=lambda _idx: tmp_path,
    )
    broadcaster.nccl_broadcast_sender = MagicMock()
    broadcaster._notify_orchestrator = MagicMock()
    broadcaster._wait_for_nccl_ready = MagicMock()

    broadcaster.broadcast_weights(MagicMock(), step=3)

    save_dir = get_step_path(get_broadcast_dir(tmp_path), 3)
    notified_runs = [(0, save_dir)]
    broadcaster._wait_for_nccl_ready.assert_called_once_with(notified_runs)
    broadcaster.nccl_broadcast_sender.broadcast_weights.assert_called_once()
    if is_master:
        broadcaster._notify_orchestrator.assert_called_once_with(notified_runs)
    else:
        broadcaster._notify_orchestrator.assert_not_called()


def test_master_notifies_orchestrator_and_clears_ready_flag(tmp_path: Path) -> None:
    broadcaster = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcaster.multi_run_manager = SimpleNamespace(ready_to_update={0: True})
    save_dir = tmp_path / "broadcast" / "step_3"

    broadcaster._notify_orchestrator([(0, save_dir)])

    assert (save_dir / "STABLE").is_file()
    assert broadcaster.multi_run_manager.ready_to_update[0] is False
