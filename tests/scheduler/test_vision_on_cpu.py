"""``--vision-on-cpu``: the vision tower lives in host RAM, on the primary rank only.

``Scheduler._encode_images`` must leave the GPU-tower path untouched, hand the soft tokens
from the primary rank to its siblings over the CPU group when the tower is host-resident,
and abort on *every* rank when the primary could not encode a request -- a sibling that
admitted it anyway would schedule a batch its siblings never run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.scheduler.mm import MMImage
from freetoken.scheduler.scheduler import Scheduler


class _Work:
    def wait(self) -> None:
        return None


class _Bus:
    """What a gloo group moves, reduced to: the root appends its bytes, a sibling copies
    the oldest entry in. One bus shared by both stub ranks = one process pair."""

    def __init__(self) -> None:
        self.queue: list[torch.Tensor] = []

    def group(self, *, primary: bool):
        bus = self

        class _Group:
            def broadcast(self, tensor, root=0):
                if primary:
                    bus.queue.append(tensor.clone())
                else:
                    tensor.copy_(bus.queue.pop(0))
                return _Work()

        return _Group()


def _imgs(*counts, tag=0):
    """Images of the given soft-token counts; the stub tower ignores their pixels."""
    return [MMImage(bytes([tag, i]) * 16, 0, c, pixel_values=torch.zeros(c, 1),
                    grid_thw=torch.tensor([[1, 1, c]])) for i, c in enumerate(counts)]


def _stub(*, primary: bool, tp_size: int, group, host_tower: bool, emb=None, boom=False):
    """A Scheduler shell (no __init__, no GPU) enough to drive _encode_images. The GPU
    "device" is CPU so the final hop is observable without hardware."""
    cpu = torch.device("cpu")

    def encode(*_args):
        sched.tower_calls += 1
        if boom:
            raise ValueError("corrupt image")
        return emb

    sched = Scheduler.__new__(Scheduler)
    sched.tower_calls = 0
    sched.engine = SimpleNamespace(
        encode_images=encode,
        device=cpu,
        dtype=torch.bfloat16,
        vision_device=cpu if host_tower else torch.device("cuda", 0),
    )
    sched.config = SimpleNamespace(tp_info=DistributedInfo(0 if primary else 1, tp_size))
    sched.tp_cpu_group = group
    return sched


def test_gpu_tower_never_touches_the_cpu_group():
    bus = _Bus()
    sched = _stub(primary=True, tp_size=2, group=bus.group(primary=True),
                  host_tower=False, emb=torch.zeros(2, 4))
    out = sched._encode_images(_imgs(2))
    assert out.shape == (2, 4)
    assert bus.queue == []


def test_host_tower_without_tp_runs_locally():
    bus = _Bus()
    emb = torch.randn(3, 8, dtype=torch.bfloat16)
    sched = _stub(primary=True, tp_size=1, group=bus.group(primary=True), host_tower=True, emb=emb)
    assert torch.equal(sched._encode_images(_imgs(3)), emb)
    assert bus.queue == []


def test_host_tower_ships_soft_tokens_to_the_sibling():
    bus = _Bus()
    emb = torch.randn(5, 6, dtype=torch.bfloat16)
    primary = _stub(primary=True, tp_size=2, group=bus.group(primary=True), host_tower=True, emb=emb)
    # the sibling owns no tower at all: encode_images would return None
    sibling = _stub(primary=False, tp_size=2, group=bus.group(primary=False), host_tower=True)
    primary._encode_images(_imgs(5))
    got = sibling._encode_images(_imgs(5))
    # bf16 -> fp32 over the group -> bf16 is exact: the ranks scatter identical rows
    assert torch.equal(got, emb)
    assert got.dtype == torch.bfloat16
    assert bus.queue == []  # both collectives consumed


def test_host_tower_round_trip_survives_a_second_request():
    bus = _Bus()
    for emb in (torch.randn(2, 3, dtype=torch.bfloat16), torch.randn(4, 3, dtype=torch.bfloat16)):
        primary = _stub(primary=True, tp_size=2, group=bus.group(primary=True),
                        host_tower=True, emb=emb)
        sibling = _stub(primary=False, tp_size=2, group=bus.group(primary=False), host_tower=True)
        primary._encode_images(_imgs(emb.shape[0]))
        assert torch.equal(sibling._encode_images(_imgs(emb.shape[0])), emb)
    assert bus.queue == []


def test_primary_failure_aborts_the_sibling_too():
    bus = _Bus()
    primary = _stub(primary=True, tp_size=2, group=bus.group(primary=True),
                    host_tower=True, boom=True)
    sibling = _stub(primary=False, tp_size=2, group=bus.group(primary=False), host_tower=True)
    with pytest.raises(ValueError, match="corrupt image"):
        primary._encode_images(_imgs(1))
    with pytest.raises(RuntimeError, match="primary rank"):
        sibling._encode_images(_imgs(1))
    assert bus.queue == []

