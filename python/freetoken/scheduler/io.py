from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Final, List, Sequence, Tuple

import torch
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
from freetoken.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


def broadcast_count(
    group, count: int | None, extra: Sequence[float] | None = None, extra_len: int = 0
) -> Tuple[int, List[float] | None]:
    """Rank 0 sends ``count`` (and ``extra_len`` floats of ``extra``), the other ranks pass
    ``None``; every rank gets both back. One broadcast: the extra values ride the message count
    the ranks exchange every iteration anyway."""
    if not extra_len:
        t = torch.tensor(-1 if count is None else count)
        group.broadcast(t, root=0).wait()
        return int(t.item()), None
    if count is None:
        t = torch.full((1 + extra_len,), -1.0, dtype=torch.float64)
    else:
        assert extra is not None and len(extra) == extra_len, "rank 0 must send every extra value"
        t = torch.tensor([float(count), *extra], dtype=torch.float64)
    group.broadcast(t, root=0).wait()
    return int(t[0].item()), t[1:].tolist()


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        # values rank 0 sends with every message poll (--ngram-spec planning inputs under TP):
        # tp_sync_source() on rank 0, tp_sync_len on all ranks, the result in tp_sync_values
        self.tp_sync_len = 0
        self.tp_sync_source: Callable[[], Sequence[float]] | None = None
        self.tp_sync_values: List[float] | None = None
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        n = self.tp_sync_len
        _, values = broadcast_count(
            self.tp_cpu_group, len(pending_raw_msgs), self.tp_sync_source() if n else None, n
        )
        if n:
            self.tp_sync_values = values

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        n = self.tp_sync_len
        dst_length, values = broadcast_count(self.tp_cpu_group, None, None, n)
        if n:
            self.tp_sync_values = values

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[BaseTokenizerMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[BaseTokenizerMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
