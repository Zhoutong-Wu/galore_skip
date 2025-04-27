#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import torch
import torch.distributed.launcher.api as api
from torch.distributed.run import main
from torch.distributed.elastic.agent.server.api import WorkerSpec, log, _TERMINAL_STATE_SYNC_ID, SignalException
from torch.distributed.elastic.multiprocessing.api import Std
from torch.distributed.elastic.rendezvous.api import RendezvousHandler, RendezvousParameters
from torch.distributed.elastic.rendezvous.api import rendezvous_handler_registry as handler_registry
from torch.distributed import Store, PrefixStore
from datetime import timedelta
from typing import List, Optional, Tuple

if int(os.environ.get('MEGATRON_USE_REDIS_STORE_BACKEND', '0')):
    USE_REDIS_STORE = True
    from megatron.store import MonitoredBarrier, connect_redis, RedisStore, start_redis_server
else:
    USE_REDIS_STORE = False


class RedisRendezvous(RendezvousHandler):
    def __init__(
        self,
        run_id: str,
    ):
        self.run_id = run_id
        self._store: Optional[Store] = None
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--node_rank", type=int)
        parser.add_argument("--nnodes", type=int)
        parser.add_argument("--nproc_per_node", type=int)
        self._args, _ = parser.parse_known_args()
        if self._args.node_rank == 0:
            start_redis_server()
            log.warning(f"[torchrun] redis server started in RedisRendezvous. nnodes ({self._args.nnodes})")

        self.redis_cli = connect_redis()
        self.world_size = self._args.nnodes
        self.rank = self._args.node_rank

    def get_backend(self) -> str:
        return "redis"

    def next_rendezvous(self) -> Tuple[Store, int, int]:
        log.warning("Creating RedisStore as the c10d::Store implementation")
        if not self._store:
            self._store = RedisStore(self.redis_cli, self.world_size)
        store = PrefixStore(self.run_id, self._store)
        return store, self.rank, self.world_size

    def is_closed(self):
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self):
        return 0

    def get_run_id(self) -> str:
        return self.run_id

    def shutdown(self) -> bool:
        return True


def create_redis_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    run_id = params.run_id
    return RedisRendezvous(run_id)


def create_redis_handler(params: RendezvousParameters) -> RendezvousHandler:
    return create_redis_rdzv_handler(params)


handler_registry.register("redis", create_redis_handler)


def get_all(store, prefix: str, size: int, timeout: int):
    r"""
    Given a store and a prefix, the method goes through the array of keys
    of the following format: ``{prefix}{idx}``, where idx is in a range
    from 0 to size, and tries to retrieve the data.

    Usage

    ::

     values = get_all(store, 'torchelastic/data', 3)
     value1 = values[0] # retrieves the data for key torchelastic/data0
     value2 = values[1] # retrieves the data for key torchelastic/data1
     value3 = values[2] # retrieves the data for key torchelastic/data2

    """
    start = time.time()
    data_arr = []
    for idx in range(size):
        elapsed = time.time() - start
        if elapsed > timeout:
            raise RuntimeError("Exit barrier Timeout")

        log.warn(f"Exit barrier waiting for agent: {idx}, elapsed: {elapsed} seconds")
        data = store.get(f"{prefix}{idx}")
        data_arr.append(data)
    return data_arr


def synchronize(
    store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    barrier_timeout: float = 300,
) -> List[bytes]:
    """
    Synchronizes ``world_size`` agents between each other using the underlying c10d store.
    The ``data`` will be available on each of the agents.

    Note: The data on the path is not deleted, as a result there can be stale data if
        you use the same key_prefix twice.
    """
    store.set_timeout(timedelta(seconds=barrier_timeout))
    store.set(f"{key_prefix}{rank}", data)
    agent_data = get_all(store, key_prefix, world_size, barrier_timeout)
    return agent_data


def barrier(
    store, rank: int, world_size: int, key_prefix: str, barrier_timeout: float = 300
) -> None:
    """
    A global lock between agents.

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """
    data = f"{rank}".encode(encoding="UTF-8")
    synchronize(store, data, rank, world_size, key_prefix, barrier_timeout)

def my_start_processes(
    name,
    entrypoint,
    args,
    envs,
    log_dir,
    start_method = "spawn",
    redirects = Std.NONE,
    tee = Std.NONE,
):
    old_entrypoint = entrypoint
    entrypoint = 'numactl'
    for local_rank in range(len(args)):
        numa_id = local_rank // 4
        args[local_rank] = (f'--cpunodebind={numa_id}', f'--membind={numa_id}') + \
                (old_entrypoint,) + args[local_rank]

    log.info(f'using numactl to control NUMA policy')
    for local_rank in range(len(args)):
        log.info(f'    local_rank={local_rank}: {" ".join(args[local_rank])}')

    return torch.distributed.elastic.agent.server.local_elastic_agent.old_start_processes(
        name,
        entrypoint,
        args,
        envs,
        log_dir,
        start_method,
        redirects,
        tee,
    )

class CustomLocalElasticAgent(api.LocalElasticAgent):

    def __init__(
        self,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, start_method, exit_barrier_timeout, log_dir)
        self._exit_barrier_timeout = int(os.environ.get("MARIANA_TORCHRUN_EXIT_BARRIER_TIMEOUT", "300"))
        if USE_REDIS_STORE:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--node_rank', type=int)
            parser.add_argument('--nnodes', type=int)
            parser.add_argument('--nproc_per_node', type=int)
            self._args, _ = parser.parse_known_args()
            if self._args.node_rank == 0:
                start_redis_server()
                log.warn(f'[torchrun] redis server started. nnodes ({self._args.nnodes})')

            gpu_per_node = int(os.getenv("ARNOLD_EXECUTOR_GPU", 8))
            os.environ["WORLD_SIZE"] = str(self._args.nnodes * gpu_per_node)  # fixme(liuxin.ai): patch for redis-store
            redis_cli = connect_redis()
            world_size = self._args.nnodes * self._args.nproc_per_node
            store = RedisStore(redis_cli, world_size)
            self.monitored_barrier = MonitoredBarrier(store, self._args.node_rank, list(range(self._args.nnodes)))

    def _exit_barrier(self):
        """
        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times. This barrier keeps the agent process alive until all workers finish.
        """
        log.info(
            f"Local worker group finished ({self._worker_group.state}). "
            f"Waiting {self._exit_barrier_timeout} seconds for other agents to finish"
        )
        start = time.time()
        try:
            if USE_REDIS_STORE:
                self.monitored_barrier.barrier(_TERMINAL_STATE_SYNC_ID, timeout=timedelta(seconds=self._exit_barrier_timeout))
            else:
                barrier(
                    self._store,
                    self._worker_group.group_rank,
                    self._worker_group.group_world_size,
                    key_prefix=_TERMINAL_STATE_SYNC_ID,
                    barrier_timeout=self._exit_barrier_timeout,
                )

            log.info(
                f"Done waiting for other agents. Elapsed: {time.time() - start} seconds"
            )
        except SignalException as e:
            log.warn(f"Got termination signal: {e.sigval}")
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            log.exception(
                f"Error waiting on exit barrier. Elapsed: {time.time() - start} seconds"
            )

    def _start_workers(self, worker_group):
        use_numactl = os.getenv('MARIANA_USE_NUMACTL', '0') in ['1']
        if use_numactl:
            # replace start_processes
            torch.distributed.elastic.agent.server.local_elastic_agent.old_start_processes = \
                    torch.distributed.elastic.agent.server.local_elastic_agent.start_processes
            torch.distributed.elastic.agent.server.local_elastic_agent.start_processes = \
                    my_start_processes
        ret_val =  super()._start_workers(worker_group)
        if use_numactl:
            # restore start_processes
            torch.distributed.elastic.agent.server.local_elastic_agent.start_processes = \
                    torch.distributed.elastic.agent.server.local_elastic_agent.old_start_processes
        return ret_val

api.oldLocalElasticAgent = api.LocalElasticAgent
api.LocalElasticAgent = CustomLocalElasticAgent


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
