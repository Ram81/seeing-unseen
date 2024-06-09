import os
import threading
from typing import Tuple

import ifcfg
import torch
from torch import distributed as distrib

EXIT = threading.Event()
EXIT.clear()
REQUEUE = threading.Event()
REQUEUE.clear()
SAVE_STATE = threading.Event()
SAVE_STATE.clear()

# Default port to initialized the TCP store on
DEFAULT_PORT = 8738
DEFAULT_PORT_RANGE = 127
# Default address of world rank 0
DEFAULT_MAIN_ADDR = "127.0.0.1"

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
RESUME_STATE_BASE_NAME = ".habitat-resume-state"


def get_ifname() -> str:
    return ifcfg.default_interface()["device"]


def get_distrib_size() -> Tuple[int, int, int]:
    # Check to see if we should parse from torch.distributed.launch
    if os.environ.get("LOCAL_RANK", None) is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    # Else parse from SLURM is using SLURM
    elif os.environ.get("SLURM_JOBID", None) is not None:
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    # Otherwise setup for just 1 process, this is nice for testing
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1

    return local_rank, world_rank, world_size


def get_main_addr() -> str:
    return os.environ.get("MAIN_ADDR", DEFAULT_MAIN_ADDR)


def rank0_only():
    return (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)


def init_distrib_slurm(
    backend: str = "nccl",
) -> Tuple[int, torch.distributed.TCPStore]:  # type: ignore
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when ``srun`` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    local_rank, world_rank, world_size = get_distrib_size()

    main_port = int(os.environ.get("MAIN_PORT", DEFAULT_PORT))
    if SLURM_JOBID is not None:
        main_port += int(SLURM_JOBID) % int(
            os.environ.get("MAIN_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    main_addr = get_main_addr()

    tcp_store = distrib.TCPStore(  # type: ignore
        main_addr, main_port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store
