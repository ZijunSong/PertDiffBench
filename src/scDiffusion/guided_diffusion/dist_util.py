"""
Helpers for distributed training.

If mpi4py cannot load (no libmpi / OpenMPI), falls back to single-process
torch.distributed so cell_train / classifier_* work without installing MPI.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# mpi4py requires a system MPI (libmpi). Many single-GPU setups omit it.
try:
    from mpi4py import MPI  # type: ignore
except Exception:  # ImportError, OSError, RuntimeError (dlopen libmpi)
    MPI = None  # type: ignore

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def _using_mpi() -> bool:
    return MPI is not None


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if not _using_mpi():
        # Single process: no OpenMPI — use env:// with rank 0 only.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(_find_free_port())
        dist.init_process_group(backend=backend, init_method="env://")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    comm = MPI.COMM_WORLD
    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if not _using_mpi():
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)

    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
