"""
Data-parallel MPI support for pacemaker / tensorpotential.

Design: "replicated optimizer". Every MPI rank runs the identical scipy optimizer
loop. Each rank owns a shard of the (precomputed) batch list and computes its
partial (loss, jacobian); an MPI Allreduce(SUM) makes every rank see the
full-dataset value, so all ranks take identical optimizer steps. Only file/log
I/O is guarded to the root rank.

This module is a NO-OP unless the environment variable PACEMAKER_MPI=1 (or "true"/
"yes"/"on"). When disabled:
  * mpi4py is never imported (so importing this module is free and MPI_Init is
    never called),
  * world_size() == 1, rank() == 0, is_root() == True,
  * shard(lst) returns lst unchanged,
  * every reduction is the identity.
So the single-GPU code path executes exactly the stock code.

Optional env var PACEMAKER_MPI_REPRODUCIBLE=1 switches SUM reductions from
MPI Allreduce (reduction order set by the MPI library) to a gather -> sum in
fixed rank order on root -> broadcast, so the reduction no longer depends on the
MPI library's algorithm. (Results still depend on the rank count, since each
rank's partial sum does, and on TF's non-deterministic GPU reductions.)

GPU pinning is NOT done here -- the launch script sets
CUDA_VISIBLE_DEVICES=$SLURM_LOCALID so each rank sees exactly one GPU and
init_gpu_config(gpu_ind=0) binds it.
"""

import os
import time

# ------------------------------------------------------------------ config gate

_TRUE = ("1", "true", "yes", "on")


def is_enabled():
    return os.environ.get("PACEMAKER_MPI", "0").strip().lower() in _TRUE


def is_reproducible():
    return os.environ.get("PACEMAKER_MPI_REPRODUCIBLE", "0").strip().lower() in _TRUE


def shard_mode():
    """How the per-iteration work is split across ranks:
      'batchlist' (default) -- each rank owns a contiguous slice of the batch LIST.
                               Per-GPU memory unchanged (one full batch still
                               lands on one GPU).
      'inbatch'             -- every rank sees every batch, but only its contiguous
                               slice of that batch's STRUCTURES. Per-GPU memory
                               ~1/world_size -> lets `batch_size` go up to ~Wx.
    inbatch makes W times more, smaller calls per rank; it keeps up with batchlist
    only when each GPU still gets ~100 structures per call.
    """
    m = os.environ.get("PACEMAKER_MPI_SHARD", "batchlist").strip().lower()
    return m if m in ("batchlist", "inbatch") else "batchlist"


# ------------------------------------------------------------ lazy communicator

_comm_cache = None
_warned = False


def _comm():
    """Return MPI.COMM_WORLD, importing mpi4py lazily. None if MPI disabled."""
    global _comm_cache, _warned
    if not is_enabled():
        return None
    if _comm_cache is None:
        try:
            from mpi4py import MPI
        except Exception as e:  # pragma: no cover
            if not _warned:
                print("[pacemaker.mpi] PACEMAKER_MPI=1 but mpi4py import failed "
                      "({}); running single-process.".format(e))
                _warned = True
            return None
        _comm_cache = MPI.COMM_WORLD
    return _comm_cache


def _mpi():
    from mpi4py import MPI
    return MPI


# ------------------------------------------------------------------ rank helpers

def rank():
    c = _comm()
    return c.Get_rank() if c is not None else 0


def world_size():
    c = _comm()
    return c.Get_size() if c is not None else 1


def local_rank():
    """Node-local rank. Prefer SLURM_LOCALID; fall back to a shared-memory split."""
    v = os.environ.get("SLURM_LOCALID")
    if v is not None:
        return int(v)
    v = os.environ.get("MPI_LOCALRANKID") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    if v is not None:
        return int(v)
    c = _comm()
    if c is None:
        return 0
    MPI = _mpi()
    node = c.Split_type(MPI.COMM_TYPE_SHARED, key=c.Get_rank())
    lr = node.Get_rank()
    node.Free()
    return lr


def is_root():
    return rank() == 0


def is_distributed():
    return world_size() > 1


# ------------------------------------------------------------------ collectives

def barrier():
    c = _comm()
    if c is not None:
        c.Barrier()


def bcast(obj, root=0):
    c = _comm()
    if c is None:
        return obj
    return c.bcast(obj, root=root)


# instrumentation: total seconds spent inside reductions (rank-local)
COMM_SECONDS = 0.0


def _tic():
    return time.perf_counter()


def _toc(t0):
    global COMM_SECONDS
    COMM_SECONDS += time.perf_counter() - t0


def allreduce_scalar_sum(x):
    """Sum a python/numpy scalar across ranks. Identity if MPI disabled."""
    c = _comm()
    if c is None:
        return x
    t0 = _tic()
    if is_reproducible():
        parts = c.gather(float(x), root=0)
        s = math_fsum(parts) if c.Get_rank() == 0 else None
        out = c.bcast(s, root=0)
    else:
        out = c.allreduce(float(x), op=_mpi().SUM)
    _toc(t0)
    return out


def allreduce_max(x):
    c = _comm()
    if c is None:
        return x
    t0 = _tic()
    out = c.allreduce(x, op=_mpi().MAX)
    _toc(t0)
    return out


def allreduce_array_sum(arr):
    """
    Sum a numpy array elementwise across ranks, returning a new array.
    Identity (returns arr) if MPI disabled.
    """
    c = _comm()
    if c is None:
        return arr
    import numpy as np
    a = np.ascontiguousarray(arr, dtype=np.float64)
    t0 = _tic()
    if is_reproducible():
        parts = c.gather(a, root=0)
        if c.Get_rank() == 0:
            s = np.zeros_like(a)
            for p in parts:            # fixed rank order
                s += p
        else:
            s = None
        out = c.bcast(s, root=0)
    else:
        out = np.empty_like(a)
        c.Allreduce(a, out, op=_mpi().SUM)
    _toc(t0)
    return out


def gather_concat(arr, root=0):
    """Gather variable-length numpy arrays to `root` and vstack them (root only).
    Non-root ranks get None. Identity if MPI disabled."""
    c = _comm()
    if c is None:
        return arr
    import numpy as np
    parts = c.gather(np.asarray(arr), root=root)
    if c.Get_rank() == root:
        return np.vstack([p for p in parts if len(p)])
    return None


def allgather_vstack(arr):
    """Allgather variable-length 2D numpy arrays and vstack them in rank order
    on every rank. Identity (returns arr) if MPI disabled / single rank."""
    c = _comm()
    if c is None or c.Get_size() == 1:
        return arr
    import numpy as np
    a = np.ascontiguousarray(arr)
    t0 = _tic()
    parts = c.allgather(a)
    _toc(t0)
    return np.vstack([p for p in parts if p.size])


def math_fsum(parts):
    import math
    return math.fsum(parts)


# ------------------------------------------------------------------ data shard

def shard(seq):
    """Return this rank's slice of `seq` as a list: seq[rank::world_size].
    Returns list(seq) unchanged when MPI disabled / single rank."""
    w = world_size()
    if w == 1:
        return list(seq)
    r = rank()
    return list(seq)[r::w]


def shard_indices(n):
    """The indices this rank owns out of range(n): np.arange(n)[rank::world]."""
    import numpy as np
    w = world_size()
    if w == 1:
        return np.arange(n)
    return np.arange(n)[rank()::w]


# ------------------------------------------------------------------ diagnostics

def summary():
    if not is_enabled():
        return "MPI disabled (PACEMAKER_MPI!=1)"
    return "MPI rank {}/{} (local_rank {}), reproducible={}".format(
        rank(), world_size(), local_rank(), is_reproducible())
