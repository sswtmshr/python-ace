"""pacetorch: PyTorch backend for pacemaker (linear/FS B-basis ACE), a drop-in for TensorPotential 0.2."""
import logging



def _git_version(default):
    """`git describe` of this checkout (editable installs: the tag the running code is at, e.g. v0.2.3 or
    v0.2.3-2-gabc1234 for dev), else the packaged version"""
    import os
    import subprocess
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", d, "describe", "--tags", "--dirty"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=5, check=True, universal_newlines=True).stdout.strip()
        return out.lstrip("v") or default
    except Exception:  # noqa: BLE001 - no git, not a checkout: packaged version
        return default


__version__ = _git_version("0.2.3")
log = logging.getLogger(__name__)


def choose_index(n_visible, gpu_ind, distributed, local_rank):
    """Device index among the n_visible devices this process can see.

    MPI (data-parallel) runs: rank -> local_rank % n_visible, whatever gpu_ind says. With select_gpu.sh (one visible
    GPU per rank) that is 0, as before; if every rank sees every GPU it spreads the ranks instead of stacking them
    all on device 0. Single process: gpu_ind (default 0)."""
    if distributed:
        return local_rank % n_visible
    return int(gpu_ind) % n_visible


def select_device(gpu_config=None):
    """gpu_config {gpu_ind: i (-1 = CPU), device: 'cuda'|'xpu'|'cpu'|'cuda:1'...} -> torch.device

    CUDA covers NVIDIA and AMD ROCm builds of torch; xpu is Intel (Aurora). MPS (Apple) is never chosen: no float64."""
    import torch
    cfg = dict(gpu_config or {})
    if cfg.get("device"):
        dev = torch.device(cfg["device"])
        if dev.type == "cuda" and dev.index is not None:
            torch.cuda.set_device(dev)
        return dev
    ind = int(cfg.get("gpu_ind", 0))
    if ind < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        kind, n = "cuda", torch.cuda.device_count()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        kind, n = "xpu", torch.xpu.device_count()
    else:
        return torch.device("cpu")
    try:
        from pyace import mpi
        dist, lrank = mpi.is_distributed(), mpi.local_rank()
    except ImportError:  # pragma: no cover
        dist, lrank = False, 0
    i = choose_index(n, ind, dist, lrank)
    dev = torch.device(kind, i)
    if kind == "cuda":   # the current device must be the model's: Triton kernels launch on the current device
        torch.cuda.set_device(dev)
    if dist:
        log.info("[MPI rank {} / local {}] {} visible {} device(s) -> {}".format(mpi.rank(), lrank, n, kind, dev))
    return dev


def _device_uuid(dev):
    import torch
    try:
        mod = torch.cuda if dev.type == "cuda" else torch.xpu
        return str(mod.get_device_properties(dev).uuid)
    except Exception:  # noqa: BLE001 - no uuid on this backend: cannot check
        return None


_shared_checked = False


def check_shared_devices(dev, mpi):
    """Ranks on the same node must not share a physical GPU (the launcher decides what each rank can see; if it shows
    one GPU to several ranks, no index choice can fix it). COLLECTIVE: call only where every rank is (the fitter's
    fit()); runs once per process. Warns, does not fail."""
    global _shared_checked
    if _shared_checked or dev.type == "cpu":
        return
    _shared_checked = True
    import socket
    ids = mpi.node_allgather(_device_uuid(dev))
    if mpi.local_rank() != 0 or any(u is None for u in ids):
        return
    seen = {}
    for lr, u in enumerate(ids):
        seen.setdefault(u, []).append(lr)
    shared = {u: r for u, r in seen.items() if len(r) > 1}
    for u, r in shared.items():
        log.warning("{}: local ranks {} all run on the SAME physical GPU {} -- the launcher shows these ranks only one "
                    "GPU (check srun --gpus-per-task / select_gpu.sh / ZE_AFFINITY_MASK). The fit is correct but "
                    "slow.".format(socket.gethostname(), r, u))
