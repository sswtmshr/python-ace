"""pacetorch: PyTorch backend for pacemaker (linear/FS B-basis ACE), a drop-in for TensorPotential 0.2."""
__version__ = "0.1.0"


def select_device(gpu_config=None):
    """gpu_config {gpu_ind: i (-1 = CPU), device: 'cuda'|'xpu'|'mps'|'cpu'} -> torch.device"""
    import torch
    cfg = dict(gpu_config or {})
    if cfg.get("device"):
        return torch.device(cfg["device"])
    ind = int(cfg.get("gpu_ind", 0))
    if ind < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():          # NVIDIA CUDA and AMD ROCm builds
        return torch.device("cuda", ind)
    if hasattr(torch, "xpu") and torch.xpu.is_available():   # Intel GPUs (Aurora)
        return torch.device("xpu", ind)
    return torch.device("cpu")              # MPS has no float64: not used for fitting
