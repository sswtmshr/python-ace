"""Fused GPU kernels (Triton) for the two A-basis primitives of ace._SegOuter / ace._GatherContract.

Both read only the per-pair factors and per-atom arrays, never a [pairs, n, m] intermediate:
  seg_outer(U, V, ptr, nat)        A[a, n, m] = sum_k sum_{p in segment a} U[k, p, n] V[k, p, m]
                                   (pairs sorted by central atom, segment a = ptr[a]:ptr[a+1]; one program per atom,
                                   accumulates in registers, writes once: no atomics, deterministic)
  gather_contract(G, idx, W, dim)  dim 2: out[p, n] = sum_m G[idx[p], n, m] W[p, m]
                                   dim 1: out[p, m] = sum_n G[idx[p], n, m] W[p, n]
They are forward kernels only: the derivatives stay in ace.py as compositions of the same two operations.
float64, CUDA (and other Triton GPU backends); `available(t)` says whether a tensor can use them."""
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


def available(t):
    return _HAVE_TRITON and t.is_cuda and t.dtype == torch.float64


def _p2(x):
    return max(2, triton.next_power_of_2(int(x)))


if _HAVE_TRITON:
    @triton.jit
    def _seg_outer_kernel(U, V, PTR, OUT, K, P, n, m, BN: tl.constexpr, BM: tl.constexpr, BP: tl.constexpr):
        a = tl.program_id(0)
        start = tl.load(PTR + a)
        end = tl.load(PTR + a + 1)
        rn = tl.arange(0, BN)
        rm = tl.arange(0, BM)
        rp = tl.arange(0, BP)
        acc = tl.zeros((BN, BM), dtype=tl.float64)
        for k in range(K):
            for p0 in range(start, end, BP):
                pp = p0 + rp
                mp = pp < end
                u = tl.load(U + k * P * n + pp[:, None] * n + rn[None, :],
                            mask=mp[:, None] & (rn[None, :] < n), other=0.)
                v = tl.load(V + k * P * m + pp[:, None] * m + rm[None, :],
                            mask=mp[:, None] & (rm[None, :] < m), other=0.)
                acc += tl.sum(u[:, :, None] * v[:, None, :], axis=0)
        tl.store(OUT + a * n * m + rn[:, None] * m + rm[None, :], acc, mask=(rn[:, None] < n) & (rm[None, :] < m))

    @triton.jit
    def _gather_contract_kernel(G, IDX, W, OUT, P, n, m, DIM: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr,
                                BP: tl.constexpr):
        pid = tl.program_id(0)
        pp = pid * BP + tl.arange(0, BP)
        mp = pp < P
        rn = tl.arange(0, BN)
        rm = tl.arange(0, BM)
        c = tl.load(IDX + pp, mask=mp, other=0)
        g = tl.load(G + c[:, None, None] * n * m + rn[None, :, None] * m + rm[None, None, :],
                    mask=mp[:, None, None] & (rn[None, :, None] < n) & (rm[None, None, :] < m), other=0.)
        if DIM == 2:
            w = tl.load(W + pp[:, None] * m + rm[None, :], mask=mp[:, None] & (rm[None, :] < m), other=0.)
            out = tl.sum(g * w[:, None, :], axis=2)
            tl.store(OUT + pp[:, None] * n + rn[None, :], out, mask=mp[:, None] & (rn[None, :] < n))
        else:
            w = tl.load(W + pp[:, None] * n + rn[None, :], mask=mp[:, None] & (rn[None, :] < n), other=0.)
            out = tl.sum(g * w[:, :, None], axis=1)
            tl.store(OUT + pp[:, None] * m + rm[None, :], out, mask=mp[:, None] & (rm[None, :] < m))


def seg_outer(us, vs, ptr, nat):
    U = torch.stack(us).contiguous()
    V = torch.stack(vs).contiguous()
    K, P, n = U.shape
    m = V.shape[2]
    out = U.new_empty(nat, n, m)
    if nat:
        BN, BM = _p2(n), _p2(m)
        BP = max(1, min(16, 4096 // (BN * BM)))
        _seg_outer_kernel[(nat,)](U, V, ptr, out, K, P, n, m, BN=BN, BM=BM, BP=BP)
    return out


def gather_contract(G, idx, w, dim):
    G = G.contiguous()
    w = w.contiguous()
    nat, n, m = G.shape
    P = w.shape[0]
    out = w.new_empty(P, n if dim == 2 else m)
    if P:
        BN, BM = _p2(n), _p2(m)
        BP = max(1, min(64, 4096 // (BN * BM)))
        _gather_contract_kernel[(triton.cdiv(P, BP),)](G, idx, w, out, P, n, m, DIM=dim, BN=BN, BM=BM, BP=BP)
    return out
