"""Device-agnostic PyTorch replacements for the former CUDA extension (vgtk/cuda).

Each function reproduces the behaviour of the CUDA kernel it replaces, so the
code runs unchanged on MPS, CPU and CUDA. Only the ops that are reachable from
the live code were ported:

    gather_points            <- gathering_cuda_kernel.cu : gather_points_forward
    ball_query               <- grouping_cuda_kernel.cu  : ball_query
    furthest_point_sampling  <- grouping_cuda_kernel.cu  : furthest_point_sampling
    initial_anchor_query     <- grouping_cuda_kernel.cu  : initial_anchor_query
"""
import torch


def get_device(name=None):
    """Pick the compute device: explicit name, else mps > cuda > cpu."""
    if name is not None and name != 'auto':
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def synchronize(device=None):
    """Wait for pending GPU work (for timing). No-op on CPU."""
    kind = torch.device(device).type if device is not None else get_device().type
    if kind == 'cuda':
        torch.cuda.synchronize()
    elif kind == 'mps':
        torch.mps.synchronize()


def max_memory_allocated_gb(device=None):
    """Memory used by tensors in GB.

    CUDA reports the peak since the last reset. MPS has no peak counter, so the
    currently allocated amount is reported instead. CPU reports 0.
    """
    kind = torch.device(device).type if device is not None else get_device().type
    if kind == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    if kind == 'mps':
        return torch.mps.current_allocated_memory() / (1024 ** 3)
    return 0.0


def reset_peak_memory_stats(device=None):
    kind = torch.device(device).type if device is not None else get_device().type
    if kind == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def gather_points(points, idx):
    """[b, c, n] x [b, m] -> [b, c, m]. Differentiable w.r.t. points."""
    idx = idx.long().unsqueeze(1).expand(-1, points.shape[1], -1)
    return torch.gather(points, 2, idx)


def ball_query(new_xyz, xyz, radius, nsample):
    """For every query point, the first `nsample` support points (in index
    order) that lie strictly within `radius`.

    new_xyz: [b, 3, m] query points
    xyz:     [b, 3, n] support points
    returns: [b, m, nsample] int32 indices into xyz

    Matches the CUDA kernel, including its padding rules:
      * fewer than nsample-1 hits -> the found indices are repeated cyclically
      * exactly nsample-1 hits    -> the last slot is left at index 0
      * no hits                   -> all slots are index 0
    """
    b, _, m = new_xyz.shape
    n = xyz.shape[2]
    device = xyz.device

    d2 = torch.zeros(b, m, n, dtype=xyz.dtype, device=device)
    for axis in range(3):
        diff = new_xyz[:, axis, :, None] - xyz[:, axis, None, :]
        d2 = d2 + diff * diff
    mask = d2 < radius * radius

    # smallest `k` indices among the hits, ascending; misses are pushed to `n`
    k = min(nsample, n)
    order = torch.arange(n, device=device, dtype=torch.int32)
    key = torch.where(mask, order, torch.full_like(order, n))
    first = torch.topk(key, k, dim=-1, largest=False, sorted=True)[0]   # [b, m, k]

    cnt = mask.sum(-1, dtype=torch.int32).clamp(max=nsample).unsqueeze(-1)    # [b, m, 1]
    slot = torch.arange(nsample, device=device, dtype=torch.int32).expand(b, m, nsample)
    src = torch.remainder(slot, cnt.clamp(min=1))
    idx = torch.gather(first, 2, src.long())

    filled = (cnt > 0) & ((slot < cnt) | (cnt < nsample - 1))
    idx = torch.where(filled, idx, torch.zeros_like(idx))
    return idx.int().contiguous()


def furthest_point_sampling(xyz, m):
    """Iterative furthest point sampling, starting from point 0.

    xyz: [b, 3, n] -> [b, m] int32 indices.
    As in the CUDA kernel, points with squared norm <= 1e-3 are never selected
    after the first one.
    """
    b, _, n = xyz.shape
    device = xyz.device
    idxs = torch.zeros(b, m, dtype=torch.long, device=device)
    if m <= 1:
        return idxs.int()

    valid = (xyz * xyz).sum(1) > 1e-3                                   # [b, n]
    temp = torch.full((b, n), 1e10, dtype=xyz.dtype, device=device)
    neg = torch.full((b, n), -1.0, dtype=xyz.dtype, device=device)
    old = torch.zeros(b, dtype=torch.long, device=device)
    for j in range(1, m):
        centre = torch.gather(xyz, 2, old.view(b, 1, 1).expand(b, 3, 1))  # [b, 3, 1]
        d = ((xyz - centre) ** 2).sum(1)
        temp = torch.where(valid, torch.minimum(d, temp), temp)
        old = torch.argmax(torch.where(valid, temp, neg), dim=1)
        idxs[:, j] = old
    return idxs.int()


def initial_anchor_query(centers, xyz, kernel_points, radius, sigma, max_elements=2 ** 24):
    """Kernel-point weights of a fragment around a set of centers.

    centers:       [b, 3, nc]
    xyz:           [m, 3] fragment points (shared by the whole batch)
    kernel_points: [ks, na, 3]
    returns: (anchor_weights, anchor_ctn), both [b, ks, nc, na]

    For every fragment point within `radius` of a center, each kernel point
    receives max(0, 1 - dist^2 / sigma); anchor_ctn counts those points.
    """
    b, _, nc = centers.shape
    m = xyz.shape[0]
    ks, na, _ = kernel_points.shape
    dtype, device = xyz.dtype, xyz.device

    weights = torch.zeros(b, ks, nc, na, dtype=dtype, device=device)
    ctn = torch.zeros(b, 1, nc, 1, dtype=dtype, device=device)
    c = centers.transpose(1, 2)                                          # [b, nc, 3]
    kp = kernel_points.reshape(1, 1, 1, ks * na, 3)

    chunk = max(1, max_elements // max(1, b * nc * ks * na * 3))
    for start in range(0, m, chunk):
        pts = xyz[start:start + chunk]                                    # [mc, 3]
        rel = pts[None, None] - c[:, :, None]                             # [b, nc, mc, 3]
        inside = (rel.pow(2).sum(-1).sqrt() <= radius).to(dtype)          # [b, nc, mc]
        dist2 = (rel.unsqueeze(3) - kp).pow(2).sum(-1)                    # [b, nc, mc, ks*na]
        w = torch.relu(1.0 - dist2 / sigma) * inside.unsqueeze(-1)
        weights += w.sum(2).view(b, nc, ks, na).permute(0, 2, 1, 3)
        ctn += inside.sum(2).view(b, 1, nc, 1)

    return [weights, ctn.expand(b, ks, nc, na).contiguous()]
