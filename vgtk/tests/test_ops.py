"""Parity tests: vectorised ops in vgtk.ops vs. loop-for-loop transcriptions
of the original CUDA kernels."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vgtk import ops  # noqa: E402

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


def ref_ball_query(new_xyz, xyz, radius, nsample):
    b, _, m = new_xyz.shape
    n = xyz.shape[2]
    idx = np.zeros((b, m, nsample), dtype=np.int32)
    r2 = np.float32(radius) * np.float32(radius)
    for bi in range(b):
        for j in range(m):
            cnt = 0
            for k in range(n):
                if cnt >= nsample:
                    break
                d = new_xyz[bi, :, j] - xyz[bi, :, k]
                if (d * d).sum(dtype=np.float32) < r2:
                    idx[bi, j, cnt] = k
                    cnt += 1
            if cnt < nsample - 1:
                k = 0
                while k + cnt < nsample:
                    idx[bi, j, k + cnt] = idx[bi, j, k]
                    k += 1
    return idx


def ref_fps(xyz, m):
    b, _, n = xyz.shape
    out = np.zeros((b, m), dtype=np.int32)
    for bi in range(b):
        temp = np.full(n, 1e10, dtype=np.float32)
        old = 0
        for j in range(1, m):
            besti, best = 0, np.float32(-1)
            for k in range(n):
                p = xyz[bi, :, k]
                if (p * p).sum() <= 1e-3:
                    continue
                d = ((p - xyz[bi, :, old]) ** 2).sum()
                d2 = min(d, temp[k])
                temp[k] = d2
                if d2 > best:
                    besti, best = k, d2
            old = besti
            out[bi, j] = old
    return out


def ref_initial_anchor_query(centers, xyz, kp, radius, sigma):
    b, _, nc = centers.shape
    ks, na, _ = kp.shape
    w = np.zeros((b, ks, nc, na), dtype=np.float64)
    ctn = np.zeros((b, ks, nc, na), dtype=np.float64)
    for bi in range(b):
        for pm in range(xyz.shape[0]):
            for pn in range(nc):
                c = centers[bi, :, pn]
                if np.sqrt(((c - xyz[pm]) ** 2).sum()) <= radius:
                    for kn in range(ks):
                        for an in range(na):
                            d2 = ((kp[kn, an] + c - xyz[pm]) ** 2).sum()
                            weight = 1 - d2 / sigma
                            if weight > 0:
                                w[bi, kn, pn, an] += weight
                            ctn[bi, kn, pn, an] += 1
    return w, ctn


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('radius,nsample', [(0.15, 8), (0.3, 16), (0.6, 12), (0.02, 4), (5.0, 80)])
def test_ball_query(device, radius, nsample):
    g = torch.Generator().manual_seed(0)
    xyz = torch.rand(2, 3, 60, generator=g)
    new_xyz = xyz[:, :, ::2].contiguous()
    new_xyz[:, :, 0] = 10.0     # a query with no neighbours at all
    got = ops.ball_query(new_xyz.to(device), xyz.to(device), radius, nsample)
    assert got.dtype == torch.int32 and got.shape == (2, 30, nsample)
    np.testing.assert_array_equal(got.cpu().numpy(), ref_ball_query(new_xyz.numpy(), xyz.numpy(), radius, nsample))


@pytest.mark.parametrize('device', DEVICES)
def test_ball_query_hits_every_padding_branch(device):
    # 1 query at the origin, support points placed so the hit count is exact
    for hits in range(0, 7):
        xyz = torch.full((1, 3, 10), 5.0)
        xyz[0, :, 2:2 + hits] = 0.01
        q = torch.zeros(1, 3, 1)
        got = ops.ball_query(q.to(device), xyz.to(device), 0.5, 5).cpu().numpy()
        np.testing.assert_array_equal(got, ref_ball_query(q.numpy(), xyz.numpy(), 0.5, 5))


@pytest.mark.parametrize('device', DEVICES)
def test_furthest_point_sampling(device):
    g = torch.Generator().manual_seed(1)
    xyz = torch.rand(3, 3, 97, generator=g) - 0.5
    xyz[:, :, 5] = 0.0          # near-origin point must be skipped
    got = ops.furthest_point_sampling(xyz.to(device), 33)
    assert got.dtype == torch.int32
    np.testing.assert_array_equal(got.cpu().numpy(), ref_fps(xyz.numpy(), 33))


@pytest.mark.parametrize('device', DEVICES)
def test_gather_points_and_grad(device):
    g = torch.Generator().manual_seed(2)
    pts = torch.rand(2, 4, 9, generator=g).to(device).requires_grad_()
    idx = torch.randint(0, 9, (2, 6), generator=g, dtype=torch.int32).to(device)
    out = ops.gather_points(pts, idx)
    p, i = pts.detach().cpu().numpy(), idx.cpu().numpy()
    ref = np.stack([p[bi][:, i[bi]] for bi in range(2)])
    np.testing.assert_allclose(out.detach().cpu().numpy(), ref)
    out.sum().backward()
    counts = np.stack([np.bincount(i[bi], minlength=9) for bi in range(2)])[:, None].repeat(4, 1)
    np.testing.assert_allclose(pts.grad.cpu().numpy(), counts)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('max_elements', [2 ** 24, 2000])
def test_initial_anchor_query(device, max_elements):
    g = torch.Generator().manual_seed(3)
    xyz = torch.rand(40, 3, generator=g)
    centers = torch.rand(2, 3, 5, generator=g)
    kp = (torch.rand(4, 3, 3, generator=g) - 0.5) * 0.4
    w, ctn = ops.initial_anchor_query(centers.to(device), xyz.to(device), kp.to(device), 0.5, 0.1,
                                      max_elements=max_elements)
    rw, rc = ref_initial_anchor_query(centers.numpy(), xyz.numpy(), kp.numpy(), 0.5, 0.1)
    np.testing.assert_allclose(w.cpu().numpy(), rw, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ctn.cpu().numpy(), rc)
