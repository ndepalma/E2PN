"""BasicS2ConvV2.forward (index + matmul) must equal the original gather + einsum."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vgtk.so3conv.modules import BasicS2ConvV2  # noqa: E402

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_forward_matches_reference(device):
    torch.manual_seed(0)
    conv = BasicS2ConvV2(5, 7, 13, 12).to(device)
    x = torch.randn(2, 5, 13, 9, 12, device=device)
    ref, out = conv.forward_reference(x), conv(x)
    assert out.shape == ref.shape == (2, 7, 9, 12)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    g_ref, = torch.autograd.grad(ref.pow(2).sum(), conv.W)
    g_out, = torch.autograd.grad(out.pow(2).sum(), conv.W)
    torch.testing.assert_close(g_out, g_ref, rtol=1e-3, atol=1e-3)
    assert set(conv.state_dict()) == {'W', 'trace_idxv_ori', 'trace_idxv_rot', 'trace_idx_ori', 'trace_idx_rot', 'idx_map'}
