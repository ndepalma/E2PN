"""The rotation anchors must match what the original code produced with its
pinned trimesh==3.2.0 (reference arrays generated with that version)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import vgtk.so3conv.functional as L  # noqa: E402

REF = np.load(os.path.join(os.path.dirname(__file__), 'anchors_trimesh32_reference.npz'))


def test_anchors_match_trimesh32_reference():
    np.testing.assert_allclose(L.Rs, REF['Rs'], atol=1e-6)
    np.testing.assert_array_equal(L.R_idx, REF['R_idx'])
    np.testing.assert_allclose(L.vs, REF['vs'], atol=1e-6)
    np.testing.assert_array_equal(np.array(L.v_adjs), REF['v_adjs'])
    np.testing.assert_array_equal(L.v_level2s, REF['v_level2s'])
    np.testing.assert_array_equal(L.v_opps, REF['v_opps'])
    np.testing.assert_allclose(L.vRs, REF['vRs'], atol=1e-6)


def test_anchor_rotations_form_a_group():
    for Rs in (L.get_anchors(60), L.get_anchorsV()):
        assert Rs.shape == (60, 3, 3)
        np.testing.assert_allclose(Rs @ Rs.transpose(0, 2, 1), np.broadcast_to(np.eye(3), Rs.shape), atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(Rs), 1.0, atol=1e-5)
        # closure: every product Ri Rj is again one of the 60 anchors, and each row is a permutation
        prod = np.einsum('iab,jbc->ijac', Rs, Rs)
        trace = np.einsum('ijac,kac->ijk', prod, Rs)
        np.testing.assert_allclose(trace.max(-1), 3.0, atol=1e-4)
        table = trace.argmax(-1)
        assert all(sorted(row) == list(range(60)) for row in table)


def test_vertex_permutation_indices():
    ori, rot = L.get_relativeV_index()
    assert ori.shape == (60, 12) and rot.shape == (60, 12)
    assert all(sorted(r) == list(range(12)) for r in ori)
    assert all(sorted(r) == list(range(12)) for r in rot)
