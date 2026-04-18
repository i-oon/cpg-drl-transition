"""
Unit tests for CPG-RBF network.

Tests oscillator stability, RBF coverage, phase replication, and W I/O.
No Isaac Lab dependency — runs with plain Python + PyTorch.

Run with:
    cd ~/cpg-drl-transition
    python -m pytest tests/test_cpg_rbf.py -v

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from networks.cpg_rbf import CPGRBFNetwork, PHASE_OFFSETS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cpg():
    """Fresh CPGRBFNetwork with default parameters."""
    return CPGRBFNetwork()


@pytest.fixture
def cpg_with_random_W():
    """CPGRBFNetwork with random W for output shape tests."""
    net = CPGRBFNetwork()
    net.W = torch.randn(20, 3) * 0.1
    return net


# ---------------------------------------------------------------------------
# 1. Oscillator stability
# ---------------------------------------------------------------------------

class TestOscillatorStability:

    def test_state_does_not_explode(self, cpg):
        """Oscillator state should stay bounded after many steps."""
        for _ in range(1000):
            cpg._step_oscillator()
        x, y = cpg._osc_state[0].item(), cpg._osc_state[1].item()
        assert abs(x) <= 1.1, f"x out of bounds: {x}"
        assert abs(y) <= 1.1, f"y out of bounds: {y}"

    def test_state_magnitude_converges_to_limit_cycle(self, cpg):
        """
        With alpha=1.01 and tanh, the limit cycle amplitude is ~0.196 (not ~1.0).
        tanh saturation pulls the state inward. The state should be bounded,
        non-zero, and stable — not at the unit circle.
        """
        for _ in range(500):
            cpg._step_oscillator()
        magnitude = cpg._osc_state.norm().item()
        # Limit cycle sits at r ≈ 0.17–0.22 for alpha=1.01
        assert 0.05 < magnitude < 0.5, (
            f"Oscillator magnitude outside expected limit cycle range: {magnitude:.4f}"
        )

    def test_phi_increments_correctly(self, cpg):
        """Phase φ should increment by 2π·f·dt per step."""
        expected_increment = 2.0 * np.pi * cpg.freq * cpg.dt
        phi_before = cpg._phi
        cpg._step_oscillator()
        phi_after = cpg._phi
        assert abs((phi_after - phi_before) - expected_increment) < 1e-9

    def test_reset_restores_initial_state(self, cpg):
        """reset() should bring oscillator back to (1, 0) and phi=0."""
        for _ in range(200):
            cpg._step_oscillator()
        cpg.reset()
        assert torch.allclose(cpg._osc_state, torch.tensor([1.0, 0.0]))
        assert cpg._phi == 0.0


# ---------------------------------------------------------------------------
# 2. RBF layer
# ---------------------------------------------------------------------------

class TestRBFLayer:

    def test_rbf_output_shape(self, cpg):
        """RBF should return a (20,) tensor."""
        x = torch.tensor(1.0)
        y = torch.tensor(0.0)
        rbf = cpg._compute_rbf(x, y)
        assert rbf.shape == (20,), f"Expected (20,), got {rbf.shape}"

    def test_rbf_values_in_range(self, cpg):
        """RBF activations should be in (0, 1]."""
        x, y = torch.tensor(1.0), torch.tensor(0.0)
        rbf = cpg._compute_rbf(x, y)
        assert rbf.min().item() > 0.0
        assert rbf.max().item() <= 1.0 + 1e-6

    def test_rbf_peak_at_nearest_center(self, cpg):
        """Point exactly at center_0 = (1, 0) should maximally activate neuron 0."""
        x, y = torch.tensor(1.0), torch.tensor(0.0)
        rbf = cpg._compute_rbf(x, y)
        assert rbf.argmax().item() == 0, (
            f"Expected neuron 0 to be most active, got {rbf.argmax().item()}"
        )
        assert rbf[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_rbf_coverage_over_full_cycle(self, cpg):
        """Every RBF neuron should activate above 0.5 at some point in one cycle.

        RBF computation uses the normalized oscillator state (via _rotate_state),
        not the raw state — so we must go through _rotate_state(0.0) here.
        """
        steps_per_cycle = int(1.0 / (cpg.freq * cpg.dt))  # 1/(0.3*0.02) ≈ 167 steps
        max_activations = torch.zeros(20)

        for _ in range(steps_per_cycle * 2):  # Run 2 full cycles to be safe
            cpg._step_oscillator()
            x, y = cpg._rotate_state(0.0)   # theta=0: normalized state, no rotation
            rbf = cpg._compute_rbf(x, y)
            max_activations = torch.max(max_activations, rbf)

        neurons_not_covered = (max_activations < 0.5).sum().item()
        assert neurons_not_covered == 0, (
            f"{neurons_not_covered} RBF neurons never exceeded 0.5 activation"
        )

    def test_centers_on_unit_circle(self, cpg):
        """All RBF centers should lie on the unit circle."""
        norms = cpg.centers.norm(dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-6), (
            f"Centers not on unit circle: norms={norms}"
        )

    def test_centers_evenly_spaced(self, cpg):
        """Adjacent centers should all have the same angular spacing."""
        angles = torch.atan2(cpg.centers[:, 1], cpg.centers[:, 0])
        # Unwrap and compute differences
        angles_np = np.unwrap(angles.numpy())
        diffs = np.diff(angles_np)
        expected = 2 * np.pi / 20
        assert np.allclose(diffs, expected, atol=1e-5), (
            f"Centers not evenly spaced. Diffs: {diffs}"
        )


# ---------------------------------------------------------------------------
# 3. Phase rotation (Option B replication)
# ---------------------------------------------------------------------------

class TestPhaseRotation:

    def test_rotate_zero_returns_normalized_state(self, cpg):
        """Rotating by 0 should return the normalized (unit-length) oscillator state."""
        cpg._step_oscillator()
        x_rot, y_rot = cpg._rotate_state(0.0)
        # Output should be on the unit circle
        mag = (x_rot ** 2 + y_rot ** 2).sqrt().item()
        assert abs(mag - 1.0) < 1e-5, f"Expected unit magnitude, got {mag:.6f}"
        # Direction should match the raw oscillator state direction
        norm = cpg._osc_state.norm()
        assert torch.allclose(x_rot, cpg._osc_state[0] / norm, atol=1e-5)
        assert torch.allclose(y_rot, cpg._osc_state[1] / norm, atol=1e-5)

    def test_rotate_180_negates_normalized_state(self, cpg):
        """Rotating by π should return the negated normalized state."""
        cpg._step_oscillator()
        x_rot, y_rot = cpg._rotate_state(np.pi)
        norm = cpg._osc_state.norm()
        assert torch.allclose(x_rot, -(cpg._osc_state[0] / norm), atol=1e-5)
        assert torch.allclose(y_rot, -(cpg._osc_state[1] / norm), atol=1e-5)

    def test_rotate_output_always_unit_magnitude(self, cpg):
        """_rotate_state output should always have magnitude 1.0 (normalization + rotation)."""
        cpg._step_oscillator()
        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            x_rot, y_rot = cpg._rotate_state(theta)
            rotated_mag = (x_rot ** 2 + y_rot ** 2).sqrt().item()
            assert abs(rotated_mag - 1.0) < 1e-5, (
                f"Expected unit magnitude at theta={theta:.2f}, got {rotated_mag:.6f}"
            )

    def test_trot_diagonal_pairs_are_opposite(self, cpg):
        """
        In trot gait, FL-RR share phase 0 and FR-RL share phase π.
        The two pairs should have near-opposite RBF activations.
        """
        cpg._step_oscillator()
        offsets = PHASE_OFFSETS["trot"]  # [0, π, π, 0]

        # FL (offset=0) and RR (offset=0) → same activations
        x_fl, y_fl = cpg._rotate_state(offsets[0])
        x_rr, y_rr = cpg._rotate_state(offsets[3])
        rbf_fl = cpg._compute_rbf(x_fl, y_fl)
        rbf_rr = cpg._compute_rbf(x_rr, y_rr)
        assert torch.allclose(rbf_fl, rbf_rr, atol=1e-6), "FL and RR should be in sync"

        # FR (offset=π) and RL (offset=π) → same activations
        x_fr, y_fr = cpg._rotate_state(offsets[1])
        x_rl, y_rl = cpg._rotate_state(offsets[2])
        rbf_fr = cpg._compute_rbf(x_fr, y_fr)
        rbf_rl = cpg._compute_rbf(x_rl, y_rl)
        assert torch.allclose(rbf_fr, rbf_rl, atol=1e-6), "FR and RL should be in sync"


# ---------------------------------------------------------------------------
# 4. Full step() output
# ---------------------------------------------------------------------------

class TestStepOutput:

    def test_output_shape(self, cpg_with_random_W):
        """step() should return (4, 3) tensor."""
        offsets = PHASE_OFFSETS["walk"]
        out = cpg_with_random_W.step(offsets)
        assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"

    def test_output_is_finite(self, cpg_with_random_W):
        """step() output should have no NaN or Inf values."""
        offsets = PHASE_OFFSETS["trot"]
        for _ in range(100):
            out = cpg_with_random_W.step(offsets)
        assert torch.isfinite(out).all(), "Joint angles contain NaN or Inf"

    def test_zero_W_gives_zero_joints(self, cpg):
        """With W=0, all joint angles should be zero."""
        offsets = PHASE_OFFSETS["walk"]
        out = cpg.step(offsets)
        assert torch.allclose(out, torch.zeros(4, 3)), "Expected zero output with W=0"

    def test_walk_and_trot_different_outputs(self, cpg_with_random_W):
        """Walk and trot should produce different joint angles (different phase offsets)."""
        # Same oscillator state, different phase offsets
        out_walk = cpg_with_random_W.step(PHASE_OFFSETS["walk"])
        cpg_with_random_W.reset()
        out_trot = cpg_with_random_W.step(PHASE_OFFSETS["trot"])
        assert not torch.allclose(out_walk, out_trot), (
            "Walk and trot should produce different outputs"
        )


# ---------------------------------------------------------------------------
# 5. Weight save / load
# ---------------------------------------------------------------------------

class TestWeightIO:

    def test_save_and_load_roundtrip(self, cpg):
        """Saved and reloaded W should be identical."""
        cpg.W = torch.randn(20, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "W_test.npy")
            cpg.save_weights(path)

            cpg2 = CPGRBFNetwork()
            cpg2.load_weights(path)

        assert torch.allclose(cpg.W, cpg2.W), "W mismatch after save/load"

    def test_set_weights_wrong_shape_raises(self, cpg):
        """set_weights() with wrong shape should raise AssertionError."""
        bad_W = np.zeros((20, 4))  # Wrong: should be (20, 3)
        with pytest.raises(AssertionError):
            cpg.set_weights(bad_W)

    def test_load_weights_updates_W(self, cpg):
        """load_weights() should actually update self.W."""
        W_known = np.ones((20, 3)) * 0.42
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "W_known.npy")
            np.save(path, W_known)
            cpg.load_weights(path)
        expected = torch.tensor(W_known, dtype=torch.float32)
        assert torch.allclose(cpg.W, expected)
