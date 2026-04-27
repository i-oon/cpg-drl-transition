"""
CPG-RBF network: SO(2) oscillator + RBF layer + indirect 4-leg encoding.

Implements the Thor et al. (2021) SO(2) oscillator with Gaussian RBF basis
functions. Uses Option B phase replication: one oscillator, state rotated per
leg to produce phase-shifted RBF activations.

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import numpy as np
import torch

# Leg order throughout this project: FL, FR, RL, RR
# Phase offsets in radians for each gait
PHASE_OFFSETS = {
    "walk":  np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2]),  # FL=0, FR=180, RL=90, RR=270
    "trot":  np.array([0.0, np.pi, np.pi, 0.0]),                 # FL-RR=0, FR-RL=180
    "steer": np.array([0.0, np.pi, np.pi / 2, 3 * np.pi / 2]),  # Same as walk
}


class CPGRBFNetwork:
    """
    CPG-RBF locomotion network with indirect 4-leg encoding.

    Architecture:
        SO(2) oscillator → RBF layer (H=20) → W (20×3) → 3 joints per leg
        Phase replication: rotate oscillator state by θ_k for each leg k.

    W matrix (20×3) maps RBF activations to [hip, thigh, calf] joint angles
    for one master leg. Output is replicated to all 4 legs via phase offsets.

    Args:
        alpha:   Self-excitation gain (paper: 1.01).
        freq:    Oscillator frequency in Hz (training: 0.3 Hz).
        dt:      Timestep in seconds (50 Hz control → 0.02 s).
        sigma2:  RBF variance (paper: 0.04).
        device:  Torch device string ("cpu" or "cuda").
    """

    NUM_RBF = 20
    NUM_JOINTS_PER_LEG = 3   # hip, thigh, calf
    NUM_LEGS = 4

    def __init__(
        self,
        alpha: float = 1.01,
        freq: float = 0.3,
        dt: float = 0.02,
        sigma2: float = 0.04,
        device: str = "cpu",
    ):
        self.alpha = alpha
        self.freq = freq
        self.dt = dt
        self.sigma2 = sigma2
        self.device = torch.device(device)

        # RBF centers evenly distributed on unit circle
        # c_i = (cos(2π·i/H), sin(2π·i/H)) for i = 0, …, H-1
        angles = 2.0 * np.pi * np.arange(self.NUM_RBF) / self.NUM_RBF
        centers_np = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (20, 2)
        self.centers = torch.tensor(centers_np, dtype=torch.float32, device=self.device)

        # Oscillator state: initialized on unit circle at phase 0
        self._osc_state = torch.tensor([1.0, 0.0], device=self.device)

        # Current phase angle φ (radians) — drives the rotation matrix
        self._phi = 0.0

        # Weight matrix W ∈ ℝ^(20×3): zeroed, set by PIBB training
        self.W = torch.zeros(self.NUM_RBF, self.NUM_JOINTS_PER_LEG, device=self.device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset oscillator to initial state (phase 0, point (1, 0))."""
        self._osc_state = torch.tensor([1.0, 0.0], device=self.device)
        self._phi = 0.0

    def step(self, phase_offsets: np.ndarray) -> torch.Tensor:
        """
        Advance oscillator one timestep and return joint angles for all legs.

        Args:
            phase_offsets: (4,) array of phase offsets in radians [FL, FR, RL, RR].

        Returns:
            joint_angles: (4, 3) tensor — rows are legs, columns are joints.
        """
        self._step_oscillator()

        joint_angles = []
        for theta in phase_offsets:
            x_rot, y_rot = self._rotate_state(float(theta))
            rbf = self._compute_rbf(x_rot, y_rot)   # (20,)
            joints = rbf @ self.W                    # (20,) @ (20, 3) = (3,)
            joint_angles.append(joints)

        return torch.stack(joint_angles)  # (4, 3)

    def get_osc_state(self) -> torch.Tensor:
        """Return current oscillator (x, y) state — used in observation space."""
        return self._osc_state.clone()

    def get_phi(self) -> float:
        """Return current phase angle φ in radians."""
        return self._phi

    def set_weights(self, W: np.ndarray):
        """Set W matrix from a (20, 3) numpy array."""
        assert W.shape == (self.NUM_RBF, self.NUM_JOINTS_PER_LEG), (
            f"Expected W shape (20, 3), got {W.shape}"
        )
        self.W = torch.tensor(W, dtype=torch.float32, device=self.device)

    def save_weights(self, path: str):
        """Save W matrix to a .npy file."""
        np.save(path, self.W.cpu().numpy())

    def load_weights(self, path: str):
        """Load W matrix from a .npy file."""
        W_np = np.load(path)
        self.set_weights(W_np)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_oscillator(self):
        """Update oscillator state: o(t+1) = tanh(α·R(Δφ) @ o(t)), then φ += Δφ.

        Uses FIXED rotation angle Δφ per step (not accumulated φ).
        The SO(2) oscillator applies the same rotation at every step.
        """
        delta_phi = 2.0 * np.pi * self.freq * self.dt   # fixed per step
        cos_dp = float(np.cos(delta_phi))
        sin_dp = float(np.sin(delta_phi))

        R = self.alpha * torch.tensor(
            [[cos_dp,  sin_dp],
             [-sin_dp, cos_dp]],
            dtype=torch.float32,
            device=self.device,
        )
        self._osc_state = torch.tanh(R @ self._osc_state)
        self._phi += delta_phi   # track accumulated phase for offsets

    def _rotate_state(self, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize oscillator state to unit circle, then rotate by theta.

        The SO(2) oscillator with alpha=1.01 converges to a limit cycle of
        amplitude ~0.196 (tanh saturation pulls the state inward). Normalizing
        projects the state onto the unit circle so that RBF centers — which are
        evenly distributed on the unit circle — are actually reachable.
        The phase direction is preserved; only the amplitude is discarded.

        Rotation convention: [cos θ, -sin θ; sin θ, cos θ] @ [x_n; y_n]
        """
        norm = self._osc_state.norm()
        x_n = self._osc_state[0] / norm
        y_n = self._osc_state[1] / norm

        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x_rot = cos_t * x_n - sin_t * y_n
        y_rot = sin_t * x_n + cos_t * y_n
        return x_rot, y_rot

    def _compute_rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute 20 RBF activations for oscillator position (x, y).

        RBF_i = exp(-‖(x,y) - center_i‖² / σ²)

        Returns:
            rbf: (20,) activation tensor.
        """
        pos = torch.stack([x, y])              # (2,)
        diff = pos.unsqueeze(0) - self.centers  # (20, 2)
        dist2 = (diff ** 2).sum(dim=1)          # (20,)
        return torch.exp(-dist2 / self.sigma2)   # (20,)
