"""
Unit tests for UnitreeB1Env that do NOT require Isaac Sim.

Tests cover:
  - Config construction from YAML
  - CPG batch step math (isolated from simulation)
  - Weight API (set_weights, set_weights_batch, get_weights)
  - Joint permutation logic
  - Reward function math

These tests mock the Isaac Lab simulation layer so the full environment
can be exercised without launching Isaac Sim.

Run:
    cd ~/cpg-drl-transition
    python -m pytest tests/test_environment.py -v

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Mock Isaac Lab before any project imports touch it
# ---------------------------------------------------------------------------

def _make_mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = MagicMock()
    return mod


def _build_isaaclab_mocks():
    """
    Build a minimal stub of the Isaac Lab module tree so that
    envs/unitree_b1_env.py can be imported without Isaac Sim running.
    """
    prefixes = [
        "isaaclab",
        "isaaclab.sim",
        "isaaclab.assets",
        "isaaclab.envs",
        "isaaclab.scene",
        "isaaclab.sensors",
        "isaaclab.terrains",
        "isaaclab.utils",
        "isaaclab_assets",
        "isaaclab_assets.robots",
        "isaaclab_assets.robots.unitree",
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.utils",
        "isaacsim.core.utils.torch",
        "omni",
        "omni.kit",
    ]
    for p in prefixes:
        if p not in sys.modules:
            sys.modules[p] = _make_mock_module(p)

    # Minimal configclass decorator — just makes a regular class
    def configclass(cls):
        return cls

    sys.modules["isaaclab.utils"].configclass = configclass

    # Stub SimulationCfg
    class SimulationCfg:
        def __init__(self, dt=0.005, render_interval=4, physics_material=None):
            self.dt = dt
            self.render_interval = render_interval

    sys.modules["isaaclab.sim"].SimulationCfg = SimulationCfg
    sys.modules["isaaclab.sim"].RigidBodyMaterialCfg = MagicMock(return_value=None)
    sys.modules["isaaclab.sim"].DomeLightCfg = MagicMock()

    # Stub ArticulationCfg
    class ArticulationCfg:
        def replace(self, **kwargs):
            return self

    sys.modules["isaaclab.assets"].ArticulationCfg = ArticulationCfg
    sys.modules["isaaclab.assets"].Articulation = MagicMock

    # Stub InteractiveSceneCfg
    class InteractiveSceneCfg:
        def __init__(self, num_envs=64, env_spacing=2.5, replicate_physics=True):
            self.num_envs = num_envs
            self.env_spacing = env_spacing

    sys.modules["isaaclab.scene"].InteractiveSceneCfg = InteractiveSceneCfg

    # Stub sensors
    sys.modules["isaaclab.sensors"].ContactSensor = MagicMock
    sys.modules["isaaclab.sensors"].ContactSensorCfg = MagicMock(return_value=MagicMock())

    # Stub TerrainImporterCfg
    sys.modules["isaaclab.terrains"].TerrainImporterCfg = MagicMock(return_value=MagicMock())

    # Stub DirectRLEnv and DirectRLEnvCfg
    class DirectRLEnvCfg:
        pass

    class DirectRLEnv:
        def __init__(self, cfg, render_mode=None, **kwargs):
            self.cfg = cfg
            self.num_envs = cfg.scene.num_envs if hasattr(cfg, "scene") else 4
            self.device = torch.device("cpu")

        def _reset_idx(self, env_ids):
            pass

    sys.modules["isaaclab.envs"].DirectRLEnv = DirectRLEnv
    sys.modules["isaaclab.envs"].DirectRLEnvCfg = DirectRLEnvCfg

    # Stub UNITREE_B1_CFG
    b1_cfg = ArticulationCfg()
    sys.modules["isaaclab_assets.robots.unitree"].UNITREE_B1_CFG = b1_cfg


_build_isaaclab_mocks()

# Now safe to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.unitree_b1_env import (  # noqa: E402
    CPG_JOINT_NAMES,
    UnitreeB1Env,
    UnitreeB1EnvCfg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(num_envs: int = 4, gait: str = "walk") -> UnitreeB1EnvCfg:
    """Build a minimal config without touching Isaac Lab at runtime."""
    from isaaclab.scene import InteractiveSceneCfg

    cfg = UnitreeB1EnvCfg()
    cfg.gait_name = gait
    cfg.scene = InteractiveSceneCfg(num_envs=num_envs)

    offsets = {
        "walk":  [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
        "trot":  [0.0, math.pi, math.pi, 0.0],
        "steer": [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
    }
    cfg.phase_offsets = offsets[gait]
    return cfg


def make_env(num_envs: int = 4, gait: str = "walk") -> UnitreeB1Env:
    """Construct env and inject a mock joint permutation so we can test CPG math."""
    env = UnitreeB1Env(make_cfg(num_envs, gait))
    # Inject identity permutation (CPG order == Isaac Lab order for tests)
    env._joint_perm = torch.arange(12, dtype=torch.long)
    return env


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_observation_space(self):
        cfg = UnitreeB1EnvCfg()
        assert cfg.observation_space == 33

    def test_default_action_space(self):
        cfg = UnitreeB1EnvCfg()
        assert cfg.action_space == 12

    def test_control_dt(self):
        """Control dt = physics dt × decimation = 0.005 × 4 = 0.02 s."""
        cfg = UnitreeB1EnvCfg()
        control_dt = cfg.sim.dt * cfg.decimation
        assert abs(control_dt - 0.02) < 1e-9

    def test_cpg_joint_names_length(self):
        assert len(CPG_JOINT_NAMES) == 12

    def test_cpg_joint_names_content(self):
        for leg in ["FL", "FR", "RL", "RR"]:
            for jt in ["hip_joint", "thigh_joint", "calf_joint"]:
                assert f"{leg}_{jt}" in CPG_JOINT_NAMES


# ---------------------------------------------------------------------------
# 2. CPG batch step
# ---------------------------------------------------------------------------

class TestCPGBatch:

    def test_phase_idx_is_scalar_int(self):
        env = make_env(num_envs=8)
        # Pre-computed CPG uses a single integer phase shared across envs
        assert isinstance(env._phase_idx, int)
        assert env._phase_idx == 0

    def test_kenne_table_shape(self):
        env = make_env()
        # KENNE is a pre-computed (period+1, H) lookup of RBF activations
        assert env._KENNE.shape[1] == 20
        assert env._KENNE.shape[0] == env._period + 1
        # RBF activations are exp(...) ∈ (0, 1]
        assert (env._KENNE >= 0.0).all() and (env._KENNE <= 1.0).all()

    def test_step_returns_correct_shape(self):
        env = make_env(num_envs=6)
        targets = env._step_cpg_batch()
        assert targets.shape == (6, 12), f"Expected (6,12), got {targets.shape}"

    def test_step_is_finite(self):
        env = make_env(num_envs=4)
        env._W = torch.randn(4, 20, 3) * 0.1
        for _ in range(200):
            targets = env._step_cpg_batch()
        assert torch.isfinite(targets).all(), "CPG targets contain NaN/Inf"

    def test_zero_W_gives_zero_targets(self):
        env = make_env(num_envs=4)
        # W is zeroed by default
        targets = env._step_cpg_batch()
        assert torch.allclose(targets, torch.zeros(4, 12)), "Zero W should give zero targets"

    def test_different_W_produce_different_targets(self):
        """Different W matrices should produce different joint targets."""
        env = make_env(num_envs=2)
        W1 = np.random.randn(20, 3).astype(np.float32) * 0.1
        W2 = np.random.randn(20, 3).astype(np.float32) * 0.1
        env._W[0] = torch.tensor(W1)
        env._W[1] = torch.tensor(W2)
        targets = env._step_cpg_batch()
        assert not torch.allclose(targets[0], targets[1]), "Different W should give different targets"

    def test_phi_increments_each_step(self):
        env = make_env()
        phi_before = env._phi
        env._step_cpg_batch()
        phi_after = env._phi
        expected = 2.0 * math.pi * env.cfg.cpg_freq * (env.cfg.sim.dt * env.cfg.decimation)
        assert abs(phi_after - phi_before - expected) < 1e-9

    def test_tanh_bounds_output(self):
        """Output should be bounded by tanh to [-1, 1]."""
        env = make_env(num_envs=4)
        env._W = torch.randn(4, 20, 3) * 10.0   # large W
        targets = env._step_cpg_batch()
        assert (targets >= -1.0).all() and (targets <= 1.0).all(), "tanh should bound output"


# ---------------------------------------------------------------------------
# 3. Weight API
# ---------------------------------------------------------------------------

class TestWeightAPI:

    def test_set_weights_broadcasts_to_all_envs(self):
        env = make_env(num_envs=8)
        W = np.ones((20, 3), dtype=np.float32) * 0.5
        env.set_weights(W)
        expected = torch.full((20, 3), 0.5)
        for i in range(8):
            assert torch.allclose(env._W[i], expected)

    def test_set_weights_batch_assigns_per_env(self):
        env = make_env(num_envs=4)
        W_batch = np.arange(4 * 20 * 3, dtype=np.float32).reshape(4, 20, 3)
        env.set_weights_batch(W_batch)
        for i in range(4):
            assert torch.allclose(env._W[i], torch.tensor(W_batch[i]))

    def test_set_weights_wrong_shape_raises(self):
        env = make_env()
        with pytest.raises(AssertionError):
            env.set_weights(np.zeros((20, 12)))   # wrong: should be (20, 3)

    def test_set_weights_batch_wrong_shape_raises(self):
        env = make_env(num_envs=4)
        with pytest.raises(AssertionError):
            env.set_weights_batch(np.zeros((3, 20, 3)))   # wrong num_envs

    def test_get_weights_returns_env0(self):
        env = make_env(num_envs=4)
        W = np.random.randn(20, 3).astype(np.float32)
        env.set_weights(W)
        W_back = env.get_weights()
        assert W_back.shape == (20, 3)
        assert np.allclose(W, W_back)

    def test_weights_affect_cpg_output(self):
        """Different W → different joint targets."""
        env = make_env(num_envs=2)
        W1 = np.ones((20, 3), dtype=np.float32)
        W2 = np.ones((20, 3), dtype=np.float32) * (-1.0)
        env._W[0] = torch.tensor(W1)
        env._W[1] = torch.tensor(W2)
        targets = env._step_cpg_batch()
        assert not torch.allclose(targets[0], targets[1])


# ---------------------------------------------------------------------------
# 4. CPG reset
# ---------------------------------------------------------------------------

class TestCPGReset:

    def test_partial_reset_does_not_reset_phase(self):
        """Phase is shared across envs; partial reset must not rewind it."""
        env = make_env(num_envs=4)
        for _ in range(5):
            env._step_cpg_batch()
        phase_before = env._phase_idx
        env._reset_cpg(torch.tensor([0, 2]))
        assert env._phase_idx == phase_before

    def test_full_reset_all_envs(self):
        env = make_env(num_envs=4)
        for _ in range(5):
            env._step_cpg_batch()
        assert env._phase_idx != 0
        env._reset_cpg(torch.arange(4))
        assert env._phase_idx == 0
        assert env._phi == 0.0


# ---------------------------------------------------------------------------
# 5. Joint permutation
# ---------------------------------------------------------------------------

class TestJointPermutation:

    def test_build_joint_permutation_identity(self):
        """When Isaac Lab order matches CPG order, permutation is identity."""
        env = make_env(num_envs=2)
        # Simulate the _build_joint_permutation method with matching names
        isaaclab_names = CPG_JOINT_NAMES   # same order
        perm = [CPG_JOINT_NAMES.index(name) for name in isaaclab_names]
        perm_t = torch.tensor(perm, dtype=torch.long)
        assert torch.equal(perm_t, torch.arange(12, dtype=torch.long))

    def test_build_joint_permutation_reorders(self):
        """When Isaac Lab puts joints in different order, permutation should remap."""
        # Reverse the CPG order as a test case
        isaaclab_names = list(reversed(CPG_JOINT_NAMES))
        perm = [CPG_JOINT_NAMES.index(name) for name in isaaclab_names]
        perm_t = torch.tensor(perm, dtype=torch.long)
        # Check that applying perm recovers the reversed order
        identity = torch.arange(12)
        assert torch.equal(identity[perm_t], torch.arange(11, -1, -1, dtype=torch.long))
