"""
Phase 1 – Task 1: Train W_walk via PIBB.

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/train_phase1_walk.py --headless
    python scripts/train_phase1_walk.py --headless --num_envs 32   # override envs

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# ---- Isaac Sim must launch first ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train W_walk (Phase 1 PIBB)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", type=str, default="configs/phase1_walk.yaml")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override num_envs from config")
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---- Safe to import project modules now ----
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from envs.unitree_b1_env import UnitreeB1Env, UnitreeB1EnvCfg, make_env_from_config
from algorithms.pibb_trainer import PIBBTrainer

# Isaac Lab may have already set up Python logging — force-reconfigure so our
# messages appear on stdout alongside Isaac Lab's own output.
root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
for _h in root_log.handlers[:]:
    root_log.removeHandler(_h)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
root_log.addHandler(_handler)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

config_path = Path(args.config)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

log_dir = Path(config.get("logging", {}).get("log_dir", "logs/phase1"))

logger.info("Building environment from %s", config_path)
env = make_env_from_config(str(config_path), num_envs=args.num_envs)

logger.info("Starting PIBB training — walk gait (direct encoding)")
trainer = PIBBTrainer(env, config, log_dir=log_dir)

# Random init (LocoNets approach — no warm-start needed with tanh output)
# W starts at zero; PIBB explores from scratch
W_walk = trainer.train()

logger.info("W_walk saved to %s", config["output"]["weights_path"])

env.close()
sim_app.close()
