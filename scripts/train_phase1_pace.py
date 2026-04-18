"""
Phase 1 – Task 3: Train W_pace via PIBB.

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/train_phase1_pace.py --headless

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train W_pace (Phase 1 PIBB)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", type=str, default="configs/phase1_pace.yaml")
parser.add_argument("--num_envs", type=int, default=None)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from envs.unitree_b1_env import make_env_from_config
from algorithms.pibb_trainer import PIBBTrainer

root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
for _h in root_log.handlers[:]:
    root_log.removeHandler(_h)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
root_log.addHandler(_handler)
logger = logging.getLogger(__name__)

config_path = Path(args.config)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

log_dir = Path(config.get("logging", {}).get("log_dir", "logs/phase1"))

logger.info("Building environment from %s", config_path)
env = make_env_from_config(str(config_path), num_envs=args.num_envs)

logger.info("Starting PIBB training — pace gait")
trainer = PIBBTrainer(env, config, log_dir=log_dir)

# Warm-start from hand-designed W_init
init_path = Path("weights/W_pace_init.npy")
if init_path.exists():
    trainer.W = np.load(init_path).astype(np.float32)
    logger.info("Warm-started from %s (norm=%.3f)", init_path, np.linalg.norm(trainer.W))

W_pace = trainer.train()

logger.info("W_pace saved to %s", config["output"]["weights_path"])

env.close()
sim_app.close()
