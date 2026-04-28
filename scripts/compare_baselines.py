"""
Compare baseline methods from per-step CSV files.

Reads CSVs produced by play_b1_phase2.py (--save_csv) and prints a
formatted comparison table across all methods.

Usage:
    python scripts/compare_baselines.py \\
        --csvs \\
            discrete:logs/phase2/baselines/discrete/playback.csv \\
            linear_ramp:logs/phase2/baselines/linear_ramp/playback.csv \\
            smoothstep_ramp:logs/phase2/baselines/smoothstep_ramp/playback.csv \\
            e2e:logs/phase2/baselines/e2e/playback.csv \\
            residual_v7:logs/phase2/phase2_v7/playback.csv

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False

parser = argparse.ArgumentParser(description="Compare Phase 2 baseline methods from CSV logs")
parser.add_argument("--csvs", nargs="+", required=True,
                    metavar="LABEL:PATH",
                    help="Space-separated list of label:csv_path pairs.")
parser.add_argument("--robot_mass_kg", type=float, default=50.0)
parser.add_argument("--g", type=float, default=9.81)
parser.add_argument("--control_dt", type=float, default=0.02,
                    help="Control step duration in seconds (sim.dt × decimation = 0.005 × 4 = 0.02).")
args = parser.parse_args()


def load_csv(path: str) -> dict:
    """Load a playback CSV and return a dict of numpy arrays keyed by column."""
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    cols = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except ValueError:
                cols[k].append(v)
    return {k: np.array(v) if isinstance(v[0], float) else v for k, v in cols.items()}


def compute_metrics(data: dict, mass_kg: float, g: float, control_dt: float) -> dict:
    vx = data["vx"]
    h  = data["h"]
    tilt = data["tilt"]
    power = data.get("power_W", data.get("power_inst_hist", None))
    x_w = data.get("x_w", data.get("x_pos_hist", None))

    # Joint acceleration RMS (ja0..ja11)
    ja_cols = [f"ja{i}" for i in range(12)]
    ja = np.stack([data[c] for c in ja_cols if c in data], axis=1) if ja_cols[0] in data else None

    # CoT
    if power is not None and x_w is not None:
        distance = max(x_w[-1] - x_w[0], 1e-6)
        energy   = float(np.sum(power) * control_dt)
        cot      = energy / (mass_kg * g * distance)
    else:
        cot = float("nan")

    return {
        "vx_mean":     float(np.mean(vx)),
        "vx_std":      float(np.std(vx)),
        "tilt_mean":   float(np.mean(tilt)),
        "tilt_max":    float(np.max(tilt)),
        "h_mean":      float(np.mean(h)),
        "CoT":         cot,
        "jacc_rms":    float(np.sqrt(np.mean(ja ** 2))) if ja is not None else float("nan"),
        "n_steps":     len(vx),
    }


def fmt(val, fmt_str="+.3f"):
    if np.isnan(val):
        return "  —  "
    return format(val, fmt_str)


results = {}
for entry in args.csvs:
    if ":" not in entry:
        print(f"ERROR: expected LABEL:PATH, got '{entry}'")
        sys.exit(1)
    label, csv_path = entry.split(":", 1)
    p = Path(csv_path)
    if not p.exists():
        print(f"  WARNING: CSV not found for '{label}': {csv_path}")
        results[label] = None
        continue
    data = load_csv(str(p))
    results[label] = compute_metrics(data, args.robot_mass_kg, args.g, args.control_dt)
    print(f"  Loaded {label}: {len(data.get('vx', []))} steps from {csv_path}")

# Print comparison table
sep  = "=" * 90
sep2 = "-" * 90
print(f"\n{sep}")
print("  BASELINE COMPARISON — Phase 2 gait transition quality")
print(sep)

hdr = f"  {'Method':<22} {'vx_mean':>8} {'vx_std':>7} {'tilt_mean':>10} {'tilt_max':>9} {'h_mean':>7} {'CoT':>7} {'jacc_RMS':>9}"
print(hdr)
print(f"  {sep2}")

for label, m in results.items():
    if m is None:
        print(f"  {label:<22} (CSV not found)")
        continue
    print(
        f"  {label:<22}"
        f" {fmt(m['vx_mean']):>8}"
        f" {fmt(m['vx_std'], '.3f'):>7}"
        f" {fmt(m['tilt_mean'], '.4f'):>10}"
        f" {fmt(m['tilt_max'], '.4f'):>9}"
        f" {fmt(m['h_mean'], '.3f'):>7}"
        f" {fmt(m['CoT'], '.3f'):>7}"
        f" {fmt(m['jacc_rms'], '.1f'):>9}"
    )

print(sep)
print()
print("  Column descriptions:")
print("    vx_mean   — mean forward velocity [m/s]  (target: +0.40)")
print("    vx_std    — forward velocity std [m/s]   (lower = more consistent)")
print("    tilt_mean — mean |grav_xy|²               (lower = more upright)")
print("    tilt_max  — peak |grav_xy|²               (lower = graceful transitions)")
print("    h_mean    — mean body height [m]          (target: 0.42)")
print("    CoT       — Cost of Transport             (lower = more efficient; biol. ~0.2)")
print("    jacc_RMS  — joint accel RMS [rad/s²]      (lower = smoother trajectories)")
print(sep)
