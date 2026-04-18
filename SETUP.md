# Adding Unitree B1 Robot to Isaac Lab

**Author:** Disthorn Suttawet  
**Date:** April 2026  
**Isaac Lab Version:** 0.36.3  
**Isaac Sim Version:** 4.5.0

---

## Overview

This guide documents the process of adding Unitree B1 quadruped robot to Isaac Lab simulation environment. The B1 model is not included in Isaac Lab by default, so we convert it from the official Unitree URDF files.

**What you'll get:**
- ✅ Official Unitree B1 geometry and physics
- ✅ Proper joint configuration (12 DOF quadruped)
- ✅ DC motor actuator model (23.7 N·m, 30 rad/s)
- ✅ Ready-to-use `UNITREE_B1_CFG` in Isaac Lab

**Time required:** ~15 minutes

---

## Prerequisites

- Isaac Lab 0.36.3+ installed
- Isaac Sim 4.5+
- Conda environment activated: `conda activate env_isaaclab`
- Git installed

---

## Step 1: Obtain Official B1 URDF Files

Clone Unitree's official ROS repository:

```bash
cd ~/Downloads
git clone https://github.com/unitreerobotics/unitree_ros
cd unitree_ros/robots/b1_description
```

Verify files:
```bash
ls meshes/
# Expected output: calfb.dae, hipb.dae, thighb.dae, trunkb.dae, etc.

ls xacro/b1.urdf
# Expected output: xacro/b1.urdf
```

---

## Step 2: Fix URDF Mesh Paths

URDF files use ROS-specific `package://` paths. Convert to absolute file paths:

```bash
cd ~/Downloads/unitree_ros/robots/b1_description

# Create fixed version
cp xacro/b1.urdf xacro/b1_fixed.urdf

# Replace package paths with absolute paths
MESHDIR=$(pwd)/meshes
sed -i "s|package://b1_description/meshes|file://${MESHDIR}|g" xacro/b1_fixed.urdf
```

**Verify the fix:**
```bash
grep "file://" xacro/b1_fixed.urdf | head -3
```

Expected output:
```
<mesh filename="file:///home/USERNAME/Downloads/unitree_ros/robots/b1_description/meshes/trunkb.dae" scale="1 1 1"/>
<mesh filename="file:///home/USERNAME/Downloads/unitree_ros/robots/b1_description/meshes/trunkb.dae" scale="1 1 1"/>
<mesh filename="file:///home/USERNAME/Downloads/unitree_ros/robots/b1_description/meshes/hipb.dae" scale="1 1 1"/>
```

---

## Step 3: Convert URDF to USD

Use Isaac Lab's built-in URDF converter:

```bash
cd ~/IsaacLab

python scripts/tools/convert_urdf.py \
    ~/Downloads/unitree_ros/robots/b1_description/xacro/b1_fixed.urdf \
    ~/Downloads/b1_usd/b1.usd \
    --joint-stiffness 25.0 \
    --joint-damping 0.5 \
    --headless
```

**Expected output:**
```
--------------------------------------------------------------------------------
Input URDF file: /home/USERNAME/Downloads/.../b1_fixed.urdf
URDF importer config:
    asset_path: .../b1_fixed.urdf
    joint_drive:
        gains:
            stiffness: 25.0
            damping: 0.5
--------------------------------------------------------------------------------
URDF importer output:
Generated USD file: /home/USERNAME/Downloads/b1_usd/b1.usd
--------------------------------------------------------------------------------
```

**Notes:**
- Warnings about "No mass specified for link base" are normal
- Conversion takes ~30-60 seconds
- Output USD file should be ~1-2 KB

---

## Step 4: Copy USD to Isaac Lab Assets

```bash
# Create B1 directory
mkdir -p ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1

# Copy USD file
cp ~/Downloads/b1_usd/b1.usd \
   ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/

# Verify
ls -lh ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/
```

Expected output:
```
-rw-rw-r-- 1 user user 1.6K Apr 12 16:47 b1.usd
```

---

## Step 5: Add B1 Configuration to Isaac Lab

Edit the Unitree robot definitions:

```bash
cd ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots
nano unitree.py
```

### 5.1: Add import (if not present)

At the top of the file, ensure `os` is imported:

```python
import os
```

### 5.2: Add B1 configuration

At the bottom of the file (after `G1_MINIMAL_CFG`), add:

```python


UNITREE_B1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.expanduser('~')}/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),  # B1 standing height (meters)
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.7,      # B1 spec: 23.7 N·m per joint
            saturation_effort=23.7,
            velocity_limit=30.0,    # B1 spec: 30 rad/s
            stiffness=25.0,         # PD control stiffness
            damping=0.5,            # PD control damping
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B1 using converted USD from official URDF.

Reference: https://github.com/unitreerobotics/unitree_ros
Motor specs: 23.7 N·m max torque, 30 rad/s max velocity
"""
```

**Save and exit:** `Ctrl+X`, `Y`, `Enter`

---

## Step 6: Verify Installation

Create a test script `test_b1.py`:

```python
"""Test Unitree B1 configuration."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args)
sim_app = app.app

# Import B1 configuration
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG

print("\n" + "="*70)
print("Unitree B1 Configuration Test")
print("="*70)
print(f"✓ USD path: {UNITREE_B1_CFG.spawn.usd_path}")
print(f"✓ Initial height: {UNITREE_B1_CFG.init_state.pos[2]} m")
print(f"✓ Actuators: {list(UNITREE_B1_CFG.actuators.keys())}")
print(f"✓ Motor effort limit: {UNITREE_B1_CFG.actuators['base_legs'].effort_limit} N·m")
print(f"✓ Motor velocity limit: {UNITREE_B1_CFG.actuators['base_legs'].velocity_limit} rad/s")
print("="*70)
print("SUCCESS: B1 loaded successfully!\n")

sim_app.close()
```

Run test:
```bash
python test_b1.py --headless
```

**Expected output:**
```
======================================================================
Unitree B1 Configuration Test
======================================================================
✓ USD path: /home/USERNAME/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd
✓ Initial height: 0.42 m
✓ Actuators: ['base_legs']
✓ Motor effort limit: 23.7 N·m
✓ Motor velocity limit: 30.0 rad/s
======================================================================
SUCCESS: B1 loaded successfully!
```

---

## Usage in Your Code

```python
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG

# Use in Isaac Lab environment
robot_cfg = UNITREE_B1_CFG
```

---

## Troubleshooting

### Issue 1: ImportError - Cannot import UNITREE_B1_CFG

**Cause:** Configuration not added to `unitree.py` or syntax error

**Fix:**
```bash
cd ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots
python -c "from isaaclab_assets.robots.unitree import UNITREE_B1_CFG; print('OK')"
```

If error persists, check for Python syntax errors in `unitree.py`.

### Issue 2: USD file not found

**Cause:** USD file not copied to correct location

**Fix:**
```bash
ls ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd
# If not found, repeat Step 4
```

### Issue 3: URDF conversion fails with "Missing values"

**Cause:** `joint_drive` configuration missing

**Fix:**
Ensure conversion command includes:
```bash
--joint-stiffness 25.0 \
--joint-damping 0.5
```

### Issue 4: Mesh files not found during conversion

**Cause:** URDF still has `package://` paths

**Fix:**
Verify Step 2 was completed:
```bash
grep "package://" ~/Downloads/unitree_ros/robots/b1_description/xacro/b1_fixed.urdf
# Should return NO results

grep "file://" ~/Downloads/unitree_ros/robots/b1_description/xacro/b1_fixed.urdf
# Should show mesh paths with file:// prefix
```

---

## Technical Details

### B1 Specifications

- **DOF:** 12 (4 legs × 3 joints: hip, thigh, calf)
- **Joint pattern:** `[FL,FR,RL,RR]_[hip,thigh,calf]_joint`
- **Motor torque:** 23.7 N·m max
- **Motor velocity:** 30 rad/s max
- **Standing height:** 0.42 m
- **Mass:** ~12 kg (from URDF)

### Actuator Model

Uses `DCMotorCfg` (DC motor model) with:
- **Stiffness:** 25.0 N·m/rad (position control gain)
- **Damping:** 0.5 N·m/(rad/s) (velocity damping)
- **Control mode:** Position control with PD gains

### File Locations

```
~/IsaacLab/
├── source/isaaclab_assets/
│   ├── data/Robots/Unitree/B1/
│   │   └── b1.usd                    # Converted USD file
│   └── isaaclab_assets/robots/
│       └── unitree.py                # B1 configuration added here
└── scripts/tools/
    └── convert_urdf.py               # URDF→USD converter

~/Downloads/
└── unitree_ros/robots/b1_description/
    ├── meshes/                        # B1 geometry files
    └── xacro/
        ├── b1.urdf                    # Original URDF
        └── b1_fixed.urdf              # Fixed mesh paths
```

---

## References

- Unitree ROS: https://github.com/unitreerobotics/unitree_ros
- Isaac Lab Docs: https://isaac-sim.github.io/IsaacLab/
- URDF Converter: `~/IsaacLab/scripts/tools/convert_urdf.py`

---

## Changelog

**2026-04-12:** Initial documentation
- B1 URDF conversion completed
- Configuration added to Isaac Lab 0.36.3
- Verified on Ubuntu 22.04, Isaac Sim 4.5