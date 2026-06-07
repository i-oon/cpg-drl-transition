#!/usr/bin/env python3
"""
Experiment 1: Joint Number Mapping & Limit Verification

Goal: confirm which joint index maps to which physical leg/joint,
and that joint limits match the SDK constants.

Joint order (from SDK quadruped.h):
  0: FR hip   1: FR thigh   2: FR calf
  3: FL hip   4: FL thigh   5: FL calf
  6: RR hip   7: RR thigh   8: RR calf
  9: RL hip  10: RL thigh  11: RL calf

Joint limits (from SDK b1_const.h):
  Hip:   min=-0.8,  max= 0.8  rad
  Thigh: min=-1.0,  max= 3.5  rad
  Calf:  min=-2.66, max=-0.55 rad

Run:
  Terminal 1: ros2 launch b1_interface b1interface.launch.py
  Terminal 2: python3 test_joint_mapping.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import time

# Joint index map (from SDK)
JOINT_NAMES = [
    "FR_hip",   "FR_thigh",   "FR_calf",   # 0  1  2
    "FL_hip",   "FL_thigh",   "FL_calf",   # 3  4  5
    "RR_hip",   "RR_thigh",   "RR_calf",   # 6  7  8
    "RL_hip",   "RL_thigh",   "RL_calf",   # 9 10 11
]

# Joint limits (from b1_const.h)
LIMITS = {
    "hip":   (-0.8,  0.8),
    "thigh": (-1.0,  3.5),
    "calf":  (-2.66, -0.55),
}

HOME = [0.0] * 12  # all joints at zero


class JointTester(Node):
    def __init__(self):
        super().__init__("joint_tester")
        self.pub = self.create_publisher(Float32MultiArray, "/B1/joint_target", 10)
        self.conn = self.create_publisher(Bool, "/B1/connection", 10)
        self.sub = self.create_subscription(
            Float32MultiArray, "/B1/joint_position", self.joint_cb, 10
        )
        self.current_pos = [0.0] * 12

    def joint_cb(self, msg):
        self.current_pos = list(msg.data)

    def enable(self):
        msg = Bool()
        msg.data = True
        self.conn.publish(msg)

    def send(self, targets):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in targets]
        self.pub.publish(msg)

    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def go_home(self):
        print("\n>> Going HOME (all zeros)...")
        self.enable()  # re-arm: the command-heartbeat watchdog disables control during idle pauses
        for _ in range(50):
            self.send(HOME)
            self.wait(0.02)
        self.wait(2.0)

    def wiggle_joint(self, idx, delta=0.15, label=""):
        """Move one joint by +delta then back to zero."""
        print(f"\n>> Wiggling joint {idx} ({label})  delta={delta:.2f} rad")
        self.enable()  # re-arm connection before moving (see go_home)
        cmd = HOME[:]

        # Move positive
        cmd[idx] = delta
        for _ in range(50):
            self.send(cmd)
            self.wait(0.02)
        self.wait(1.0)
        pos_after = self.current_pos[idx]
        print(f"   Sent +{delta:.2f} rad  →  actual: {pos_after:.4f} rad")

        # Return home
        cmd[idx] = 0.0
        for _ in range(50):
            self.send(cmd)
            self.wait(0.02)
        self.wait(1.0)


def main():
    rclpy.init()
    node = JointTester()

    print("=" * 50)
    print("EXPERIMENT 1: Joint Mapping & Limit Verification")
    print("=" * 50)
    print()
    print("Joint map:")
    for i, name in enumerate(JOINT_NAMES):
        j_type = name.split("_")[1]
        lo, hi = LIMITS[j_type]
        print(f"  [{i:2d}] {name:12s}  limits: [{lo:.2f}, {hi:.2f}] rad")

    print()
    input("SAFETY CHECK: Is robot on flat surface with clear space? (Enter to continue)")
    print("NOTE: actuation also requires `ros2 param set /b1_interface motor_output true`")
    input("WARNING: Motors will move! Keep clear. (Enter to continue)")

    # Enable connection
    print("\n>> Enabling connection...")
    node.enable()
    node.wait(1.0)

    # Go to home position first
    node.go_home()

    # Test each joint one by one with a small safe wiggle
    # Use small delta to stay well within limits
    SAFE_DELTAS = {
        "hip":   0.2,   # safe small hip movement
        "thigh": 0.2,   # safe small thigh movement
        "calf":  -0.2,  # negative direction (calf range is negative)
    }

    print("\n" + "=" * 50)
    print("Starting joint wiggle test (one at a time)...")
    print("Watch which leg/joint moves on the robot!")
    print("=" * 50)

    for i, name in enumerate(JOINT_NAMES):
        j_type = name.split("_")[1]
        delta = SAFE_DELTAS[j_type]
        node.go_home()
        input(f"\nPress Enter to wiggle joint {i} ({name})...")
        node.wiggle_joint(i, delta=delta, label=name)

    # Return home at end
    node.go_home()
    print("\n>> Done! All joints tested.")
    print("\nCheck: did each joint move match the expected leg/joint?")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
