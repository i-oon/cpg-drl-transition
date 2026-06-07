#!/usr/bin/env python3
"""Safe waypoint interpolator: move from current pose to a target pose over N seconds.

Publishes /B1/joint_target at 50 Hz, linearly interpolating from the current
measured position to the target. Aborts if any joint diverges beyond
`abort_threshold` rad from the commanded trajectory (tracking failure).

Usage (CLI):
    ros2 run b1_interface go_to_pose.py \
        --ros-args \
        -p target:="[0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0]" \
        -p duration:=5.0

Usage (home pose shortcut):
    ros2 run b1_interface go_to_pose.py  --ros-args -p preset:=home

Params:
    target          float[12]  target joint positions in control space
                               (FR/FL/RR/RL × hip/thigh/calf)
    duration        float      seconds to complete the move (default 5.0)
    rate            float      publish rate Hz (default 50.0)
    abort_threshold float      max allowed tracking error per joint rad (default 0.5)
    preset          string     "home" fills target with home pose; overrides target param

SDK / control-space joint order:
    [FR_hip, FR_thigh, FR_calf,
     FL_hip, FL_thigh, FL_calf,
     RR_hip, RR_thigh, RR_calf,
     RL_hip, RL_thigh, RL_calf]

Home pose (control space zeros → standing):
    hip=0.0, thigh=0.0, calf=0.0 for all legs
    (b1_interface adds the offsets: hip±0.2, thigh+0.85, calf-1.56)
"""

import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


HOME_POSE = [0.0, 0.0, 0.0] * 4   # control space: all zeros = standing home

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

# Control-space joint limits (for clamping target)
LIMITS = [
    (-0.8,  0.8),   # hip
    (-1.0,  3.5),   # thigh
    (-2.66,-0.55),  # calf
] * 4


class GoToPose(Node):
    def __init__(self):
        super().__init__("go_to_pose")

        duration  = float(self.declare_parameter("duration",  5.0).value)
        rate_hz   = float(self.declare_parameter("rate",     50.0).value)
        threshold = float(self.declare_parameter("abort_threshold", 0.5).value)
        preset    = str(self.declare_parameter("preset", "").value)

        # Target
        if preset == "home":
            target = list(HOME_POSE)
            self.get_logger().info("Preset: home pose")
        else:
            raw = self.declare_parameter(
                "target", [0.0] * 12).get_parameter_value().double_array_value
            target = list(raw)

        if len(target) != 12:
            self.get_logger().error(f"target must have 12 values, got {len(target)}")
            sys.exit(1)

        # Clamp target to limits
        for i in range(12):
            lo, hi = LIMITS[i]
            clamped = max(lo, min(hi, target[i]))
            if abs(clamped - target[i]) > 1e-6:
                self.get_logger().warning(
                    f"{JOINT_NAMES[i]} target {target[i]:.3f} clamped to {clamped:.3f}")
            target[i] = clamped

        self._target    = target
        self._duration  = duration
        self._rate_hz   = rate_hz
        self._threshold = threshold
        self._dt        = 1.0 / rate_hz

        self._current   = None   # filled on first /B1/joint_position message
        self._start     = None   # filled when motion begins
        self._done      = False
        self._aborted   = False

        self._pub = self.create_publisher(Float32MultiArray, "/B1/joint_target", 1)
        self._sub = self.create_subscription(
            Float32MultiArray, "/B1/joint_position", self._on_position, 1)

        self.get_logger().info(
            f"go_to_pose: waiting for /B1/joint_position before starting "
            f"({duration:.1f}s move)…")

    def _on_position(self, msg: Float32MultiArray):
        if len(msg.data) < 12:
            return
        self._current = list(msg.data[:12])

    def run(self):
        """Blocking execution loop. Returns True on success, False on abort."""
        rate = self.create_rate(self._rate_hz)

        # Wait for first position reading
        deadline = time.time() + 5.0
        while rclpy.ok() and self._current is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() > deadline:
                self.get_logger().error(
                    "Timeout waiting for /B1/joint_position — is b1_interface running?")
                return False

        start_pose = list(self._current)
        self._start = time.time()

        self.get_logger().info(
            f"Starting move: {self._duration:.1f}s  abort_threshold={self._threshold:.2f}rad")
        self.get_logger().info(
            f"  from: {[f'{v:.3f}' for v in start_pose]}")
        self.get_logger().info(
            f"  to:   {[f'{v:.3f}' for v in self._target]}")

        while rclpy.ok() and not self._done and not self._aborted:
            rclpy.spin_once(self, timeout_sec=0.0)

            elapsed = time.time() - self._start
            alpha   = min(elapsed / self._duration, 1.0)

            # Interpolated command
            cmd = [start_pose[i] + alpha * (self._target[i] - start_pose[i])
                   for i in range(12)]

            # Safety check: is the robot tracking?
            if self._current is not None:
                for i in range(12):
                    err = abs(self._current[i] - cmd[i])
                    if err > self._threshold:
                        self.get_logger().error(
                            f"ABORT: {JOINT_NAMES[i]} tracking error {err:.3f} rad "
                            f"> threshold {self._threshold:.3f} rad at t={elapsed:.2f}s")
                        self._aborted = True
                        break

            if not self._aborted:
                msg = Float32MultiArray()
                msg.data = [float(v) for v in cmd]
                self._pub.publish(msg)

                # Progress log every second
                if int(elapsed) != int(elapsed - self._dt):
                    pct = int(alpha * 100)
                    self.get_logger().info(f"  progress: {pct}% ({elapsed:.1f}/{self._duration:.1f}s)")

            if alpha >= 1.0:
                self._done = True

            try:
                rate.sleep()
            except Exception:
                break

        if self._done and not self._aborted:
            # Hold target exactly on completion
            msg = Float32MultiArray()
            msg.data = [float(v) for v in self._target]
            self._pub.publish(msg)
            self.get_logger().info("Move complete.")
            return True
        else:
            self.get_logger().warning("Move aborted — robot did not track commanded trajectory.")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = GoToPose()
    success = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
