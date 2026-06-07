#!/usr/bin/env python3
"""Bridge: Gazebo joint_sine -> real B1 robot via b1_interface.

Subscribes to the sine commands published by joint_sine.py
(/joint_position_controller/commands, Float64MultiArray, robot space)
and republishes them as /B1/joint_target (Float32MultiArray, control space)
for b1_interface to forward to the real robot.

Delta-based conversion (safe for partial-joint targets):
    joint_sine.py sends DEFAULT_ROBOT for non-oscillating joints and
    DEFAULT_ROBOT[i] + amp*sin(t) for oscillating ones.

    Instead of converting directly (which would move all non-target joints
    to DEFAULT_ROBOT simultaneously), this bridge computes the DELTA from
    DEFAULT_ROBOT and applies it to the CURRENT measured position:

        delta[i]  = (sine_cmd[i] - DEFAULT_ROBOT[i]) / direction[i]
        target[i] = current_measured[i] + delta[i]

    Non-oscillating joints: delta=0 → target = current measured → no movement.
    Oscillating joints:     delta = amp*sin(t)/direction → oscillates in place.

Usage:
    ros2 launch b1_interface sine_replay_real.launch.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Bool

# SDK joint order: (FR, FL, RR, RL) x (hip, thigh, calf)
OFFSET = [
    -0.2,  0.85, -1.56,   # FR hip, thigh, calf
     0.2,  0.85, -1.56,   # FL
    -0.2,  0.85, -1.56,   # RR
     0.2,  0.85, -1.56,   # RL
]
DIRECTION = [
     1.0,  1.0,  1.0,    # FR
    -1.0,  1.0,  1.0,    # FL  (hip sign-flipped)
     1.0,  1.0,  1.0,    # RR
    -1.0,  1.0,  1.0,    # RL  (hip sign-flipped)
]

# Must match joint_sine.py DEFAULT_POSITION (robot space, measured standing pose)
DEFAULT_ROBOT = [
    -0.03, 1.08, -1.94,   # FR hip, thigh, calf
     0.03, 1.08, -1.94,   # FL
    -0.03, 1.08, -1.94,   # RR
     0.03, 1.08, -1.94,   # RL
]


class SineToRealBridge(Node):
    def __init__(self):
        super().__init__('sine_to_real_bridge')

        self.declare_parameter('publish_rate', 50.0)
        rate = self.get_parameter('publish_rate').value

        # Latest sine command from joint_sine.py (robot space)
        self._latest = list(DEFAULT_ROBOT)
        self._have_cmd = False

        # Current measured joint positions from robot (control space)
        self._current = [0.0] * 12
        self._have_state = False

        # Subscribe to sine output and robot feedback
        self.create_subscription(
            Float64MultiArray,
            '/joint_position_controller/commands',
            self._on_sine_cmd, 10)
        self.create_subscription(
            Float32MultiArray,
            '/B1/joint_position',
            self._on_joint_pos, 1)

        # Publish to b1_interface
        self._pub_target = self.create_publisher(Float32MultiArray, '/B1/joint_target', 1)
        self._pub_conn   = self.create_publisher(Bool, '/B1/connection', 1)

        self._conn_sent = False
        self._conn_retry_ticks = 0   # retry /B1/connection for 3 s after start
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'sine_to_real_bridge ready at {rate:.0f} Hz — '
            f'delta mode: non-target joints hold current position')

    def _on_sine_cmd(self, msg: Float64MultiArray):
        if len(msg.data) >= 12:
            self._latest = list(msg.data[:12])
            self._have_cmd = True

    def _on_joint_pos(self, msg: Float32MultiArray):
        if len(msg.data) >= 12:
            self._current = list(msg.data[:12])
            self._have_state = True

    def _compute_targets(self, robot_cmd):
        """Delta-based conversion: apply only the deviation from DEFAULT_ROBOT.

        Non-oscillating joints (robot_cmd == DEFAULT_ROBOT) → delta = 0 → no movement.
        Oscillating joints (robot_cmd = DEFAULT_ROBOT + amp*sin) → delta applied.
        """
        targets = []
        for i in range(12):
            delta_robot   = robot_cmd[i] - DEFAULT_ROBOT[i]
            delta_control = delta_robot / DIRECTION[i]

            if self._have_state:
                # Anchor to current measured position + oscillation delta
                targets.append(self._current[i] + delta_control)
            else:
                # Fallback before first state arrives: direct conversion
                targets.append((robot_cmd[i] - OFFSET[i]) / DIRECTION[i])

        return targets

    def _tick(self):
        # Retry /B1/connection for the first 3 s (rate * 3 ticks) so the signal
        # is not lost if b1_interface hasn't finished initialising yet.
        if self._conn_retry_ticks < int(self.get_parameter('publish_rate').value * 3):
            self._conn_retry_ticks += 1
            conn = Bool()
            conn.data = True
            self._pub_conn.publish(conn)
            if not self._conn_sent:
                self._conn_sent = True
                self.get_logger().info('Sending /B1/connection true (retrying for 3 s)')

        if not self._have_cmd:
            return

        if not self._have_state:
            self.get_logger().warning(
                'Waiting for /B1/joint_position — not publishing yet')
            return

        targets = self._compute_targets(self._latest)

        msg = Float32MultiArray()
        msg.data = [float(v) for v in targets]
        self._pub_target.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SineToRealBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
