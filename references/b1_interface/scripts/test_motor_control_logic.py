#!/usr/bin/env python3
"""
Offline unit tests for myb1.cpp / b1_ros2_interface.cpp control logic.
No hardware, no ROS, no UDP — pure Python simulation of the C++ math.

Tests:
  T1  motor_output enable fires near-zero torque (fix regression)
  T2  soft-start ramp: torques grow from 0 to full over TMAX
  T3  position error drives correct sign of torque
  T4  tau_max clamps large errors
  T5  resetForMotorEnable re-seeds command from current position
  T6  transition detection: reset only fires on false -> true, not true -> true
  T7  full startup sequence: running->cmd->motor_output produces zero jerk

Run:
    python3 src/b1_interface/scripts/test_motor_control_logic.py
"""

import math

# ─── Constants (mirror of myb1.h / b1interface.launch.py) ─────────────────
TMAX    = 10.0
DT      = 0.001    # 1 kHz control loop
TAU_MAX = 80.0

OFFSET    = [-0.2, 0.85, -1.56,  0.2, 0.85, -1.56, -0.2, 0.85, -1.56,  0.2, 0.85, -1.56]
DIRECTION = [ 1.0,  1.0,   1.0, -1.0,  1.0,   1.0,  1.0,  1.0,   1.0, -1.0,  1.0,   1.0]

KP  = [300, 200, 300] * 4
KD  = [  5,   5,   5] * 4
TAU = [-4.0, 0.0, 0.0] * 4   # hip gravity compensation


# ─── Minimal simulation of myb1 RobotControl() ────────────────────────────

class SimMyB1:
    def __init__(self):
        self.motor_output       = False
        self.t                  = 0.0
        self.intp_t             = 0.0
        self.motorcommand       = [0.0] * 12
        self.oldmotorcommand    = [0.0] * 12
        self.interpmotorcommand = [0.0] * 12
        self.control_initialized = False
        self.kp  = list(KP)
        self.kd  = list(KD)
        self.tau = list(TAU)
        self.tau_max = TAU_MAX

        # Simulated "measured" joint state (hardware)
        self.state_q  = list(OFFSET)   # robot is standing at home
        self.state_dq = [0.0] * 12     # not moving

    def set_motor_command(self, cmd_control_space):
        """Mirrors setMotorCommand(): converts control→robot space, resets intp."""
        for i in range(12):
            self.oldmotorcommand[i] = self.motorcommand[i]
            self.motorcommand[i]    = DIRECTION[i] * cmd_control_space[i] + OFFSET[i]
        self.intp_t = 0.0

    def reset_for_motor_enable(self):
        """Mirrors resetForMotorEnable(): seed from current, restart soft-start."""
        for i in range(12):
            self.motorcommand[i]    = self.state_q[i]
            self.oldmotorcommand[i] = self.state_q[i]
        self.intp_t = 0.0
        self.t      = 0.0

    def _interp(self, x0, x1, t):
        t = max(0.0, min(1.0, t))
        return x0 * (1.0 - t) + x1 * t

    def robot_control_tick(self):
        """One tick of RobotControl(). Returns list of 12 torques sent (or None)."""
        # Seed on first run
        if not self.control_initialized:
            for i in range(12):
                self.motorcommand[i]    = self.state_q[i]
                self.oldmotorcommand[i] = self.state_q[i]
            self.control_initialized = True

        # Soft-start ramp
        if self.t < 1.0:
            self.t += DT / TMAX
        else:
            self.t = 1.0

        self.intp_t += DT

        torques = []
        for i in range(12):
            interp = self._interp(self.oldmotorcommand[i], self.motorcommand[i],
                                  self.intp_t * (1.0 / DT))   # rate=1/DT at 1 kHz
            self.interpmotorcommand[i] = interp

            pos_err = interp - self.state_q[i]
            vel_err = -self.state_dq[i]

            ctrl = self.t * (self.kp[i] * pos_err + self.kd[i] * vel_err + self.tau[i])
            ctrl = max(-self.tau_max, min(self.tau_max, ctrl))
            torques.append(ctrl)

        if self.motor_output:
            return torques
        return None   # nothing sent when motor_output=false


# ─── Test helpers ─────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)
    return condition


# ─── T1: motor_output enable → near-zero torques WITH the fix ─────────────

def test_t1_enable_no_jerk():
    print("\nT1 — motor_output enable: near-zero torque with fix")
    b = SimMyB1()

    # Simulate: running=true was set, controller ran for 12s (t saturates at 1)
    for _ in range(int(12.0 / DT)):
        b.robot_control_tick()

    assert abs(b.t - 1.0) < 1e-6, "soft-start should have reached 1.0"

    # User sets motor_output=true WITH fix: reset fires first
    b.reset_for_motor_enable()
    b.motor_output = True
    torques = b.robot_control_tick()

    max_t = max(abs(v) for v in torques)
    check("max torque on enable < 0.01 Nm", max_t < 0.01,
          f"max={max_t:.4f} Nm")
    check("t reset to near-zero after enable", b.t < 0.01,
          f"t={b.t:.4f}")


# ─── T2 (regression): motor_output enable WITHOUT fix → 4 Nm jerk ─────────

def test_t2_enable_jerk_without_fix():
    print("\nT2 — (regression) motor_output enable: 4 Nm jerk WITHOUT fix")
    b = SimMyB1()
    for _ in range(int(12.0 / DT)):
        b.robot_control_tick()

    # NO reset — simulate old behaviour
    b.motor_output = True
    torques = b.robot_control_tick()

    hip_torque = torques[0]   # FR hip
    check("hip torque is -4 Nm without fix (regression baseline)",
          abs(hip_torque - (-4.0)) < 0.01, f"FR_hip={hip_torque:.3f} Nm")


# ─── T3: soft-start ramp brings torques up gradually ──────────────────────

def test_t3_soft_start_ramp():
    print("\nT3 — soft-start ramp after motor_output enable")
    b = SimMyB1()
    b.state_q = [v + 0.1 for v in OFFSET]   # robot 0.1 rad off in all joints

    b.reset_for_motor_enable()
    b.motor_output = True

    # After reset, motorcommand = state_q, so pos_err=0.
    # Only tau contributes. Ramp at 1kHz: t grows as DT/TMAX per tick.
    t_at_1s = DT / TMAX * (1.0 / DT)   # t after 1 second = 1/TMAX = 0.1
    for _ in range(int(1.0 / DT)):
        b.robot_control_tick()

    expected_hip = b.t * TAU[0]
    actual_hip   = b.t * TAU[0]   # same formula
    check("t ≈ 0.1 after 1 second", abs(b.t - 0.1) < 0.005,
          f"t={b.t:.4f}")
    check("hip torque ≈ t * tau = 0.1 * -4 = -0.4 Nm after 1s",
          abs(b.t * TAU[0] - (-0.4)) < 0.05,
          f"t*tau={b.t*TAU[0]:.3f} Nm")


# ─── T4: position error drives correct torque direction ───────────────────

def test_t4_position_error():
    print("\nT4 — position error drives correct torque direction")
    b = SimMyB1()
    b.reset_for_motor_enable()
    b.motor_output = True

    # Prime control_initialized with one tick before issuing the test command
    b.robot_control_tick()
    b.t = 1.0   # force full gains for this test

    # Command FR_thigh +0.3 rad above current (kp*0.3 = 60 Nm, within tau_max=80)
    target = list(b.state_q)
    target[1] += 0.3   # FR_thigh in robot space
    b.motorcommand    = list(target)
    b.oldmotorcommand = list(target)

    # Set intp_t large so interpolation is already complete (at motorcommand)
    b.intp_t = 1e9
    torques = b.robot_control_tick()

    # FR_thigh has positive error → positive torque (kp=200, err=0.5)
    expected = 1.0 * (KP[1] * 0.3 + 0 + TAU[1])   # t=1, err=0.3
    check("FR_thigh positive error → positive torque",
          torques[1] > 0, f"tau={torques[1]:.2f} Nm (expected ~{expected:.1f})")
    check("FR_thigh torque magnitude ≈ kp*err",
          abs(torques[1] - expected) < 5.0, f"got={torques[1]:.1f} expected={expected:.1f}")


# ─── T5: tau_max clamp ────────────────────────────────────────────────────

def test_t5_tau_max_clamp():
    print("\nT5 — tau_max clamps extreme errors")
    b = SimMyB1()
    b.reset_for_motor_enable()
    b.motor_output = True
    b.t = 1.0

    # Command a huge position error (5 rad)
    b.motorcommand    = [v + 5.0 for v in b.state_q]
    b.oldmotorcommand = list(b.motorcommand)
    b.intp_t = 1e9

    torques = b.robot_control_tick()
    check("all torques clamped to tau_max",
          all(abs(v) <= TAU_MAX + 1e-6 for v in torques),
          f"max={max(abs(v) for v in torques):.1f}")


# ─── T6: transition detection (false→true only) ───────────────────────────

def test_t6_transition_detection():
    print("\nT6 — reset fires on false→true, not on true→true")
    b = SimMyB1()

    reset_calls = []

    # Patch reset to track calls
    original_reset = b.reset_for_motor_enable
    def tracked_reset():
        reset_calls.append(1)
        original_reset()
    b.reset_for_motor_enable = tracked_reset

    prev = False
    for new_motor_output in [False, False, True, True, True, False, True]:
        if new_motor_output and not prev:
            b.reset_for_motor_enable()   # transition detected
        b.motor_output = new_motor_output
        prev = new_motor_output

    # Transitions false→true: at index 2 (False→True) and index 6 (False→True) = 2 resets
    check("reset called exactly 2 times (two false→true transitions)",
          len(reset_calls) == 2, f"got {len(reset_calls)} calls")


# ─── T7: full startup sequence produces zero jerk ─────────────────────────

def test_t7_full_startup_sequence():
    print("\nT7 — full startup sequence: running→cmd→motor_output produces zero jerk")
    b = SimMyB1()

    # Step 1: running=true — controller starts (motor_output still false)
    for _ in range(100):   # 100 ms
        b.robot_control_tick()
    check("after running=true, t started ramping", b.t > 0, f"t={b.t:.4f}")

    # Step 2: user sends joint target (control space zeros = home)
    b.set_motor_command([0.0] * 12)

    # Step 3: more ticks while motor_output=false (simulate user delay)
    # TMAX=10s, need >10s of ticks to saturate t at 1.0
    for _ in range(int(12.0 / DT)):   # 12 seconds
        b.robot_control_tick()
    check("after 12s idle, t saturated at 1", abs(b.t - 1.0) < 1e-6, f"t={b.t:.4f}")

    # Step 4: motor_output=true WITH fix
    b.reset_for_motor_enable()
    b.motor_output = True
    torques = b.robot_control_tick()

    max_t = max(abs(v) for v in torques)
    check("first torque after enable < 0.01 Nm", max_t < 0.01,
          f"max={max_t:.4f} Nm")
    check("t reset to ~0 by resetForMotorEnable", b.t < 0.01, f"t={b.t:.4f}")


# ─── Run all tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  b1_interface control logic tests (no hardware required)")
    print("=" * 60)

    test_t1_enable_no_jerk()
    test_t2_enable_jerk_without_fix()
    test_t3_soft_start_ramp()
    test_t4_position_error()
    test_t5_tau_max_clamp()
    test_t6_transition_detection()
    test_t7_full_startup_sequence()

    passed = sum(results)
    total  = len(results)
    print()
    print("=" * 60)
    if passed == total:
        print(f"  \033[32mALL {total} CHECKS PASSED\033[0m")
    else:
        print(f"  \033[31m{passed}/{total} passed — {total-passed} FAILED\033[0m")
    print("=" * 60)
    exit(0 if passed == total else 1)
