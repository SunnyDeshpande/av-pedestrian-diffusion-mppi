"""PACMod bridge node — converts MPPI control output to PACMod commands.

Subscribes to /mppi/control_output [steer_rad, accel_m_s2] and publishes
all PACMod actuator commands. Supports two modes via the 'mode' parameter:
  - 'pid':    PID speed tracking (integrates accel into v_cmd, PID on error)
  - 'linear': Direct linear mapping from acceleration to throttle %

Usage:
  ros2 run mppi_controller pacmod_bridge_node --ros-args -p mode:=pid
  ros2 run mppi_controller pacmod_bridge_node --ros-args -p mode:=linear
"""
import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32MultiArray
from pacmod2_msgs.msg import (
    GlobalCmd, PositionWithSpeed, SystemCmdFloat, SystemCmdInt,
    VehicleSpeedRpt,
)


class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp, self.ki, self.kd, self.wg = kp, ki, kd, wg
        self.iterm = 0.0
        self.last_e = 0.0
        self.last_t = None

    def reset(self):
        self.iterm = 0.0
        self.last_e = 0.0
        self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt, de = 0.0, 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0.0 else 0.0
        self.iterm += e * dt
        if self.wg is not None:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)
        self.last_e = e
        self.last_t = t
        return self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter:
    def __init__(self, cutoff, fs):
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * max(cutoff, 1e-6) / max(fs, 1e-6))
        self._y = None

    def get_data(self, x):
        self._y = x if self._y is None else (self.alpha * x + (1.0 - self.alpha) * self._y)
        return self._y


def front2steer(f_angle_deg):
    """Front-wheel angle (deg) -> steering-wheel angle (deg), GEM calibration."""
    a = max(min(f_angle_deg, 35.0), -35.0)
    mag = abs(a)
    sw = -0.1084 * mag * mag + 21.775 * mag
    sw = sw if a >= 0 else -sw
    return max(min(sw, 450.0), -450.0)


class PACModBridgeNode(Node):
    def __init__(self):
        super().__init__('pacmod_bridge_node')

        # --- Parameters ---
        self.declare_parameter('mode', 'pid')
        self.declare_parameter('desired_speed', 2.0)
        self.declare_parameter('max_throttle', 0.5)
        self.declare_parameter('linear_gain', 0.3)
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('require_pacmod_enable', True)
        self.declare_parameter('steering_speed_limit', 4.0)

        self.declare_parameter('pid/kp', 0.6)
        self.declare_parameter('pid/ki', 0.0)
        self.declare_parameter('pid/kd', 0.1)
        self.declare_parameter('pid/wg', 10.0)

        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30.0)

        p = lambda n: self.get_parameter(n).value
        self.mode = str(p('mode'))
        self.desired_speed = min(5.0, float(p('desired_speed')))
        self.max_throttle = float(p('max_throttle'))
        self.linear_gain = float(p('linear_gain'))
        self.rate_hz = float(p('rate_hz'))
        self.require_pacmod_enable = bool(p('require_pacmod_enable'))
        self.steering_speed_limit = float(p('steering_speed_limit'))

        self.pid_speed = PID(
            kp=float(p('pid/kp')),
            ki=float(p('pid/ki')),
            kd=float(p('pid/kd')),
            wg=float(p('pid/wg')),
        )
        self.speed_filter = OnlineFilter(
            cutoff=float(p('filter/cutoff')),
            fs=float(p('filter/fs')),
        )

        # --- State ---
        self.pacmod_enable = False
        self._pacmod_primed = False
        self.speed = 0.0
        self._v_cmd = 0.0
        self._last_steer = 0.0  # radians (front-wheel)
        self._last_accel = 0.0  # m/s^2
        self._has_control_input = False

        # --- Subscribers ---
        self.create_subscription(
            Float32MultiArray, '/mppi/control_output',
            self._control_cb, 10,
        )
        self.create_subscription(
            Bool, '/pacmod/enabled',
            self._enable_cb, 10,
        )
        self.create_subscription(
            VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt',
            self._speed_cb, 10,
        )

        # --- PACMod Publishers ---
        self.global_pub = self.create_publisher(GlobalCmd, '/pacmod/global_cmd', 10)
        self.gear_pub = self.create_publisher(SystemCmdInt, '/pacmod/shift_cmd', 10)
        self.brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 10)
        self.turn_pub = self.create_publisher(SystemCmdInt, '/pacmod/turn_cmd', 10)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        # --- Timer ---
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'pacmod_bridge_node up at {self.rate_hz:.0f} Hz, mode={self.mode}'
        )

    # --- Callbacks ---
    def _control_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self._last_steer = float(msg.data[0])
            self._last_accel = float(msg.data[1])
            self._has_control_input = True

    def _enable_cb(self, msg: Bool):
        self.pacmod_enable = msg.data

    def _speed_cb(self, msg: VehicleSpeedRpt):
        self.speed = float(self.speed_filter.get_data(msg.vehicle_speed))

    # --- Priming ---
    def _prime_pacmod(self):
        global_cmd = GlobalCmd(enable=True, clear_override=True)
        self.global_pub.publish(global_cmd)
        gear_cmd = SystemCmdInt(command=3)  # FORWARD
        self.gear_pub.publish(gear_cmd)
        brake_cmd = SystemCmdFloat(command=0.0)
        self.brake_pub.publish(brake_cmd)
        accel_cmd = SystemCmdFloat(command=0.0)
        self.accel_pub.publish(accel_cmd)
        turn_cmd = SystemCmdInt(command=1)  # NO_SIGNAL
        self.turn_pub.publish(turn_cmd)
        self._pacmod_primed = True
        self.get_logger().warn('PACMod primed: enable + FORWARD')

    # --- Main tick ---
    def _tick(self):
        if self.require_pacmod_enable and not self.pacmod_enable:
            return
        if not self._has_control_input:
            return
        if not self._pacmod_primed:
            self._prime_pacmod()

        # --- Steering ---
        sw_deg = front2steer(math.degrees(self._last_steer))
        steer_cmd = PositionWithSpeed(
            angular_position=math.radians(sw_deg),
            angular_velocity_limit=self.steering_speed_limit,
        )
        self.steer_pub.publish(steer_cmd)

        # --- Throttle ---
        if self.mode == 'pid':
            # Integrate acceleration into velocity command
            dt = 1.0 / self.rate_hz
            self._v_cmd = max(0.0, min(
                self._v_cmd + self._last_accel * dt,
                self.desired_speed,
            ))
            now = self.get_clock().now().nanoseconds * 1e-9
            speed_err = self._v_cmd - self.speed
            if abs(speed_err) < 0.05:
                speed_err = 0.0
            throttle = self.pid_speed.get_control(now, speed_err)
            throttle = max(0.0, min(throttle, self.max_throttle))
        else:
            # Linear: direct mapping from acceleration to throttle
            throttle = max(0.0, min(
                self.linear_gain * self._last_accel,
                self.max_throttle,
            ))

        # --- Publish PACMod ---
        accel_cmd = SystemCmdFloat(command=throttle)
        brake_cmd = SystemCmdFloat(command=0.0)
        global_cmd = GlobalCmd(enable=True, clear_override=True)

        self.accel_pub.publish(accel_cmd)
        self.brake_pub.publish(brake_cmd)
        self.global_pub.publish(global_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PACModBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
