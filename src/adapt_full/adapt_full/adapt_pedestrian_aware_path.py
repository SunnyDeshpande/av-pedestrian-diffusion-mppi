#!/usr/bin/env python3

import math
import scipy.signal as signal
import pygame

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Int32MultiArray
from pacmod2_msgs.msg import PositionWithSpeed, VehicleSpeedRpt, GlobalCmd, SystemCmdFloat, SystemCmdInt


pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick connected")
joystick = pygame.joystick.Joystick(0)
joystick.init()


class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.wg = wg
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def reset(self):
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt = 0.0
            de = 0.0
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
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return float(filted[0])


class PedestrianAwarePath(Node):
    def __init__(self):
        super().__init__('pedestrian_aware_path')

        self.declare_parameter('vehicle_name', "")
        self.declare_parameter('rate_hz', 20)
        self.declare_parameter('desired_speed', 5.0)
        self.declare_parameter('max_acceleration', 1)

        self.declare_parameter('pedestrian/min_danger_angle', 45)
        self.declare_parameter('pedestrian/max_danger_angle', 85)
        self.declare_parameter('pedestrian/max_danger_distance', 10)
        self.declare_parameter('pedestrian/timeout', 1.0)

        self.declare_parameter('braking/hard_brake_effort', 0.6)
        self.declare_parameter('braking/holding_effort', 0.3)

        self.declare_parameter('pid/kp', 0.6)
        self.declare_parameter('pid/ki', 0.0)
        self.declare_parameter('pid/kd', 0.1)
        self.declare_parameter('pid/wg', 10)

        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30)
        self.declare_parameter('filter/order', 4)
        vehicle_name = self.get_parameter('vehicle_name').value
        if vehicle_name == "":
            self.get_logger().warn("No vehicle_name parameter found. Using default parameters.")
        else:
            self.get_logger().info(f"Using vehicle config: {vehicle_name}")

        # Load parameters
        self.rate_hz = self.get_parameter('rate_hz').value
        self.desired_speed = min(5.0, self.get_parameter('desired_speed').value)  # Cap at 5 m/s
        self.max_accel = min(2.0, self.get_parameter('max_acceleration').value)  # Cap at 2 m/s²

        # Load pedestrian detection parameters
        self.min_danger_angle = self.get_parameter('pedestrian/min_danger_angle').value
        self.max_danger_angle = self.get_parameter('pedestrian/max_danger_angle').value
        self.max_danger_distance = self.get_parameter('pedestrian/max_danger_distance').value
        self.pedestrian_timeout = self.get_parameter('pedestrian/timeout').value

        # Load braking parameters
        self.hard_brake_effort = self.get_parameter('braking/hard_brake_effort').value
        self.holding_effort = self.get_parameter('braking/holding_effort').value

        # Initialize PID controller for speed
        self.pid_speed = PID(
            kp=self.get_parameter('pid/kp').value,
            ki=self.get_parameter('pid/ki').value,
            kd=self.get_parameter('pid/kd').value,
            wg=self.get_parameter('pid/wg').value
        )

        # Initialize speed filter
        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter('filter/cutoff').value,
            fs=self.get_parameter('filter/fs').value,
            order=self.get_parameter('filter/order').value
        )

        # Initialize vehicle state
        self.speed = 0.0
        self.pacmod_enable = False

        # Pedestrian detection state
        self.pedestrian_in_danger_zone = False
        self.last_pedestrian_msg_time = None

        # Subscriptions
        self.create_subscription(Bool, '/pacmod/enabled', self.enable_callback, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self.speed_callback, 10)
        self.create_subscription(Int32MultiArray, '/fusion_pedestrian_position', self.pedestrian_callback, 10)

        # Publishers
        self.global_pub = self.create_publisher(GlobalCmd, '/pacmod/global_cmd', 10)
        self.gear_pub = self.create_publisher(SystemCmdInt, '/pacmod/shift_cmd', 10)
        self.brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 10)
        self.turn_pub = self.create_publisher(SystemCmdInt, '/pacmod/turn_cmd', 10)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        # Commands
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd = SystemCmdInt(command=2)  # NEUTRAL
        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.turn_cmd = SystemCmdInt(command=1)  # No signal
        self.steer_cmd = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)

        # Start control loop timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Pedestrian-Aware Path Controller Initialized")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Cruise speed: {self.desired_speed:.2f} m/s")
        self.get_logger().info(f"Danger zone: {self.min_danger_angle}-{self.max_danger_angle} degrees")
        self.get_logger().info(f"Max danger distance: {self.max_danger_distance} meters")
        self.get_logger().info(f"Hard brake effort: {self.hard_brake_effort:.2f}")
        self.get_logger().info(f"Holding brake effort: {self.holding_effort:.2f}")
        self.get_logger().info("=" * 60)

    def speed_callback(self, msg):
        """Receive and filter vehicle speed"""
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_callback(self, msg):
        """Monitor PACMod enable status"""
        self.pacmod_enable = msg.data

    def pedestrian_callback(self, msg):
        """Process fused pedestrian position data"""
        self.last_pedestrian_msg_time = self.get_clock().now()

        # Parse detections (data format: [dist1, angle1, dist2, angle2, ...])
        if len(msg.data) == 0:
            self.pedestrian_in_danger_zone = False
            return

        if len(msg.data) % 2 != 0:
            self.get_logger().warn(f'Invalid pedestrian data length: {len(msg.data)}')
            return

        # Check if any pedestrian is in the danger zone
        danger_detected = False
        for i in range(0, len(msg.data), 2):
            distance = msg.data[i]
            angle = msg.data[i + 1]

            # Check if pedestrian is within danger zone
            if (self.min_danger_angle <= angle <= self.max_danger_angle and
                distance <= self.max_danger_distance):
                danger_detected = True
                self.get_logger().warn(
                    f"PEDESTRIAN IN DANGER ZONE! Distance: {distance}m, Angle: {angle}°"
                )
                break

        self.pedestrian_in_danger_zone = danger_detected

    def check_joystick_enable(self):
        """Check joystick buttons for enable/disable commands"""
        pygame.event.pump()
        try:
            lb = joystick.get_button(6)  # Left bumper
            rb = joystick.get_button(7)  # Right bumper
        except pygame.error:
            self.get_logger().warn("Joystick read failed")
            return 2
        if lb and rb:
            return 1  # Enable
        elif lb and not rb:
            return 0  # Disable
        return 2  # No change

    def control_loop(self):
        """Main control loop - called at rate_hz frequency"""
        joy_enable = self.check_joystick_enable()

        # Check for pedestrian message timeout
        if self.last_pedestrian_msg_time is not None:
            time_since_last_msg = (self.get_clock().now() - self.last_pedestrian_msg_time).nanoseconds * 1e-9
            if time_since_last_msg > self.pedestrian_timeout:
                # No recent pedestrian data - assume clear
                if self.pedestrian_in_danger_zone:
                    self.get_logger().info("Pedestrian message timeout - assuming clear")
                self.pedestrian_in_danger_zone = False

        # Handle enable request
        if joy_enable == 1 and not self.pacmod_enable:
            self.global_cmd.enable = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)

            self.gear_cmd.command = 3  # FORWARD
            self.gear_pub.publish(self.gear_cmd)

            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)

            # Keep steering straight (0 degrees)
            self.steer_cmd.angular_position = 0.0
            self.steer_pub.publish(self.steer_cmd)

            self.turn_cmd.command = 1  # No turn signal
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn('Vehicle enabled - Pedestrian-aware mode active')

        # Handle disable request
        elif joy_enable == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1  # No signal
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn('Vehicle disabled')

        # Execute pedestrian-aware controller
        elif joy_enable != 0 and self.pacmod_enable:
            # Keep steering straight
            self.steer_cmd.angular_position = 0.0
            self.steer_pub.publish(self.steer_cmd)

            # Determine target speed based on pedestrian detection
            if self.pedestrian_in_danger_zone:
                # STOP! Pedestrian detected in danger zone
                target_speed = 0.0
                emergency_stop = True
            else:
                # Safe to proceed at cruise speed
                target_speed = self.desired_speed
                emergency_stop = False

            # ========== EMERGENCY STOP MODE ==========
            if emergency_stop:
                # EMERGENCY! Bypass PID, apply hard brake immediately
                throttle_cmd = 0.0
                brake_cmd = self.hard_brake_effort

                self.get_logger().error(
                    f"🚨 EMERGENCY STOP! Pedestrian in danger zone | "
                    f"Applying hard brake: {brake_cmd:.2f}"
                )

            # ========== NORMAL CONTROL ==========
            else:
                # Calculate speed error
                now = self.get_clock().now().nanoseconds * 1e-9
                speed_error = target_speed - self.speed

                # Dead zone to prevent oscillation
                if abs(speed_error) < 0.05:
                    speed_error = 0.0

                # Get PID output
                pid_output = self.pid_speed.get_control(now, speed_error)

                # ========== ACTUATION LOGIC ==========
                if pid_output > 0:
                    # Need to accelerate
                    throttle_cmd = max(0.0, min(pid_output, self.max_accel))
                    brake_cmd = 0.0

                elif pid_output < 0:
                    # Need to brake
                    throttle_cmd = 0.0
                    brake_cmd = min(abs(pid_output), 1.0)  # Clamp brake to 1.0

                else:
                    # PID output is zero (at target)
                    throttle_cmd = 0.0
                    brake_cmd = 0.0

                # ========== HOLDING LOGIC ==========
                # If target is stopped and vehicle is nearly stopped, apply holding brake
                if target_speed == 0.0 and self.speed < 0.1:
                    brake_cmd = self.holding_effort
                    throttle_cmd = 0.0

                # Periodic debug logging
                self.get_logger().debug(
                    f"Speed: {self.speed:.2f}/{target_speed:.2f} m/s | "
                    f"Throttle: {throttle_cmd:.2f} | Brake: {brake_cmd:.2f} | "
                    f"Pedestrian: {'DETECTED' if self.pedestrian_in_danger_zone else 'Clear'}"
                )

            # ========== PUBLISH COMMANDS ==========
            self.accel_cmd.command = throttle_cmd
            self.brake_cmd.command = brake_cmd
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            # Keep global command enabled
            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)


def main(args=None):
    rclpy.init(args=args)
    controller = PedestrianAwarePath()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
