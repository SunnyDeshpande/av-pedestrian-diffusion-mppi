import os
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import pymap3d as pm
import pygame

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from pacmod2_msgs.msg import PositionWithSpeed, VehicleSpeedRpt, GlobalCmd, SystemCmdFloat, SystemCmdInt
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# Initialize pygame for joystick
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick connected")
joystick = pygame.joystick.Joystick(0)
joystick.init()


class PID:
    """
    PID controller for longitudinal speed control.
    """
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
    """
    Butterworth low-pass filter for smoothing noisy sensor data.
    Reduces high-frequency noise in speed measurements.
    """
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted[0]


class StanleyController(Node):

    def __init__(self):
        super().__init__('stanley_controller_node')
        
        # Declare parameters with default values
        self.declare_parameter('rate_hz', 20)
        self.declare_parameter('wheelbase', 2.57)
        self.declare_parameter('offset', 1.26)
        self.declare_parameter('origin_lat', 40.0927422)
        self.declare_parameter('origin_lon', -88.2359639)
        self.declare_parameter('desired_speed', 2.0)
        self.declare_parameter('max_acceleration', 0.5)

        # Stanley-specific parameters
        self.declare_parameter('stanley/k', 2.5)  # Stanley gain for cross-track error
        self.declare_parameter('stanley/k_soft', 0.5)  # Softening constant for low speeds
        
        self.declare_parameter('pid/kp', 0.6)
        self.declare_parameter('pid/ki', 0.0)
        self.declare_parameter('pid/kd', 0.1)
        self.declare_parameter('pid/wg', 10)

        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30)
        self.declare_parameter('filter/order', 4)
        self.declare_parameter('vehicle_name', "")
        
        vehicle_name = self.get_parameter('vehicle_name').value
        if vehicle_name == "":
            self.get_logger().warn("No vehicle_name parameter found. Using default parameters.")
        else:
            self.get_logger().info(f"Using vehicle config: {vehicle_name}")

        # Load parameters
        self.rate_hz = self.get_parameter('rate_hz').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.offset = self.get_parameter('offset').value
        self.olat = self.get_parameter('origin_lat').value
        self.olon = self.get_parameter('origin_lon').value
        self.desired_speed = min(5.0, self.get_parameter('desired_speed').value)
        self.max_accel = min(2.0, self.get_parameter('max_acceleration').value)
        
        # Stanley parameters
        self.stanley_k = self.get_parameter('stanley/k').value
        self.stanley_k_soft = self.get_parameter('stanley/k_soft').value
        
        self.pid_speed = PID(
            kp=self.get_parameter('pid/kp').value,
            ki=self.get_parameter('pid/ki').value,
            kd=self.get_parameter('pid/kd').value,
            wg=self.get_parameter('pid/wg').value
        )
        
        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter('filter/cutoff').value,
            fs=self.get_parameter('filter/fs').value,
            order=self.get_parameter('filter/order').value
        )

        self.goal = 0

        # Subscriptions
        self.create_subscription(NavSatFix, '/navsatfix', self.gnss_callback, 10)
        self.create_subscription(INSNavGeod, '/insnavgeod', self.ins_callback, 10) #doesnt wokr with rosbag "lpac" error
        self.create_subscription(Bool, '/pacmod/enabled', self.enable_callback, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self.speed_callback, 10)

        # Publishers
        self.global_pub = self.create_publisher(GlobalCmd, '/pacmod/global_cmd', 10)
        self.gear_pub = self.create_publisher(SystemCmdInt, '/pacmod/shift_cmd', 10)
        self.brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 10)
        self.turn_pub = self.create_publisher(SystemCmdInt, '/pacmod/turn_cmd', 10)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)
        self.waypoints_pub = self.create_publisher(Marker, '/visualization/waypoints', 10)
        self.next_waypoint_pub = self.create_publisher(Marker, '/visualization/next_waypoint', 10)
        
        # Commands
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd = SystemCmdInt(command=2)  # NEUTRAL
        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.turn_cmd = SystemCmdInt(command=1)  # no signal
        self.steer_cmd = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)

        self.read_waypoints()

        # Initialize vehicle state
        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.pacmod_enable = False

        self.dist_arr = np.zeros(len(self.path_points_lon_x))
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

    def gnss_callback(self, msg):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def ins_callback(self, msg):
        self.heading = msg.heading

    def speed_callback(self, msg):
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def read_waypoints(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/track.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        self.path_points_lon_x = [float(p[0]) for p in path_points]
        self.path_points_lat_y = [float(p[1]) for p in path_points]
        self.path_points_heading = [float(p[2]) for p in path_points]
        self.wp_size = len(self.path_points_lon_x)

    def heading_to_yaw(self, heading):
        return np.radians(90 - heading) if heading < 270 else np.radians(450 - heading)

    def wps_to_local_xy(self, lon, lat):
        x, y, _ = pm.geodetic2enu(lat, lon, 0, self.olat, self.olon, 0)
        return x, y

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def front2steer(self, f_angle):
        f_angle = max(min(f_angle, 35), -35)
        angle = abs(f_angle)
        steer_angle = -0.1084 * angle ** 2 + 21.775 * angle
        result = steer_angle if f_angle >= 0 else -steer_angle
        
        # Double-check output
        if abs(result) > 450:
            self.get_logger().error(f"Invalid steering wheel angle: {result}° - Clamping to ±450°")
            result = max(min(result, 450), -450)
        return round(result, 2)
    
    def check_joystick_enable(self):
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

    def get_gem_state(self):
        local_x, local_y = self.wps_to_local_xy(self.lon, self.lat)
        yaw = self.heading_to_yaw(self.heading)
        # Adjust for sensor offset
        x = local_x - self.offset * math.cos(yaw)
        y = local_y - self.offset * math.sin(yaw)
        return x, y, yaw

    def compute_cross_track_error(self, curr_x, curr_y, target_x, target_y, path_yaw):
        dx = curr_x - target_x
        dy = curr_y - target_y
        cross_track_error = -dx * math.sin(path_yaw) + dy * math.cos(path_yaw)
        
        return cross_track_error
    
    # add the math formula here for stanley control
    def stanley_control(self, curr_x, curr_y, curr_yaw, curr_speed, target_x, target_y, path_yaw):
        heading_error = self.normalize_angle(path_yaw - curr_yaw)
        cross_track_error = self.compute_cross_track_error(
            curr_x, curr_y, target_x, target_y, path_yaw
        )
        cross_track_term = math.atan(
            self.stanley_k * cross_track_error / (curr_speed + self.stanley_k_soft)
        )
        steering_angle = heading_error + cross_track_term
        max_steer = math.radians(35)
        steering_angle = max(min(steering_angle, max_steer), -max_steer)
        return steering_angle, heading_error, cross_track_error

    def publish_visualization_markers(self, target_x, target_y, cross_track_error):
        """
        Publish visualization markers for RViz:
        1. All waypoints (white points)
        2. Next target waypoint (green sphere)
        3. Cross-track error indicator (red line from vehicle to path)
        """
        # Draw all waypoints
        waypoints_marker = Marker()
        waypoints_marker.header.frame_id = "map"
        waypoints_marker.header.stamp = self.get_clock().now().to_msg()
        waypoints_marker.ns = "waypoints"
        waypoints_marker.id = 0
        waypoints_marker.type = Marker.POINTS
        waypoints_marker.action = Marker.ADD
        waypoints_marker.scale.x = 1.0
        waypoints_marker.scale.y = 1.0
        waypoints_marker.color.r = 1.0
        waypoints_marker.color.g = 1.0
        waypoints_marker.color.b = 1.0
        waypoints_marker.color.a = 1.0

        for x, y in zip(self.path_points_x, self.path_points_y):
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            waypoints_marker.points.append(p)

        self.waypoints_pub.publish(waypoints_marker)

        # Draw next target waypoint
        next_waypoint_marker = Marker()
        next_waypoint_marker.header.frame_id = "map"
        next_waypoint_marker.header.stamp = self.get_clock().now().to_msg()
        next_waypoint_marker.ns = "next_waypoint"
        next_waypoint_marker.id = 1
        next_waypoint_marker.type = Marker.SPHERE
        next_waypoint_marker.action = Marker.ADD
        next_waypoint_marker.pose.position.x = target_x
        next_waypoint_marker.pose.position.y = target_y
        next_waypoint_marker.pose.position.z = 0.0
        next_waypoint_marker.scale.x = 1.0
        next_waypoint_marker.scale.y = 1.0
        next_waypoint_marker.scale.z = 1.0
        next_waypoint_marker.color.r = 0.0
        next_waypoint_marker.color.g = 1.0
        next_waypoint_marker.color.b = 0.0
        next_waypoint_marker.color.a = 1.0

        self.next_waypoint_pub.publish(next_waypoint_marker)

        # Draw cross-track error visualization (line from vehicle to path)
        cte_marker = Marker()
        cte_marker.header.frame_id = "base_link"
        cte_marker.header.stamp = self.get_clock().now().to_msg()
        cte_marker.ns = "cross_track_error"
        cte_marker.id = 2
        cte_marker.type = Marker.LINE_STRIP
        cte_marker.action = Marker.ADD
        cte_marker.scale.x = 0.1
        cte_marker.color.r = 1.0
        cte_marker.color.g = 0.0
        cte_marker.color.b = 0.0
        cte_marker.color.a = 1.0

        # Start point (vehicle position)
        p1 = Point()
        p1.x = 0.0
        p1.y = 0.0
        p1.z = 0.0
        cte_marker.points.append(p1)

        # End point (perpendicular to path)
        p2 = Point()
        p2.x = 0.0
        p2.y = cross_track_error
        p2.z = 0.0
        cte_marker.points.append(p2)

        self.next_waypoint_pub.publish(cte_marker)
    
    def control_loop(self):
        joy_enable = self.check_joystick_enable()

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

            self.turn_cmd.command = 3  # LEFT signal
            self.turn_pub.publish(self.turn_cmd)
            
            self.get_logger().warn('Vehicle enabled and forward gear engaged')

        elif joy_enable == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1  # No signal
            self.turn_pub.publish(self.turn_cmd)
            self.get_logger().warn('Vehicle disabled')

        elif joy_enable != 0 and self.pacmod_enable:
            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            
            # Find closest waypoint ahead of vehicle
            min_dist = float('inf')
            self.goal = 0
            found_valid_waypoint = False
            
            for i in range(self.wp_size):
                self.dist_arr[i] = self.dist(
                    (self.path_points_x[i], self.path_points_y[i]), 
                    (curr_x, curr_y)
                )
                dx = self.path_points_x[i] - curr_x
                dy = self.path_points_y[i] - curr_y
                angle_to_wp = math.atan2(dy, dx)
                angle_diff = self.normalize_angle(angle_to_wp - curr_yaw)

                if abs(angle_diff) < math.pi/2 and self.dist_arr[i] < min_dist:
                    min_dist = self.dist_arr[i]
                    self.goal = i
                    found_valid_waypoint = True
            
            if not found_valid_waypoint:
                self.get_logger().error("No valid waypoint ahead - Stopping")
                self.brake_cmd.command = 0.5
                self.accel_cmd.command = 0.0
                self.brake_pub.publish(self.brake_cmd)
                self.accel_pub.publish(self.accel_cmd)
                return
            
            target_x = self.path_points_x[self.goal]
            target_y = self.path_points_y[self.goal]
            path_yaw = self.heading_to_yaw(self.path_points_heading[self.goal])
            
            steering_angle, heading_error, cross_track_error = self.stanley_control(
                curr_x, curr_y, curr_yaw, self.speed,
                target_x, target_y, path_yaw
            )
            
            steering_wheel_angle = self.front2steer(math.degrees(steering_angle))
            
            self.steer_cmd.angular_position = math.radians(steering_wheel_angle)
            self.steer_pub.publish(self.steer_cmd)

            now = self.get_clock().now().nanoseconds * 1e-9
            speed_error = self.desired_speed - self.speed
            if abs(speed_error) < 0.05:
                speed_error = 0.0
            throttle_cmd = self.pid_speed.get_control(now, speed_error)
            throttle_cmd = max(0.0, min(throttle_cmd, self.max_accel))

            self.accel_cmd.command = throttle_cmd  
            self.brake_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)
            
            # Publish visualization markers
            self.publish_visualization_markers(target_x, target_y, cross_track_error)

            self.get_logger().info(
                f"Stanley - Goal: {self.goal}/{self.wp_size}, "
                f"Pos: ({curr_x:.2f}, {curr_y:.2f}), "
                f"CTE: {cross_track_error:.3f}m, "
                f"Head_err: {math.degrees(heading_error):.1f}°, "
                f"Speed: {self.speed:.2f}m/s, "
                f"Throttle: {throttle_cmd:.2f}, "
                f"Steer: {steering_wheel_angle:.2f}°"
            )


def main(args=None):
    rclpy.init(args=args)
    stanley_controller = StanleyController()
    rclpy.spin(stanley_controller)
    stanley_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()