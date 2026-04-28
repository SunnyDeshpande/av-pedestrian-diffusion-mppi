"""Generic ROS2 MPPI node — sim / rosbag test harness.

Decoupled from adapt's hardware topic contract. Consumes
  /odom                     nav_msgs/Odometry
  /adapt/reference_path     nav_msgs/Path
  /obstacles                geometry_msgs/PolygonStamped
and publishes
  /pacmod/steering_cmd      std_msgs/Float64 (front-wheel rad)
  /pacmod/accel_cmd         std_msgs/Float64 (m/s^2)

For full adapt integration, use `adapt_mppi_node` (pacmod2_msgs +
septentrio_gnss_driver native)."""
import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PolygonStamped
from std_msgs.msg import Float64

from .mppi import MPPI
from .reference_path import ReferencePath


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class AdaptMPPINode(Node):
    def __init__(self):
        super().__init__('adapt_mppi_node')

        self.declare_parameter('v_ref', 3.0)
        self.declare_parameter('K', 600)
        self.declare_parameter('H', 30)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('sigma_steer', 0.05)
        self.declare_parameter('sigma_accel', 0.8)
        self.declare_parameter('lambda_', 1.0)
        self.declare_parameter('rate_hz', 10.0)

        p = lambda n: self.get_parameter(n).value
        self.rate_hz = float(p('rate_hz'))
        self.mppi = MPPI(
            K=int(p('K')),
            H=int(p('H')),
            dt=float(p('dt')),
            sigma_steer=float(p('sigma_steer')),
            sigma_accel=float(p('sigma_accel')),
            lam=float(p('lambda_')),
            v_ref=float(p('v_ref')),
        )

        self.state = None          # np.array([x,y,psi,v])
        self.ref_path = None       # ReferencePath
        self.obstacles = np.zeros((0, 2))

        self.create_subscription(Odometry, '/odom', self._odom_cb, 20)
        self.create_subscription(Path, '/adapt/reference_path', self._path_cb, 5)
        self.create_subscription(PolygonStamped, '/obstacles', self._obs_cb, 10)

        self.pub_steer = self.create_publisher(Float64, '/pacmod/steering_cmd', 10)
        self.pub_accel = self.create_publisher(Float64, '/pacmod/accel_cmd', 10)

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info('adapt_mppi_node up at %.1f Hz' % self.rate_hz)

    def _odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        psi = quat_to_yaw(msg.pose.pose.orientation)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = math.hypot(vx, vy)
        self.state = np.array([px, py, psi, v], dtype=float)

    def _path_cb(self, msg: Path):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        if len(pts) >= 2:
            self.ref_path = ReferencePath(pts)

    def _obs_cb(self, msg: PolygonStamped):
        pts = [(pt.x, pt.y) for pt in msg.polygon.points]
        self.obstacles = np.array(pts, dtype=float) if pts else np.zeros((0, 2))

    def _tick(self):
        if self.state is None or self.ref_path is None:
            return
        u = self.mppi.update(self.state, self.ref_path, self.obstacles)
        delta, accel = float(u[0]), float(u[1])
        self.pub_steer.publish(Float64(data=delta))
        self.pub_accel.publish(Float64(data=accel))


def main(args=None):
    rclpy.init(args=args)
    node = AdaptMPPINode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
