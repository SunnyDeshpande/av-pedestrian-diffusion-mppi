"""MPPI planner node — publishes raw control output to a topic.

Decoupled from PACMod actuation. Publishes [steer_rad, accel_m_s2]
on /mppi/control_output for a separate low-level bridge node to consume.

Use controller:=mppi-split in the launch file to enable this mode.
"""
import colorsys
import csv
import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32MultiArray, Float32MultiArray
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from pacmod2_msgs.msg import VehicleSpeedRpt
from septentrio_gnss_driver.msg import INSNavGeod

from .mppi import MPPI
from .reference_path import ReferencePath

# --- WGS-84 geodetic -> ENU ------------------------------------------------
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _geodetic_to_ecef(lat_deg, lon_deg, h):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sl * sl)
    return ((N + h) * cl * math.cos(lon),
            (N + h) * cl * math.sin(lon),
            (N * (1.0 - _WGS84_E2) + h) * sl)


def geodetic2enu(lat, lon, h, lat0, lon0, h0):
    x, y, z = _geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = _geodetic_to_ecef(lat0, lon0, h0)
    dx, dy, dz = x - x0, y - y0, z - z0
    slat, clat = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    slon, clon = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u = clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u


class OnlineFilter:
    def __init__(self, cutoff, fs, order=1):
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * max(cutoff, 1e-6) / max(fs, 1e-6))
        self._y = None

    def get_data(self, x):
        self._y = x if self._y is None else (self.alpha * x + (1.0 - self.alpha) * self._y)
        return self._y


def heading_to_yaw(heading_deg):
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    return math.radians(450.0 - heading_deg)


class MPPIPlannerNode(Node):
    def __init__(self):
        super().__init__('mppi_planner_node')

        # --- params ---
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('wheelbase', 1.75)
        self.declare_parameter('offset', 1.26)
        self.declare_parameter('origin_lat', 40.0927422)
        self.declare_parameter('origin_lon', -88.2359639)
        self.declare_parameter('desired_speed', 2.0)
        self.declare_parameter('waypoints_csv', '')
        self.declare_parameter('vehicle_name', '')

        self.declare_parameter('mppi/K', 600)
        self.declare_parameter('mppi/H', 30)
        self.declare_parameter('mppi/dt', 0.1)
        self.declare_parameter('mppi/sigma_steer', 0.15)
        self.declare_parameter('mppi/sigma_accel', 0.5)
        self.declare_parameter('mppi/lambda_', 0.1)
        self.declare_parameter('mppi/clearance', 3.0)
        self.declare_parameter('mppi/lookahead_m', 8.0)
        self.declare_parameter('mppi/w_pos', 15.0)
        self.declare_parameter('mppi/w_vel', 5.0)
        self.declare_parameter('mppi/w_curv', 2.0)
        self.declare_parameter('mppi/w_obs', 150.0)
        self.declare_parameter('mppi/w_obs_hard', 250.0)
        self.declare_parameter('mppi/w_obs_soft', 40.0)
        self.declare_parameter('mppi/device', '')

        self.declare_parameter('prediction_source', 'raw')

        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30.0)
        self.declare_parameter('filter/order', 4)

        # Viz params
        self.declare_parameter('viz/frame_id', 'map')
        self.declare_parameter('viz/num_samples', 19)

        p = lambda n: self.get_parameter(n).value
        self.rate_hz = float(p('rate_hz'))
        self.wheelbase = float(p('wheelbase'))
        self.offset = float(p('offset'))
        self.olat = float(p('origin_lat'))
        self.olon = float(p('origin_lon'))
        self.desired_speed = min(5.0, float(p('desired_speed')))

        device_param = str(p('mppi/device')).strip() or None
        self.mppi = MPPI(
            K=int(p('mppi/K')),
            H=int(p('mppi/H')),
            dt=float(p('mppi/dt')),
            sigma_steer=float(p('mppi/sigma_steer')),
            sigma_accel=float(p('mppi/sigma_accel')),
            lam=float(p('mppi/lambda_')),
            v_ref=self.desired_speed,
            w_pos=float(p('mppi/w_pos')),
            w_vel=float(p('mppi/w_vel')),
            w_curv=float(p('mppi/w_curv')),
            w_obs=float(p('mppi/w_obs')),
            w_obs_hard=float(p('mppi/w_obs_hard')),
            w_obs_soft=float(p('mppi/w_obs_soft')),
            clearance=float(p('mppi/clearance')),
            lookahead_m=float(p('mppi/lookahead_m')),
            wheelbase=self.wheelbase,
            device=device_param,
        )

        self.speed_filter = OnlineFilter(
            cutoff=float(p('filter/cutoff')),
            fs=float(p('filter/fs')),
            order=int(p('filter/order')),
        )

        wp_csv = str(p('waypoints_csv')) or self._default_waypoints_path()
        self.ref_path = self._load_waypoints(wp_csv)

        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.obstacles = np.zeros((0, 2))
        self.prediction_source = str(p('prediction_source'))

        # --- Subscribers ---
        self.create_subscription(NavSatFix, '/navsatfix', self._gnss_cb, 10)
        self.create_subscription(INSNavGeod, '/insnavgeod', self._ins_cb, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt',
                                 self._speed_cb, 10)

        if self.prediction_source == 'predicted':
            self.create_subscription(
                Float32MultiArray, '/pedestrian_predictions_tensor',
                self._pred_tensor_cb, 10,
            )
            self.get_logger().info(
                'Obstacle source: /pedestrian_predictions_tensor (velocity-aware)'
            )
        else:
            self.create_subscription(
                Int32MultiArray, '/fusion_pedestrian_position',
                self._ped_cb, 10,
            )
            self.get_logger().info(
                'Obstacle source: /fusion_pedestrian_position (raw detections)'
            )

        # --- Control output publisher ---
        self.control_pub = self.create_publisher(
            Float32MultiArray, '/mppi/control_output', 10)

        # --- Visualization ---
        self.viz_frame = str(p('viz/frame_id'))
        self.viz_num_samples = int(p('viz/num_samples'))
        self._sample_palette = [
            colorsys.hsv_to_rgb(i / max(self.viz_num_samples, 1), 0.45, 0.95)
            for i in range(self.viz_num_samples)
        ]

        latched_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.viz_ref_pub = self.create_publisher(
            Path, '/adapt/viz/reference_path', latched_qos)
        self.viz_chosen_pub = self.create_publisher(
            Path, '/adapt/viz/chosen_trajectory', 10)
        self.viz_samples_pub = self.create_publisher(
            MarkerArray, '/adapt/viz/sampled_trajectories', 10)
        self.viz_obstacles_pub = self.create_publisher(
            MarkerArray, '/adapt/viz/obstacles', 10)
        self._publish_reference_path()

        self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(
            f'mppi_planner_node up at {self.rate_hz:.1f} Hz, '
            f'waypoints={len(self.ref_path.xy)}, v_ref={self.desired_speed:.1f} m/s'
        )

    # --- helpers ---
    def _default_waypoints_path(self):
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('adapt_full')
        return os.path.join(share, 'waypoints', 'track.csv')

    def _load_waypoints(self, path):
        lon_x, lat_y = [], []
        with open(path) as f:
            for row in csv.reader(f):
                if not row:
                    continue
                lon_x.append(float(row[0]))
                lat_y.append(float(row[1]))
        pts = []
        for lon, lat in zip(lon_x, lat_y):
            x, y, _ = geodetic2enu(lat, lon, 0.0, self.olat, self.olon, 0.0)
            pts.append((x, y))
        if len(pts) < 2:
            raise RuntimeError(f'waypoints file {path} has <2 points')
        return ReferencePath(pts)

    # --- callbacks ---
    def _gnss_cb(self, msg: NavSatFix):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def _ins_cb(self, msg: INSNavGeod):
        self.heading = msg.heading

    def _speed_cb(self, msg: VehicleSpeedRpt):
        self.speed = float(self.speed_filter.get_data(msg.vehicle_speed))

    def _ped_cb(self, msg: Int32MultiArray):
        data = msg.data
        if not data or len(data) % 2 != 0:
            self.obstacles = np.zeros((0, 2))
            return
        if self.lat == 0.0 and self.lon == 0.0:
            self.obstacles = np.zeros((0, 2))
            return

        ex, ey, yaw = self._get_gem_state()
        out = []
        for i in range(0, len(data), 2):
            dist = float(data[i])
            deg = float(data[i + 1])
            rad = math.radians(deg)
            xe = dist * math.cos(rad)
            ye = dist * math.sin(rad)
            xw = ex + xe * math.cos(yaw) - ye * math.sin(yaw)
            yw = ey + xe * math.sin(yaw) + ye * math.cos(yaw)
            out.append((xw, yw))
        self.obstacles = np.asarray(out, dtype=float) if out else np.zeros((0, 2))

    def _pred_tensor_cb(self, msg: Float32MultiArray):
        if not msg.data:
            self.obstacles = np.zeros((0, 5))
            return
        if self.lat == 0.0 and self.lon == 0.0:
            self.obstacles = np.zeros((0, 5))
            return

        dims = msg.layout.dim
        if len(dims) >= 3:
            M = dims[0].size
            H = dims[1].size
        elif len(dims) == 2:
            M = dims[0].size
            H = dims[1].size
        else:
            self.obstacles = np.zeros((0, 5))
            return

        arr = np.array(msg.data, dtype=np.float32).reshape(M, H, 2)

        ex, ey, yaw = self._get_gem_state()
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        obs = np.zeros((M, 5), dtype=np.float64)
        dt = 0.25
        for i in range(M):
            xe, ye = float(arr[i, 0, 0]), float(arr[i, 0, 1])
            xw = ex + xe * cos_y - ye * sin_y
            yw = ey + xe * sin_y + ye * cos_y

            if H >= 2:
                dx_e = float(arr[i, 1, 0] - arr[i, 0, 0])
                dy_e = float(arr[i, 1, 1] - arr[i, 0, 1])
                vx_e, vy_e = dx_e / dt, dy_e / dt
                vx_w = vx_e * cos_y - vy_e * sin_y
                vy_w = vx_e * sin_y + vy_e * cos_y
            else:
                vx_w, vy_w = 0.0, 0.0

            obs[i] = [xw, yw, vx_w, vy_w, 0.8]

        self.obstacles = obs

    # --- geometry ---
    def _get_gem_state(self):
        local_x, local_y, _ = geodetic2enu(self.lat, self.lon, 0.0,
                                           self.olat, self.olon, 0.0)
        yaw = heading_to_yaw(self.heading)
        x = local_x - self.offset * math.cos(yaw)
        y = local_y - self.offset * math.sin(yaw)
        return x, y, yaw

    # --- control loop ---
    def _control_loop(self):
        if self.lat == 0.0 and self.lon == 0.0:
            return

        x, y, yaw = self._get_gem_state()
        state = np.array([x, y, yaw, max(self.speed, 0.0)], dtype=float)

        u = self.mppi.update(state, self.ref_path, self.obstacles)
        delta = float(u[0])
        accel = float(u[1])

        # Publish raw control output
        ctrl_msg = Float32MultiArray(data=[delta, accel])
        self.control_pub.publish(ctrl_msg)

        self.get_logger().info(
            f'MPPI | pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg '
            f'v={self.speed:.2f} steer={math.degrees(delta):.1f}deg '
            f'accel={accel:.2f} obs={len(self.obstacles)}',
            throttle_duration_sec=1.0,
        )

        self._publish_viz()

    # --- visualization ---
    def _publish_reference_path(self):
        msg = Path()
        msg.header.frame_id = self.viz_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.ref_path.xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.viz_ref_pub.publish(msg)

    def _publish_viz(self):
        if getattr(self.mppi, 'last_traj', None) is None:
            return
        stamp = self.get_clock().now().to_msg()
        traj = self.mppi.last_traj
        w = self.mppi.last_weights
        K, H, _ = traj.shape

        # Chosen trajectory (weighted mean)
        mean_traj = self.mppi.last_mean_traj
        path = Path()
        path.header.frame_id = self.viz_frame
        path.header.stamp = stamp
        for h in range(H):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(mean_traj[h, 0])
            ps.pose.position.y = float(mean_traj[h, 1])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.viz_chosen_pub.publish(path)

        # Sampled rollouts
        N = min(self.viz_num_samples, K)
        top_idx = np.argsort(w)[-N:][::-1]
        samples = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.viz_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        samples.markers.append(clear)
        for i, k in enumerate(top_idx):
            r_, g_, b_ = self._sample_palette[i]
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.header.stamp = stamp
            m.ns = 'mppi_samples'
            m.id = i + 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.r = float(r_)
            m.color.g = float(g_)
            m.color.b = float(b_)
            m.color.a = 0.75
            m.pose.orientation.w = 1.0
            for h in range(H):
                pt = Point()
                pt.x = float(traj[k, h, 0])
                pt.y = float(traj[k, h, 1])
                pt.z = 0.0
                m.points.append(pt)
            samples.markers.append(m)
        self.viz_samples_pub.publish(samples)

        # Obstacles
        obs_msg = MarkerArray()
        clear2 = Marker()
        clear2.header.frame_id = self.viz_frame
        clear2.header.stamp = stamp
        clear2.action = Marker.DELETEALL
        obs_msg.markers.append(clear2)
        r = float(self.mppi.clearance)
        for i in range(len(self.obstacles)):
            ox, oy = float(self.obstacles[i, 0]), float(self.obstacles[i, 1])
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.header.stamp = stamp
            m.ns = 'obstacles'
            m.id = i + 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = ox
            m.pose.position.y = oy
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * r
            m.scale.y = 2.0 * r
            m.scale.z = 0.15
            m.color.r = 1.0
            m.color.g = 0.25
            m.color.b = 0.25
            m.color.a = 0.35
            obs_msg.markers.append(m)
        self.viz_obstacles_pub.publish(obs_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPPIPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
