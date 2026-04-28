"""Adapt MPPI node — canonical adapt-integrated controller.

Drop-in replacement for adapt_full.adapt_stanley_controller.
Mirrors Stanley's localization, PACMod handshake, and waypoint CSV
loading, but swaps the Stanley math for the pure-numpy MPPI class in
adapt_mppi.mppi. Consumes /fusion_pedestrian_position (adapt's
existing fused pedestrian output) as the MPPI obstacle source.

Topic contract (identical to adapt_stanley_controller):
  Subscribes:
    /navsatfix                         sensor_msgs/NavSatFix
    /insnavgeod                        septentrio_gnss_driver/INSNavGeod
    /pacmod/enabled                    std_msgs/Bool
    /pacmod/vehicle_speed_rpt          pacmod2_msgs/VehicleSpeedRpt
    /fusion_pedestrian_position        std_msgs/Int32MultiArray
        flat [dist_m, bearing_deg, dist_m, bearing_deg, ...] ego frame

Backend: torch MPPI via pytorch_mppi, with the adapt repo's cost
structure (goal position + velocity + stability + pedestrian
confidence-growth obstacle cost). See mppi_controller/mppi.py for the
adaptation.

  Publishes:
    /pacmod/global_cmd                 pacmod2_msgs/GlobalCmd
    /pacmod/shift_cmd                  pacmod2_msgs/SystemCmdInt
    /pacmod/brake_cmd                  pacmod2_msgs/SystemCmdFloat
    /pacmod/accel_cmd                  pacmod2_msgs/SystemCmdFloat
    /pacmod/turn_cmd                   pacmod2_msgs/SystemCmdInt
    /pacmod/steering_cmd               pacmod2_msgs/PositionWithSpeed
"""
import colorsys
import csv
import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Bool, Int32MultiArray
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from pacmod2_msgs.msg import (
    GlobalCmd, PositionWithSpeed, SystemCmdFloat, SystemCmdInt,
    VehicleSpeedRpt,
)
from septentrio_gnss_driver.msg import INSNavGeod

from .mppi import MPPI
from .reference_path import ReferencePath

# --- WGS-84 geodetic -> ENU (vendored; avoids a pymap3d install) -------
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
    """Equivalent to pymap3d.geodetic2enu; returns (e, n, u) in metres."""
    x, y, z = _geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = _geodetic_to_ecef(lat0, lon0, h0)
    dx, dy, dz = x - x0, y - y0, z - z0
    slat, clat = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    slon, clon = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u = clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u


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
    """Simple exponential moving average. Equivalent damping to a 1st-order
    low-pass at cutoff=`cutoff` Hz, sampled at `fs` Hz. `order` kept for
    API compatibility but unused."""
    def __init__(self, cutoff, fs, order=1):
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * max(cutoff, 1e-6) / max(fs, 1e-6))
        self._y = None

    def get_data(self, x):
        self._y = x if self._y is None else (self.alpha * x + (1.0 - self.alpha) * self._y)
        return self._y


def heading_to_yaw(heading_deg):
    """Compass heading (0=N, CW positive, degrees) -> ENU yaw (radians, 0=+x, CCW)."""
    if heading_deg < 270.0:
        return math.radians(90.0 - heading_deg)
    return math.radians(450.0 - heading_deg)


def front2steer(f_angle_deg):
    """Front-wheel angle (deg) -> steering-wheel angle (deg), adapt calibration."""
    a = max(min(f_angle_deg, 35.0), -35.0)
    mag = abs(a)
    sw = -0.1084 * mag * mag + 21.775 * mag
    sw = sw if a >= 0 else -sw
    return max(min(sw, 450.0), -450.0)


class AdaptMPPINode(Node):
    def __init__(self):
        super().__init__('adapt_mppi_node')

        # --- params -----------------------------------------------------
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('wheelbase', 1.75)
        self.declare_parameter('offset', 1.26)
        self.declare_parameter('origin_lat', 40.0927422)
        self.declare_parameter('origin_lon', -88.2359639)
        self.declare_parameter('desired_speed', 2.0)
        self.declare_parameter('max_acceleration', 0.5)
        self.declare_parameter('waypoints_csv', '')
        self.declare_parameter('require_pacmod_enable', True)
        self.declare_parameter('vehicle_name', '')

        # torch-MPPI defaults (adapted from the adapt repo)
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
        # Empty string = auto-detect: 'cuda' if available else 'cpu'.
        # Set to e.g. 'cuda:0' to force GPU, 'cpu' to force CPU.
        self.declare_parameter('mppi/device', '')

        self.declare_parameter('pid/kp', 0.6)
        self.declare_parameter('pid/ki', 0.0)
        self.declare_parameter('pid/kd', 0.1)
        self.declare_parameter('pid/wg', 10.0)
        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30.0)
        self.declare_parameter('filter/order', 4)

        p = lambda n: self.get_parameter(n).value
        self.rate_hz = float(p('rate_hz'))
        self.wheelbase = float(p('wheelbase'))
        self.offset = float(p('offset'))
        self.olat = float(p('origin_lat'))
        self.olon = float(p('origin_lon'))
        self.desired_speed = min(5.0, float(p('desired_speed')))
        self.max_accel = min(2.0, float(p('max_acceleration')))
        self.require_pacmod_enable = bool(p('require_pacmod_enable'))

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
        self._log_device()

        self.pid_speed = PID(
            kp=float(p('pid/kp')), ki=float(p('pid/ki')),
            kd=float(p('pid/kd')), wg=float(p('pid/wg')),
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
        self.pacmod_enable = False
        self.obstacles = np.zeros((0, 2))
        self._pacmod_primed = False
        self._v_cmd = 0.0

        self.create_subscription(NavSatFix, '/navsatfix', self._gnss_cb, 10)
        self.create_subscription(INSNavGeod, '/insnavgeod', self._ins_cb, 10)
        self.create_subscription(Bool, '/pacmod/enabled', self._enable_cb, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt',
                                 self._speed_cb, 10)
        self.create_subscription(Int32MultiArray, '/fusion_pedestrian_position',
                                 self._ped_cb, 10)

        self.global_pub = self.create_publisher(GlobalCmd, '/pacmod/global_cmd', 10)
        self.gear_pub = self.create_publisher(SystemCmdInt, '/pacmod/shift_cmd', 10)
        self.brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 10)
        self.turn_pub = self.create_publisher(SystemCmdInt, '/pacmod/turn_cmd', 10)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)

        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd = SystemCmdInt(command=2)
        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.turn_cmd = SystemCmdInt(command=1)
        self.steer_cmd = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)

        # --- visualization --------------------------------------------------
        self.declare_parameter('viz/frame_id', 'map')
        self.declare_parameter('viz/num_samples', 19)
        self.viz_frame = str(self.get_parameter('viz/frame_id').value)
        self.viz_num_samples = int(self.get_parameter('viz/num_samples').value)
        # Pastel rainbow — each sample gets a distinct hue at high value/low
        # saturation so they're visibly different AND clearly lighter than
        # the bold-yellow "chosen" trajectory.
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
            f'adapt_mppi_node up at {self.rate_hz:.1f} Hz, '
            f'waypoints={len(self.ref_path.xy)}, v_ref={self.desired_speed:.1f} m/s'
        )

    # --- helpers --------------------------------------------------------
    def _log_device(self):
        """Log which torch device the MPPI is actually running on."""
        try:
            import torch
            dev = self.mppi.device
            if dev.type == 'cuda':
                idx = dev.index if dev.index is not None else 0
                name = torch.cuda.get_device_name(idx)
                total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
                self.get_logger().info(
                    f'MPPI device: {dev} ({name}, {total_gb:.1f} GiB VRAM)'
                )
            else:
                self.get_logger().info(f'MPPI device: {dev} (CPU)')
        except Exception as e:
            self.get_logger().warn(f'MPPI device log failed: {e}')

    def _default_waypoints_path(self):
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

    # --- callbacks ------------------------------------------------------
    def _gnss_cb(self, msg: NavSatFix):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def _ins_cb(self, msg: INSNavGeod):
        self.heading = msg.heading

    def _enable_cb(self, msg: Bool):
        self.pacmod_enable = msg.data

    def _speed_cb(self, msg: VehicleSpeedRpt):
        self.speed = float(self.speed_filter.get_data(msg.vehicle_speed))

    def _ped_cb(self, msg: Int32MultiArray):
        """Decode ego-frame polar detections and transform to world frame."""
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
            xe = dist * math.cos(rad)     # ego x forward
            ye = dist * math.sin(rad)     # ego y left
            xw = ex + xe * math.cos(yaw) - ye * math.sin(yaw)
            yw = ey + xe * math.sin(yaw) + ye * math.cos(yaw)
            out.append((xw, yw))
        self.obstacles = np.asarray(out, dtype=float) if out else np.zeros((0, 2))

    # --- geometry -------------------------------------------------------
    def _get_gem_state(self):
        local_x, local_y, _ = geodetic2enu(self.lat, self.lon, 0.0,
                                              self.olat, self.olon, 0.0)
        yaw = heading_to_yaw(self.heading)
        x = local_x - self.offset * math.cos(yaw)
        y = local_y - self.offset * math.sin(yaw)
        return x, y, yaw

    # --- loop -----------------------------------------------------------
    def _prime_pacmod(self):
        self.global_cmd.enable = True
        self.global_cmd.clear_override = True
        self.global_pub.publish(self.global_cmd)
        self.gear_cmd.command = 3
        self.gear_pub.publish(self.gear_cmd)
        self.brake_cmd.command = 0.0
        self.brake_pub.publish(self.brake_cmd)
        self.accel_cmd.command = 0.0
        self.accel_pub.publish(self.accel_cmd)
        self.turn_cmd.command = 1
        self.turn_pub.publish(self.turn_cmd)
        self._pacmod_primed = True
        self.get_logger().warn('PACMod primed: enable + FORWARD')

    def _control_loop(self):
        if self.require_pacmod_enable and not self.pacmod_enable:
            return
        if self.lat == 0.0 and self.lon == 0.0:
            return
        if not self._pacmod_primed:
            self._prime_pacmod()

        x, y, yaw = self._get_gem_state()
        state = np.array([x, y, yaw, max(self.speed, 0.0)], dtype=float)

        u = self.mppi.update(state, self.ref_path, self.obstacles)
        delta = float(u[0])
        accel = float(u[1])

        sw_deg = front2steer(math.degrees(delta))
        self.steer_cmd.angular_position = math.radians(sw_deg)
        self.steer_pub.publish(self.steer_cmd)

        self._v_cmd = max(0.0, min(self._v_cmd + accel * (1.0 / self.rate_hz),
                                   self.desired_speed))
        now = self.get_clock().now().nanoseconds * 1e-9
        speed_err = self._v_cmd - self.speed
        if abs(speed_err) < 0.05:
            speed_err = 0.0
        throttle = self.pid_speed.get_control(now, speed_err)
        throttle = max(0.0, min(throttle, self.max_accel))

        self.accel_cmd.command = throttle
        self.brake_cmd.command = 0.0
        self.accel_pub.publish(self.accel_cmd)
        self.brake_pub.publish(self.brake_cmd)
        self.global_cmd.enable = True
        self.global_pub.publish(self.global_cmd)

        ess = self.mppi.effective_sample_count()
        self.get_logger().info(
            f'MPPI | pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}deg '
            f'v={self.speed:.2f} -> v_cmd={self._v_cmd:.2f} thr={throttle:.2f} '
            f'sw={sw_deg:.1f}deg obs={len(self.obstacles)} ESS/K={ess/self.mppi.K:.2f}',
            throttle_duration_sec=1.0,
        )

        self._publish_viz()

    # --- visualization --------------------------------------------------
    def _publish_reference_path(self):
        """Publish the loaded waypoints once, latched, so late-joining RViz
        subscribers still see the track."""
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
        """Emit chosen trajectory, top-N sampled rollouts, and obstacle
        markers after each MPPI update."""
        if getattr(self.mppi, 'last_traj', None) is None:
            return
        stamp = self.get_clock().now().to_msg()
        traj = self.mppi.last_traj           # (K, H, 4)
        w = self.mppi.last_weights           # (K,)
        K, H, _ = traj.shape

        # --- chosen trajectory: weighted-mean rollout (what U tracks) ---
        mean_traj = self.mppi.last_mean_traj  # (H, 4)
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

        # --- sampled rollouts: top-N by weight, each a distinct pastel
        #     hue so they're visibly different from each other AND
        #     clearly lighter than the bold chosen path ----------------
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
                p = Point()
                p.x = float(traj[k, h, 0])
                p.y = float(traj[k, h, 1])
                p.z = 0.0
                m.points.append(p)
            samples.markers.append(m)
        self.viz_samples_pub.publish(samples)

        # --- obstacles: translucent cylinder at clearance radius --------
        obs_msg = MarkerArray()
        clear2 = Marker()
        clear2.header.frame_id = self.viz_frame
        clear2.header.stamp = stamp
        clear2.action = Marker.DELETEALL
        obs_msg.markers.append(clear2)
        r = float(self.mppi.clearance)
        for i, (ox, oy) in enumerate(self.obstacles):
            m = Marker()
            m.header.frame_id = self.viz_frame
            m.header.stamp = stamp
            m.ns = 'obstacles'
            m.id = i + 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(ox)
            m.pose.position.y = float(oy)
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
    node = AdaptMPPINode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
