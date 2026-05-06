#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import message_filters
from collections import deque
import numpy as np
import math


def _hue_to_rgb(h):
    """Tiny hue -> RGB conversion (S=V=1) so each track gets a distinct color."""
    import colorsys
    return colorsys.hsv_to_rgb(h, 0.85, 0.95)


def _moving_average(points, window):
    """Centered moving-average smoother over an iterable of (x, y).

    window=1 is a no-op. Edges shrink the window so the line still starts/ends
    at the first/last sample.
    """
    if window <= 1 or not points:
        return list(points)
    pts = list(points)
    n = len(pts)
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        sx = 0.0
        sy = 0.0
        for j in range(lo, hi):
            sx += pts[j][0]
            sy += pts[j][1]
        c = hi - lo
        out.append((sx / c, sy / c))
    return out


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.declare_parameter('matching_threshold', 2.0)  # meters
        self.declare_parameter('lidar_distance_weight', 0.8)
        self.declare_parameter('camera_distance_weight', 0.2)
        self.declare_parameter('lidar_direction_weight', 0.3)
        self.declare_parameter('camera_direction_weight', 0.7)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)  # seconds

        self.matching_threshold = self.get_parameter('matching_threshold').value
        self.lidar_dist_weight = self.get_parameter('lidar_distance_weight').value
        self.camera_dist_weight = self.get_parameter('camera_distance_weight').value
        self.lidar_dir_weight = self.get_parameter('lidar_direction_weight').value
        self.camera_dir_weight = self.get_parameter('camera_direction_weight').value
        queue_size = self.get_parameter('sync_queue_size').value
        slop = self.get_parameter('sync_slop').value

        self.lidar_sub = message_filters.Subscriber(
            self,
            Int32MultiArray,
            '/lidar_pedestrian_position'
        )
        self.camera_sub = message_filters.Subscriber(
            self,
            Int32MultiArray,
            '/rgbd_pedestrian_position'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.camera_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=True
        )
        self.ts.registerCallback(self.fusion_callback)

        # Publisher for fused detections
        self.fusion_pub = self.create_publisher(
            Int32MultiArray,
            '/fusion_pedestrian_position',
            10
        )

        # ---- Pedestrian path tracking ----
        self.declare_parameter('path_max_points', 200)
        self.declare_parameter('path_clear_after_no_detect_frames', 20)
        self.declare_parameter('path_frame_id', 'base_footprint')
        # Greedy multi-pedestrian tracker for per-track path history.
        self.declare_parameter('track_match_distance', 1.5)  # meters
        # Frames without a match before a track is dropped (and its LINE_STRIP
        # is DELETEd). Small values make paths disappear quickly when a
        # pedestrian leaves frame at the cost of brief flicker on missed
        # detections. ~3 frames @ 10 Hz ≈ 0.3 s.
        self.declare_parameter('track_max_age', 3)
        # Marker lifetime in seconds. Acts as a safety net so RViz auto-clears
        # a stale path even if the explicit DELETE marker is dropped or the
        # node stops publishing. Should be > one fusion-callback period.
        self.declare_parameter('track_marker_lifetime_sec', 0.5)
        # EMA blend on track XY; smaller = smoother, slower to react.
        # Matches the diffusion tracker's smooth_alpha pattern.
        self.declare_parameter('track_smooth_alpha', 0.15)
        # Centered moving-average window applied to the visualization path
        # only (does not affect tracker matching). 1 = off; bigger = smoother.
        self.declare_parameter('path_smoothing_window', 7)
        # Cap each pedestrian's stored history to the most recent N metres
        # of arc-length. Prevents the LINE_STRIP marker (and the diffusion
        # node's history window) from accumulating stale points while the
        # pedestrian is moving across the scene. 0 = disabled.
        self.declare_parameter('path_max_arc_length_m', 3.0)
        self.path_max_points = int(self.get_parameter('path_max_points').value)
        self.path_clear_after = int(self.get_parameter('path_clear_after_no_detect_frames').value)
        self.path_frame_id = self.get_parameter('path_frame_id').get_parameter_value().string_value
        self.track_match_dist = float(self.get_parameter('track_match_distance').value)
        self.track_max_age = int(self.get_parameter('track_max_age').value)
        self.track_marker_lifetime_sec = float(
            self.get_parameter('track_marker_lifetime_sec').value
        )
        self.track_smooth_alpha = float(self.get_parameter('track_smooth_alpha').value)
        self.path_smoothing_window = max(1, int(self.get_parameter('path_smoothing_window').value))
        self.path_max_arc_length_m = float(
            self.get_parameter('path_max_arc_length_m').value
        )
        # Single-closest path (back-compat).
        self.path_history = deque(maxlen=self.path_max_points)
        self.path_filtered_xy = None  # EMA state for back-compat single-closest path
        self.no_detect_streak = 0
        self.path_pub = self.create_publisher(Marker, '/fusion_pedestrian_path', 10)
        # Per-pedestrian paths: track_id -> {'path': deque, 'age': int, 'last_xy': (x,y)}
        self.tracks = {}
        self.next_track_id = 0
        self.paths_pub = self.create_publisher(MarkerArray, '/fusion_pedestrian_paths', 10)
        self._known_track_ids = set()  # for emitting DELETE markers when tracks die

        # Statistics for logging
        self.fusion_count = 0
        self.lidar_only_count = 0
        self.camera_only_count = 0

        self.get_logger().info('Sensor Fusion Node initialized')
        self.get_logger().info(f'Matching threshold: {self.matching_threshold}m')
        self.get_logger().info(f'Weights - Lidar: (dist={self.lidar_dist_weight}, dir={self.lidar_dir_weight})')
        self.get_logger().info(f'Weights - Camera: (dist={self.camera_dist_weight}, dir={self.camera_dir_weight})')

    def parse_detections(self, data_array):
        detections = []
        
        if len(data_array) % 2 != 0:
            self.get_logger().warn(f'Invalid data array length: {len(data_array)}')
            return detections
        
        # Parse pairs
        for i in range(0, len(data_array), 2):
            distance = data_array[i]
            direction = data_array[i + 1]
            
            # Validate data
            if distance < 0 or direction < 0 or direction >= 360:
                self.get_logger().warn(f'Invalid detection: dist={distance}, dir={direction}')
                continue

            detections.append({
                'dist': distance,
                'deg': direction
            })

        return detections

    def polar_to_cartesian(self, distance, direction_deg):
        direction_rad = math.radians(direction_deg)
        x = distance * math.cos(direction_rad)
        y = distance * math.sin(direction_rad)
        return (x, y)

    def cartesian_to_polar(self, x, y):
        distance = math.sqrt(x**2 + y**2)
        direction_rad = math.atan2(y, x)
        direction_deg = math.degrees(direction_rad)
        # Normalize to 0-360 range
        if direction_deg < 0:
            direction_deg += 360
        return (distance, direction_deg)

    def polar_to_base_footprint(self, distance, direction_deg):
        """(dist, deg with 0=right, 90=front) -> (x, y) in base_footprint."""
        rad = math.radians(direction_deg)
        return (distance * math.sin(rad), -distance * math.cos(rad))

    def _trim_path_to_arc_length(self, path):
        """Drop oldest points from a track's path until the cumulative
        arc-length walking from the most recent point backward fits within
        ``self.path_max_arc_length_m``. ``path`` is mutated in place.

        Disabled when ``path_max_arc_length_m <= 0``. With the default 3 m,
        a pedestrian walking at 1.4 m/s retains ~2 s of history (~20 frames
        at 10 Hz), which matches the diffusion model's history window.
        """
        max_arc = self.path_max_arc_length_m
        if max_arc <= 0.0 or len(path) < 2:
            return
        # Walk from the newest point backward, summing segment lengths until
        # we exceed max_arc. Everything older than that index is stale.
        keep_from_end = 1
        cum = 0.0
        for i in range(len(path) - 1, 0, -1):
            x1, y1 = path[i]
            x0, y0 = path[i - 1]
            cum += math.hypot(x1 - x0, y1 - y0)
            if cum > max_arc:
                break
            keep_from_end += 1
        n_drop = len(path) - keep_from_end
        for _ in range(n_drop):
            path.popleft()

    def update_tracks(self, fused_detections):
        """Greedy multi-pedestrian tracker. Updates self.tracks in place."""
        # Convert detections to base_footprint XY.
        detections_xy = [
            self.polar_to_base_footprint(d['dist'], d['deg'])
            for d in fused_detections
        ]

        matched_tids = set()
        matched_dets = set()
        alpha = self.track_smooth_alpha

        # Greedy match: for each existing track, find closest unmatched detection.
        for tid, tr in list(self.tracks.items()):
            best_d = self.track_match_dist
            best_idx = -1
            tx, ty = tr['last_xy']
            for i, (dx, dy) in enumerate(detections_xy):
                if i in matched_dets:
                    continue
                d = math.hypot(dx - tx, dy - ty)
                if d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx >= 0:
                dx, dy = detections_xy[best_idx]
                # EMA-smoothed XY: filters out polar-grid quantization.
                sx = alpha * dx + (1.0 - alpha) * tx
                sy = alpha * dy + (1.0 - alpha) * ty
                tr['path'].append((sx, sy))
                tr['last_xy'] = (sx, sy)
                tr['age'] = 0
                self._trim_path_to_arc_length(tr['path'])
                matched_tids.add(tid)
                matched_dets.add(best_idx)

        # Unmatched detections start new tracks.
        for i, xy in enumerate(detections_xy):
            if i in matched_dets:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                'path': deque([xy], maxlen=self.path_max_points),
                'age': 0,
                'last_xy': xy,
            }
            matched_tids.add(tid)

        # Age + prune unmatched tracks.
        for tid in list(self.tracks.keys()):
            if tid not in matched_tids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.track_max_age:
                    del self.tracks[tid]

    def publish_paths_marker_array(self):
        """One LINE_STRIP marker per active track + DELETE markers for vanished ones."""
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        active = set(self.tracks.keys())
        # DELETE markers for tracks we previously knew about but that are gone.
        for tid in self._known_track_ids - active:
            del_m = Marker()
            del_m.header.frame_id = self.path_frame_id
            del_m.header.stamp = stamp
            del_m.ns = 'fusion_pedestrian_paths'
            del_m.id = tid
            del_m.action = Marker.DELETE
            ma.markers.append(del_m)

        # ADD/MODIFY markers for active tracks.
        # Color by track id (stable hue per track).
        lt_total = max(0.0, self.track_marker_lifetime_sec)
        lt_sec = int(lt_total)
        lt_nsec = int((lt_total - lt_sec) * 1e9)
        for tid, tr in self.tracks.items():
            m = Marker()
            m.header.frame_id = self.path_frame_id
            m.header.stamp = stamp
            m.ns = 'fusion_pedestrian_paths'
            m.id = tid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD if tr['path'] else Marker.DELETE
            m.scale.x = 0.08
            m.color.a = 1.0
            # Distinct hue per track id (HSV-like via mod cycle).
            hue = (tid * 47) % 360 / 360.0
            r, g, b = _hue_to_rgb(hue)
            m.color.r, m.color.g, m.color.b = r, g, b
            m.pose.orientation.w = 1.0
            # Auto-expire in RViz if not refreshed; safety net for dropped
            # DELETE markers or a stalled fusion callback.
            m.lifetime.sec = lt_sec
            m.lifetime.nanosec = lt_nsec
            for x, y in _moving_average(tr['path'], self.path_smoothing_window):
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.05
                m.points.append(p)
            ma.markers.append(m)

        self.paths_pub.publish(ma)
        self._known_track_ids = active

    def publish_path_marker(self):
        marker = Marker()
        marker.header.frame_id = self.path_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'fusion_pedestrian_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD if self.path_history else Marker.DELETE
        marker.scale.x = 0.08  # line width (m)
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        for x, y in _moving_average(self.path_history, self.path_smoothing_window):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05  # just above ground for visibility
            marker.points.append(p)
        self.path_pub.publish(marker)

    def euclidean_distance(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx**2 + dy**2)

    def match_detections(self, lidar_detections, camera_detections):
        matched_pairs = []
        lidar_matched = [False] * len(lidar_detections)
        camera_matched = [False] * len(camera_detections)
        lidar_cartesian = []
        for det in lidar_detections:
            x, y = self.polar_to_cartesian(det['dist'], det['deg'])
            lidar_cartesian.append((x, y))

        camera_cartesian = []
        for det in camera_detections:
            x, y = self.polar_to_cartesian(det['dist'], det['deg'])
            camera_cartesian.append((x, y))
            
        for i, lidar_pos in enumerate(lidar_cartesian):
            if lidar_matched[i]:
                continue

            best_match_idx = -1
            best_distance = float('inf')

            for j, camera_pos in enumerate(camera_cartesian):
                if camera_matched[j]:
                    continue

                dist = self.euclidean_distance(lidar_pos, camera_pos)

                if dist < self.matching_threshold and dist < best_distance:
                    best_distance = dist
                    best_match_idx = j

            if best_match_idx >= 0:
                matched_pairs.append((
                    lidar_detections[i],
                    camera_detections[best_match_idx]
                ))
                lidar_matched[i] = True
                camera_matched[best_match_idx] = True

        lidar_only = [lidar_detections[i] for i in range(len(lidar_detections))
                     if not lidar_matched[i]]
        camera_only = [camera_detections[j] for j in range(len(camera_detections))
                      if not camera_matched[j]]

        return matched_pairs, lidar_only, camera_only

    def fuse_matched_pair(self, lidar_det, camera_det):
        """
        Fuse a matched Lidar-Camera pair using weighted averaging.

        Strategy:
            - Distance: Trust Lidar more (0.8 Lidar, 0.2 Camera)
            - Direction: Trust Camera more (0.3 Lidar, 0.7 Camera)

        Args:
            lidar_det: Lidar detection dict
            camera_det: Camera detection dict

        Returns:
            Fused detection dict

        Math:
            fused_distance = w_lidar * lidar_dist + w_camera * camera_dist
            fused_direction = w_lidar * lidar_deg + w_camera * camera_deg
        """
        # Weighted average for distance
        fused_distance = (self.lidar_dist_weight * lidar_det['dist'] +
                         self.camera_dist_weight * camera_det['dist'])

        # Weighted average for direction (handle angle wrapping)
        lidar_deg = lidar_det['deg']
        camera_deg = camera_det['deg']

        # Handle angle wrapping (e.g., 10� and 350� should average to 0�, not 180�)
        angle_diff = abs(lidar_deg - camera_deg)
        if angle_diff > 180:
            # Angles wrap around 360�
            if lidar_deg > camera_deg:
                camera_deg += 360
            else:
                lidar_deg += 360

        fused_direction = (self.lidar_dir_weight * lidar_deg +
                          self.camera_dir_weight * camera_deg)

        # Normalize back to 0-360 range
        fused_direction = fused_direction % 360

        # Round and cast to int
        return {
            'dist': int(round(fused_distance)),
            'deg': int(round(fused_direction))
        }

    def fusion_callback(self, lidar_msg, camera_msg):
        # Step 1: Parse detections
        lidar_detections = self.parse_detections(lidar_msg.data)
        camera_detections = self.parse_detections(camera_msg.data)

        self.get_logger().debug(
            f'Received {len(lidar_detections)} Lidar, {len(camera_detections)} Camera detections'
        )

        # Handle empty arrays gracefully
        if len(lidar_detections) == 0 and len(camera_detections) == 0:
            # No detections from either sensor - publish empty array
            fused_msg = Int32MultiArray()
            fused_msg.data = []
            self.fusion_pub.publish(fused_msg)
            self.get_logger().debug('No detections from either sensor')
            self.no_detect_streak += 1
            if self.no_detect_streak >= self.path_clear_after:
                self.path_history.clear()
                self.path_filtered_xy = None
            self.update_tracks([])
            self.publish_path_marker()
            self.publish_paths_marker_array()
            return

        # Step 2 & 3: Match detections
        matched_pairs, lidar_only, camera_only = self.match_detections(
            lidar_detections,
            camera_detections
        )

        # Update statistics
        self.fusion_count += len(matched_pairs)
        self.lidar_only_count += len(lidar_only)
        self.camera_only_count += len(camera_only)

        # Step 4: Fuse matched pairs
        fused_detections = []

        for lidar_det, camera_det in matched_pairs:
            fused_det = self.fuse_matched_pair(lidar_det, camera_det)
            fused_detections.append(fused_det)

        # Include Lidar-only detections (valid obstacles in the dark)
        fused_detections.extend(lidar_only)

        # Include Camera-only detections (Lidar may miss reflections)
        for cam_det in camera_only:
            fused_detections.append(cam_det)
            self.get_logger().debug(
                f'Camera-only detection at {cam_det["dist"]}m, {cam_det["deg"]}� '
                f'(Lidar may have missed reflection)'
            )

        # Step 5: Flatten and publish
        fused_array = []
        for det in fused_detections:
            fused_array.append(det['dist'])
            fused_array.append(det['deg'])

        fused_msg = Int32MultiArray()
        fused_msg.data = fused_array
        self.fusion_pub.publish(fused_msg)

        # Append the closest fused detection to the single-closest path history (back-compat).
        if fused_detections:
            closest = min(fused_detections, key=lambda d: d['dist'])
            x, y = self.polar_to_base_footprint(closest['dist'], closest['deg'])
            if self.path_filtered_xy is None:
                self.path_filtered_xy = (x, y)
            else:
                a = self.track_smooth_alpha
                fx, fy = self.path_filtered_xy
                self.path_filtered_xy = (a * x + (1.0 - a) * fx,
                                         a * y + (1.0 - a) * fy)
            self.path_history.append(self.path_filtered_xy)
            self.no_detect_streak = 0
        else:
            self.no_detect_streak += 1
            if self.no_detect_streak >= self.path_clear_after:
                self.path_history.clear()
                self.path_filtered_xy = None
        self.publish_path_marker()

        # Multi-pedestrian tracker: per-track paths.
        self.update_tracks(fused_detections)
        self.publish_paths_marker_array()

        # Logging
        self.get_logger().info(
            f'Published {len(fused_detections)} fused detections: '
            f'{len(matched_pairs)} matched, {len(lidar_only)} lidar-only, '
            f'{len(camera_only)} camera-only'
        )

        # Periodic statistics
        total_processed = self.fusion_count + self.lidar_only_count + self.camera_only_count
        if total_processed > 0 and total_processed % 50 == 0:
            self.get_logger().info(
                f'Statistics - Total: {total_processed}, '
                f'Fused: {self.fusion_count}, '
                f'Lidar-only: {self.lidar_only_count}, '
                f'Camera-only: {self.camera_only_count}'
            )


def main(args=None):

    rclpy.init(args=args)

    try:
        node = SensorFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in sensor fusion node: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
