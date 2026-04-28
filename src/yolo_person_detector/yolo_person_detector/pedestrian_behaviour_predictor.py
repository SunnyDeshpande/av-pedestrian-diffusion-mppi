#!/usr/bin/env python3
# ------- Imports -------
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist, TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Int32MultiArray, Float64

from pacmod2_msgs.msg import VehicleSpeedRpt

from tf2_ros import TransformBroadcaster

import numpy as np


class PedestrianBehaviorPredictor(Node):
    def __init__(self):
        super().__init__('pedestrian_behavior_predictor')

        # ------- Parameters -------
        self.declare_parameter('prediction_time', 5.0)  # seconds
        self.declare_parameter('prediction_points', 20)
        self.declare_parameter('collision_distance_threshold', 1.0)  # meters
        self.declare_parameter('max_missing_frames', 10)
        self.declare_parameter('max_path_len', 100)
        self.declare_parameter('smooth_alpha', 0.6)
        self.declare_parameter('max_prediction_distance', 15.0)  # meters

        self.prediction_time = float(self.get_parameter('prediction_time').value)
        self.prediction_points = int(self.get_parameter('prediction_points').value)
        self.collision_distance_threshold = float(self.get_parameter('collision_distance_threshold').value)
        self.max_missing_frames = int(self.get_parameter('max_missing_frames').value)
        self.max_path_len = int(self.get_parameter('max_path_len').value)
        self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)
        self.max_prediction_distance = float(self.get_parameter('max_prediction_distance').value)

        # ------- ROS I/O -------
        # ------- Inputs -------
        self.sub_ped = self.create_subscription(
            Int32MultiArray, 'fusion_pedestrian_position', self.pedestrian_cb, 10
        )
        self.sub_vehicle = self.create_subscription(
            VehicleSpeedRpt, 'vehicle_rpt', self.vehicle_cb, 10
        )

        # ------- Outputs -------
        self.pub_ped_motion = self.create_publisher(Twist, 'pedestrian_motion', 10)
        self.pub_ped_ttc = self.create_publisher(Float64, 'pedestrian_ttc', 10)

        # ------- RViz markers -------
        self.pub_person_marker = self.create_publisher(Marker, 'fusion_person_marker', 10)
        self.pub_path_marker = self.create_publisher(Marker, 'person_path', 10)
        self.pub_prediction_marker = self.create_publisher(Marker, 'person_prediction', 10)
        self.pub_car_path_marker = self.create_publisher(Marker, 'car_path', 10)
        self.pub_camera_marker = self.create_publisher(Marker, 'camera_marker', 10)

        # ------- TF broadcaster -------
        self.tf_broadcaster = TransformBroadcaster(self)
        self.publish_camera_transform()

        # ------- State -------
        self.tracks = {}  # tid -> dict
        self.next_track_id = 0

        self.vehicle_speed = 0.0  # m/s
        self.vehicle_speed_valid = False

        self.get_logger().info("Pedestrian TTC Predictor ready (fusion distance+direction input + camera marker).")

    # -------  base_link -> oak_rgb_camera_optical_frame TF -------
    def publish_camera_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'oak_rgb_camera_optical_frame'
        t.transform.translation.x = 0.535
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.683
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.7071080798594738
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.7071054825112363
        self.tf_broadcaster.sendTransform(t)

    # ------- Vehicle state callback -------
    def vehicle_cb(self, msg: VehicleSpeedRpt):
        if msg.vehicle_speed_valid:
            self.vehicle_speed = float(msg.vehicle_speed)
            self.vehicle_speed_valid = True
        else:
            self.vehicle_speed_valid = False

    # ------- Tracking + prediction helpers -------
    def _smooth_path_and_predict(self, path_3d, times):
        if len(path_3d) < 3 or len(times) < 3:
            return path_3d, []

        path_array = np.array(path_3d)

        # Spike removal
        filtered_path = [path_array[0]]
        spike_threshold = 0.5  # meters

        for i in range(1, len(path_array) - 1):
            prev_point = path_array[i - 1]
            curr_point = path_array[i]
            next_point = path_array[i + 1]
            expected = (prev_point + next_point) / 2.0
            deviation = np.linalg.norm(curr_point - expected)
            if deviation > spike_threshold:
                median_point = np.median([prev_point, curr_point, next_point], axis=0)
                filtered_path.append(median_point)
            else:
                filtered_path.append(curr_point)

        filtered_path.append(path_array[-1])
        filtered_array = np.array(filtered_path)

        # Moving average smoothing
        window = min(7, len(filtered_array))
        smoothed = []
        for i in range(len(filtered_array)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(filtered_array), i + window // 2 + 1)
            smoothed.append(np.mean(filtered_array[start_idx:end_idx], axis=0))

        # Velocity estimation using timestamps w non-constant dt
        N = len(path_3d)
        start_index = max(1, N - 15)
        velocities = []
        for i in range(start_index, N):
            p_prev = np.array(path_3d[i - 1])
            p_curr = np.array(path_3d[i])
            t_prev = times[i - 1]
            t_curr = times[i]
            dt = t_curr - t_prev
            if dt <= 0.0:
                continue
            v = (p_curr - p_prev) / dt
            velocities.append(v)

        if len(velocities) == 0:
            return smoothed, []

        velocities_array = np.array(velocities)
        mean_vel = np.mean(velocities_array, axis=0)
        std_vel = np.std(velocities_array, axis=0) + 1e-6

        filtered_velocities = []
        for vel in velocities_array:
            if np.all(np.abs(vel - mean_vel) < 2 * std_vel):
                filtered_velocities.append(vel)

        if len(filtered_velocities) > 0:
            avg_velocity = np.mean(filtered_velocities, axis=0)
        else:
            avg_velocity = mean_vel

        # flatten vertical component (z = index 2)
        avg_velocity[2] = 0.0

        speed = np.linalg.norm(avg_velocity)
        max_speed = 3.0
        if speed > max_speed:
            avg_velocity *= (max_speed / speed)

        avg_acceleration = np.zeros(3)

        last_pos = np.array(smoothed[-1])
        predicted = []

        # flatten to ground plane z = constant
        ground_height = last_pos[2]
        ground_start_pos = last_pos.copy()
        ground_start_pos[2] = ground_height

        
        for i in range(1, self.prediction_points + 1):
            t = (i / self.prediction_points) * self.prediction_time
            pred_pos = ground_start_pos + avg_velocity * t + 0.5 * avg_acceleration * t * t
            pred_pos[2] = ground_height

            distance = np.linalg.norm(pred_pos[:2] - ground_start_pos[:2])
            if distance > self.max_prediction_distance:
                direction = (pred_pos - ground_start_pos) / (distance + 1e-9)
                pred_pos = ground_start_pos + direction * self.max_prediction_distance
                pred_pos[2] = ground_height
                predicted.append(tuple(pred_pos))
                break

            predicted.append(tuple(pred_pos))

        return smoothed, predicted

    def _update_tracks(self, detections, t_now):
        deleted_ids = []
        max_dist2 = 2.0 ** 2  # 2 m in XY-plane
        used_tracks = set()

        for det in detections:
            X = det['x']  # forward
            Y = det['y']  # lateral
            Z = det['z']  # up (0)

            best_id = None
            best_dist2 = max_dist2

            for tid, tr in self.tracks.items():
                dx = X - tr['x']
                dy = Y - tr['y']
                d2 = dx * dx + dy * dy
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_id = tid

            if best_id is None:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    'x': X,
                    'y': Y,
                    'z': Z,
                    'path_3d': [(X, Y, Z)],
                    'times': [t_now],
                    'smoothed_path': [(X, Y, Z)],
                    'predicted_path': [],
                    'missed': 0,
                }
                used_tracks.add(tid)
            else:
                tr = self.tracks[best_id]
                alpha = self.smooth_alpha
                tr['x'] = alpha * X + (1.0 - alpha) * tr['x']
                tr['y'] = alpha * Y + (1.0 - alpha) * tr['y']
                tr['z'] = alpha * Z + (1.0 - alpha) * tr['z']

                tr['path_3d'].append((tr['x'], tr['y'], tr['z']))
                tr['times'].append(t_now)

                if len(tr['path_3d']) > self.max_path_len:
                    tr['path_3d'].pop(0)
                    tr['times'].pop(0)

                smoothed, predicted = self._smooth_path_and_predict(tr['path_3d'], tr['times'])
                tr['smoothed_path'] = smoothed
                tr['predicted_path'] = predicted
                tr['missed'] = 0
                used_tracks.add(best_id)

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_missing_frames:
                    deleted_ids.append(tid)
                    del self.tracks[tid]

        return deleted_ids

    def _time_to_collision_with_car(self, tr):
        # Debug prints
        print("Predicted Path:", tr['predicted_path'])
        print("Vehicle Speed Valid:", self.vehicle_speed_valid)
        print("Vehicle Speed:", self.vehicle_speed)

        if len(tr['predicted_path']) == 0:
            return float('inf'), float('inf'), None
        if not self.vehicle_speed_valid or self.vehicle_speed <= 0.01:
            return float('inf'), float('inf'), None

        v_car = self.vehicle_speed           # m/s, along +x
        dt_pred = self.prediction_time / float(self.prediction_points)

        min_distance = float('inf')
        collision_time = float('inf')
        collision_point = None

        for i, p in enumerate(tr['predicted_path']):
            px, py, pz = p  # px = forward, py = lateral, pz ≈ 0
            t = (i + 1) * dt_pred

            car_x = v_car * t
            car_pos = np.array([car_x, 0.0, 0.0])
            ped_pos = np.array([px, py, 0.0])

            dist = np.linalg.norm(ped_pos - car_pos)
            if dist < min_distance:
                min_distance = dist
                collision_point = p
            if dist < self.collision_distance_threshold and collision_time == float('inf'):
                collision_time = t
                collision_point = p

        return collision_time, min_distance, collision_point

    def _get_closest_collision_pedestrian(self, collision_peds):
        if not collision_peds:
            return None, None
        min_ttc = float('inf')
        closest_tid = None
        for tid, data in collision_peds.items():
            if data['ttc'] < min_ttc:
                min_ttc = data['ttc']
                closest_tid = tid
        if closest_tid is None:
            return None, None
        return closest_tid, collision_peds[closest_tid]

    # ------- Main Pedestrian callback -------
    def pedestrian_cb(self, msg: Int32MultiArray):
        """
        Input: /fusion_pedestrian_position (Int32MultiArray)
          data[0] = distance [m] (int)
          data[1] = direction [deg] CCW from 0° = right side of vehicle

        We convert to a 3D point in base_link-style frame:
          - x: forward (+)
          - y: left (+)
          - z: up (0 here; ground plane)
        """
        # keep TF updated
        self.publish_camera_transform()

        now = self.get_clock().now()
        t_now = float(now.nanoseconds) * 1e-9

        detections = []
        if len(msg.data) >= 2:
            dist = float(msg.data[0])
            direction_deg = float(msg.data[1])
            theta = np.deg2rad(direction_deg)

            # 0° = right side -> negative lateral y
            # 90° = forward -> +x
            # 180° = left -> +y
            # 270° = backward -> -x
            x = dist * np.sin(theta)     # forward
            y = -dist * np.cos(theta)    # lateral
            z = 0.0                      # up 

            detections.append({'x': x, 'y': y, 'z': z})

        deleted_ids = self._update_tracks(detections, t_now)

        stamp_msg = now.to_msg()

        # Clear RViz markers for disappeared tracks
        for tid in deleted_ids:
            for ns in ["person", "person_path", "person_prediction"]:
                m = Marker()
                m.header.frame_id = "base_link"
                m.header.stamp = stamp_msg
                m.ns = ns
                m.id = tid
                m.action = Marker.DELETE
                if ns == "person":
                    self.pub_person_marker.publish(m)
                elif ns == "person_path":
                    self.pub_path_marker.publish(m)
                else:
                    self.pub_prediction_marker.publish(m)

        # Collision analysis
        collision_peds = {}
        for tid, tr in self.tracks.items():
            if len(tr['predicted_path']) == 0:
                continue

            ttc, min_dist, entry_pt = self._time_to_collision_with_car(tr)

            if ttc < float('inf'):
                collision_peds[tid] = {
                    'ttc': ttc,
                    'min_distance': min_dist,
                    'entry_point': entry_pt,
                    'track': tr,
                }

        # Choose closest pedestrian for outputs
        primary_tr = None
        ttc_value = float('inf')

        if collision_peds:
            # Predicted collision
            closest_tid, closest_ped = self._get_closest_collision_pedestrian(collision_peds)
            if closest_ped is not None:
                primary_tr = closest_ped['track']
                ttc_value = closest_ped['ttc']
                self.get_logger().warn(
                    f"[TTC] collision risk with ped {closest_tid}: "
                    f"TTC={closest_ped['ttc']:.2f}s, "
                    f"min_dist={closest_ped['min_distance']:.2f} m"
                )
        else:
            # No predicted collision, pick nearest pedestrian (in XY plane) for motion output
            min_d = float('inf')
            for tid, tr in self.tracks.items():
                d = np.sqrt(tr['x'] ** 2 + tr['y'] ** 2)
                if d < min_d:
                    min_d = d
                    primary_tr = tr
            ttc_value = float('inf')

        # Publish motion + TTC if there is primary track
        if primary_tr is not None:
            # vehicle_footprint-style:
            #   linear.x = forward (m)
            #   linear.y = lateral (m)
            motion_msg = Twist()
            motion_msg.linear.x = float(primary_tr['x'])  # forward
            motion_msg.linear.y = float(primary_tr['y'])  # lateral
            motion_msg.linear.z = 0.0
            motion_msg.angular.x = 0.0
            motion_msg.angular.y = 0.0
            motion_msg.angular.z = 0.0
            self.pub_ped_motion.publish(motion_msg)

            ttc_msg = Float64()
            ttc_msg.data = float(ttc_value)  # seconds, inf if no collision
            self.pub_ped_ttc.publish(ttc_msg)

        # RViz markers
        self._publish_markers(stamp_msg)

    # ------- Marker publishing -------
    def _publish_markers(self, stamp_msg):
        # Pedestrian markers in base_link
        for tid, tr in self.tracks.items():
            x, y, z = tr['x'], tr['y'], tr['z']

            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = stamp_msg
            m.ns = "person"
            m.id = tid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color.a = 1.0
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            self.pub_person_marker.publish(m)

            path = Marker()
            path.header.frame_id = "base_link"
            path.header.stamp = stamp_msg
            path.ns = "person_path"
            path.id = tid
            path.type = Marker.LINE_STRIP
            path.action = Marker.ADD
            path.pose.orientation.w = 1.0
            path.scale.x = 0.05
            path.color.a = 1.0
            path.color.r = 1.0
            path.color.g = 1.0
            path.color.b = 0.0
            if 'smoothed_path' in tr and len(tr['smoothed_path']) > 0:
                path.points = [
                    Point(x=float(px), y=float(py), z=float(pz))
                    for (px, py, pz) in tr['smoothed_path']
                ]
            else:
                path.points = [
                    Point(x=float(px), y=float(py), z=float(pz))
                    for (px, py, pz) in tr['path_3d']
                ]
            self.pub_path_marker.publish(path)

            if len(tr['predicted_path']) > 0:
                pred = Marker()
                pred.header.frame_id = "base_link"
                pred.header.stamp = stamp_msg
                pred.ns = "person_prediction"
                pred.id = tid
                pred.type = Marker.LINE_STRIP
                pred.action = Marker.ADD
                pred.pose.orientation.w = 1.0
                pred.scale.x = 0.15
                pred.color.a = 1.0
                pred.color.r = 1.0
                pred.color.g = 0.0
                pred.color.b = 0.0
                pred.points = [
                    Point(x=float(px), y=float(py), z=float(pz))
                    for (px, py, pz) in tr['predicted_path']
                ]
                self.pub_prediction_marker.publish(pred)

        # Car path in base_link frame
        car_path = Marker()
        car_path.header.frame_id = "base_link"
        car_path.header.stamp = stamp_msg
        car_path.ns = "car_path"
        car_path.id = 0
        car_path.type = Marker.LINE_STRIP
        car_path.action = Marker.ADD
        car_path.pose.orientation.w = 1.0
        car_path.scale.x = 0.1
        car_path.color.a = 1.0
        car_path.color.r = 0.0
        car_path.color.g = 0.0
        car_path.color.b = 1.0

        points = []
        if self.vehicle_speed_valid and self.vehicle_speed > 0.01:
            dt_pred = self.prediction_time / float(self.prediction_points)
            for i in range(self.prediction_points + 1):
                t = i * dt_pred
                x = self.vehicle_speed * t
                points.append(Point(x=float(x), y=0.0, z=0.0))
        else:
            points.append(Point(x=0.0, y=0.0, z=0.0))
            points.append(Point(x=0.1, y=0.0, z=0.0))

        car_path.points = points
        self.pub_car_path_marker.publish(car_path)

        # Camera marker in oak_rgb_camera_optical_frame
        cam = Marker()
        cam.header.frame_id = "oak_rgb_camera_optical_frame"
        cam.header.stamp = stamp_msg
        cam.ns = "camera"
        cam.id = 0
        cam.type = Marker.CUBE
        cam.action = Marker.ADD
        cam.pose.position.x = 0.0
        cam.pose.position.y = 0.0
        cam.pose.position.z = 0.0
        cam.pose.orientation.w = 1.0
        cam.scale.x = 0.2
        cam.scale.y = 0.2
        cam.scale.z = 0.2
        cam.color.a = 1.0
        cam.color.r = 0.0
        cam.color.g = 0.0
        cam.color.b = 1.0
        self.pub_camera_marker.publish(cam)

# ------- Main -------
def main(args=None):
    rclpy.init(args=args)
    node = PedestrianBehaviorPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Pedestrian Behaviour Predictor")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
