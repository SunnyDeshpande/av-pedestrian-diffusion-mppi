#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from detected_object_msgs.msg import DetectedObject, DetectedObjectArray
from geometry_msgs.msg import Vector3
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header, Int32
from std_msgs.msg import Int32MultiArray
import colorsys
import math

class LidarObjectDetector(Node):
    def __init__(self):
        super().__init__('lidar_preprocessor')

        self.declare_parameter('crop_min_x', 0.0)
        self.declare_parameter('crop_max_x', 0.0)
        self.declare_parameter('crop_min_y', 0.0)
        self.declare_parameter('crop_max_y', 0.0)
        self.declare_parameter('crop_min_z', 0.0)
        self.declare_parameter('crop_max_z', 0.0)
        self.declare_parameter('voxel_size', 0.0)
        self.declare_parameter('sor_nb_neighbors', 0)
        self.declare_parameter('sor_std_ratio', 0.0)
        self.declare_parameter('ground_z_threshold', 0.0)

        self.declare_parameter('dbscan_eps', 0.0)
        self.declare_parameter('dbscan_min_points', 0)
        self.declare_parameter('track_max_distance', 0.0)
        self.declare_parameter('track_max_age', 0)
        self.declare_parameter('track_min_hits', 0)
        self.declare_parameter('ema_alpha', 0.0)

        # HUMAN & MOTION PARAMS
        self.declare_parameter('human_height_min', 0.0)  
        self.declare_parameter('human_height_max', 0.0)
        self.declare_parameter('human_width_max', 0.0)   
        self.declare_parameter('human_depth_max', 0.0)   
        self.declare_parameter('human_ratio_min', 0.0)   
        self.declare_parameter('human_footprint_max', 0.0) 
        self.declare_parameter('human_volume_min', 0.0) 
        self.declare_parameter('human_volume_max', 0.0)
        self.declare_parameter('human_compactness_max', 0.0)
        self.declare_parameter('human_xy_flatness_min', 0.0) 
        self.declare_parameter('min_motion_threshold', 0.0) 
        self.declare_parameter('static_check_frames', 0)
        self.declare_parameter('max_intensity_avg', 0.0)

        self.sub = self.create_subscription(PointCloud2, '/ouster/points', self.callback, 10)
        self.pub_processed = self.create_publisher(PointCloud2, '/processed_points', 10)
        self.pub_clustered = self.create_publisher(PointCloud2, '/clustered_points', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/cluster_markers', 10)
        self.pub_objects = self.create_publisher(DetectedObjectArray, '/detected_objects', 10)
        self.pub_pedestrian_pos = self.create_publisher(Int32MultiArray, '/lidar_pedestrian_position', 10)
        self.pub_human_debug = self.create_publisher(MarkerArray, '/human_debug_info', 10)

        self.tracker = SimpleClusterTracker(self)
        self.get_logger().info("Lidar Object Detector with Human Tracking STARTED")

    def callback(self, msg: PointCloud2):
        try:
            # loading point cloud
            field_names = [f.name for f in msg.fields]
            has_intensity = 'intensity' in field_names
            read_fields = ('x', 'y', 'z', 'intensity') if has_intensity else ('x', 'y', 'z')
            
            cloud_gen = pc2.read_points(msg, field_names=read_fields, skip_nans=True)
            pts_list = list(cloud_gen)
            if not pts_list: return

            pts_array = np.array(pts_list)
            
            if pts_array.dtype.names:
                x = pts_array['x']
                y = pts_array['y']
                z = pts_array['z']
                points_np = np.column_stack((x, y, z)).astype(np.float64)
            else:
                points_np = pts_array[:, 0:3].astype(np.float64)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)

            # Cropping
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                [self.get_parameter('crop_min_x').value, self.get_parameter('crop_min_y').value, self.get_parameter('crop_min_z').value],
                [self.get_parameter('crop_max_x').value, self.get_parameter('crop_max_y').value, self.get_parameter('crop_max_z').value]
            )
            pcd = pcd.crop(bbox)
            if len(pcd.points) < 10: return

            # Downsampling
            pcd = pcd.voxel_down_sample(self.get_parameter('voxel_size').value)

            # Statistical Outlier Removal
            try:
                if len(pcd.points) > self.get_parameter('sor_nb_neighbors').value:
                    _, ind = pcd.remove_statistical_outlier(
                        nb_neighbors=self.get_parameter('sor_nb_neighbors').value,
                        std_ratio=self.get_parameter('sor_std_ratio').value
                    )
                    pcd = pcd.select_by_index(ind)
            except: pass

            # Ground Removal
            points = np.asarray(pcd.points)
            if len(points) > 0:
                points = points[points[:, 2] > self.get_parameter('ground_z_threshold').value]
            if len(points) < 10: return

            # Publish processed point cloud
            header = msg.header
            header.stamp = self.get_clock().now().to_msg()
            self.pub_processed.publish(pc2.create_cloud_xyz32(header, points.astype(np.float32)))

            # Clustering
            clusters = self.cluster_with_dbscan(pcd)
            # Tracking
            self.tracker.update(clusters, header)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def cluster_with_dbscan(self, pcd):
        eps = self.get_parameter('dbscan_eps').value
        min_pts = self.get_parameter('dbscan_min_points').value
        
        if len(pcd.points) < min_pts: return []

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))

        clusters = []
        if len(labels) == 0: return clusters

        points_np = np.asarray(pcd.points)
        for label in np.unique(labels):
            if label == -1: continue
            mask = (labels == label)
            pts = points_np[mask]
            if len(pts) < min_pts: continue
            clusters.append({'points': pts, 'centroid': pts.mean(axis=0)})
        return clusters


class SimpleClusterTracker:
    def __init__(self, node):
        self.node = node
        self.tracks = {}
        self.next_id = 0
        self.colors = {}
        self.locked_human_id = None 

    def get_color(self, tid):
        if tid not in self.colors:
            h = (tid * 37) % 360 / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
            self.colors[tid] = (r, g, b)
        return self.colors[tid]

    def update(self, clusters, header):
        centroids = [c['centroid'] for c in clusters]
        assignments = []
        used = set()

        # Track cluster
        for cent in centroids:
            best_id = None
            best_dist = self.node.get_parameter('track_max_distance').value
            for tid, tr in self.tracks.items():
                if tid in used: continue
                d = np.linalg.norm(tr['centroid'] - cent)
                if d < best_dist:
                    best_dist = d
                    best_id = tid
            if best_id is not None:
                assignments.append(best_id)
                used.add(best_id)
                # Exponential Moving Average smoothing
                a = self.node.get_parameter('ema_alpha').value
                self.tracks[best_id]['centroid'] = a * cent + (1 - a) * self.tracks[best_id]['centroid']
                self.tracks[best_id]['hits'] += 1
                self.tracks[best_id]['age'] = 0
                self.tracks[best_id]['points'] = clusters[len(assignments)-1]['points']
                start_pos = self.tracks[best_id].get('start_pos', self.tracks[best_id]['centroid'])
                self.tracks[best_id]['start_pos'] = start_pos
                self.tracks[best_id]['displacement'] = np.linalg.norm(start_pos - self.tracks[best_id]['centroid'])
                self.tracks[best_id]['total_frames'] = self.tracks[best_id].get('total_frames', 0) + 1

            else:
                assignments.append(self.next_id)
                self.tracks[self.next_id] = {
                    'centroid': cent.copy(), 
                    'hits': 1, 
                    'age': 0,
                    'start_pos': cent.copy(),
                    'displacement': 0.0,
                    'total_frames': 1,
                    'points': clusters[len(assignments)-1]['points']
                }
                self.next_id += 1

        # Age management
        for tid in list(self.tracks.keys()):
            if tid not in used:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.node.get_parameter('track_max_age').value:
                    del self.tracks[tid]
                    if tid == self.locked_human_id:
                        self.locked_human_id = None

        self.publish_visualization(assignments, clusters, header)
        self.publish_detected_objects(assignments, clusters, header)
        self.filter_and_publish_human(header)

    def publish_visualization(self, assignments, clusters, header):
        marker_array = MarkerArray()
        colored_points = []
        active_ids = set()

        for cluster, track_id in zip(clusters, assignments):
            track = self.tracks[track_id]
            if track['hits'] < self.node.get_parameter('track_min_hits').value:
                continue

            active_ids.add(track_id)
            centroid = track['centroid']
            r, g, b = self.get_color(track_id)

            # Colored point cloud
            pts = cluster['points'].copy()
            rgb_packed = (int(r*255) << 16) | (int(g*255) << 8) | int(b*255)
            pts_rgb = np.zeros((len(pts), 4))
            pts_rgb[:, :3] = pts
            pts_rgb[:, 3] = rgb_packed
            colored_points.append(pts_rgb)

            # Text marker
            text = Marker()
            text.header = header
            text.ns = "id_text"
            text.id = track_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.MODIFY
            text.pose.position.x = centroid[0]
            text.pose.position.y = centroid[1]
            text.pose.position.z = centroid[2] + 1.0
            text.text = str(track_id)
            text.scale.z = 0.8
            text.color.r, text.color.g, text.color.b, text.color.a = r, g, b, 1.0
            text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()  # short lifetime
            marker_array.markers.append(text)

            # Center sphere
            sphere = Marker()
            sphere.header = header
            sphere.ns = "center"
            sphere.id = track_id + 10000
            sphere.type = Marker.SPHERE
            sphere.action = Marker.MODIFY
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = centroid
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.4
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 1.0
            sphere.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(sphere)

        # Delete old markers
        for tid in self.tracks:
            if tid not in active_ids:
                for ns, base in [("id_text", 0), ("center", 10000)]:
                    del_marker = Marker()
                    del_marker.header = header
                    del_marker.ns = ns
                    del_marker.id = tid + base
                    del_marker.action = Marker.DELETE
                    marker_array.markers.append(del_marker)

        # Publish colored point cloud
        if colored_points:
            all_arr = np.vstack(colored_points)
            dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)]
            structured = np.zeros(all_arr.shape[0], dtype=dtype)
            structured['x'] = all_arr[:,0]
            structured['y'] = all_arr[:,1]
            structured['z'] = all_arr[:,2]
            structured['rgb'] = all_arr[:,3].astype(np.uint32)

            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1),
            ]
            cloud_msg = pc2.create_cloud(header, fields, structured)
            self.node.pub_clustered.publish(cloud_msg)

        self.node.pub_markers.publish(marker_array)

    def publish_detected_objects(self, assignments, clusters, header):
        arr = DetectedObjectArray()
        arr.header = header
        for cluster, tid in zip(clusters, assignments):
            track = self.tracks[tid]
            if track['hits'] < self.node.get_parameter('track_min_hits').value:
                continue
            obj = DetectedObject()
            obj.id = tid
            obj.center = Vector3(x=float(-track['centroid'][0]), # Negate X
                                 y=float(-track['centroid'][1]), # Negate Y
                                 z=float(track['centroid'][2] + 2.0)) # offset height
            obj.point_count = len(cluster['points'])
            dx, dy = track['centroid'][0], track['centroid'][1]
            obj.distance = math.hypot(dx, dy)
            obj.angle_deg = math.degrees(math.atan2(-dx, dy))
            arr.objects.append(obj)
        self.node.pub_objects.publish(arr)

    # Human Detection and Filtering
    def filter_and_publish_human(self, header):
        h_min = self.node.get_parameter('human_height_min').value
        h_max = self.node.get_parameter('human_height_max').value
        w_max = self.node.get_parameter('human_width_max').value
        d_max = self.node.get_parameter('human_depth_max').value
        ratio_min = self.node.get_parameter('human_ratio_min').value
        footprint_max = self.node.get_parameter('human_footprint_max').value
        vol_min = self.node.get_parameter('human_volume_min').value
        vol_max = self.node.get_parameter('human_volume_max').value
        compact_max = self.node.get_parameter('human_compactness_max').value
        flatness_min = self.node.get_parameter('human_xy_flatness_min').value
        
        min_motion = self.node.get_parameter('min_motion_threshold').value
        static_frames = self.node.get_parameter('static_check_frames').value
        
        min_hits = self.node.get_parameter('track_min_hits').value

        candidates = [] 

        for tid, track in self.tracks.items():
            if track['hits'] < min_hits: continue
            if track['total_frames'] > static_frames:
                if track['displacement'] < min_motion:
                    continue # REJECT STATIC

            pts = track['points']
            if len(pts) < 5: continue

            # Geometric Checks
            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            dims = max_pt - min_pt
            dx, dy, dz = dims[0], dims[1], dims[2]
            
            xy_extent = max(dx, dy)
            xy_min = min(dx, dy)
            footprint = dx * dy
            volume = dx * dy * dz
            aspect_ratio = dz / xy_extent if xy_extent > 0 else 0
            
            if not (h_min < dz < h_max): continue
            if (dx > d_max) or (dy > w_max): continue
            if aspect_ratio < ratio_min: continue
            if footprint > footprint_max: continue
            if not (vol_min < volume < vol_max): continue
            if xy_extent > 0 and (xy_min / xy_extent) < flatness_min: continue

            z_bins = np.linspace(min_pt[2], max_pt[2], 4)
            hist, _ = np.histogram(pts[:, 2], bins=z_bins)
            if np.count_nonzero(hist) < 2: continue 

            centroid_xy = track['centroid'][:2]
            dists_to_center = np.linalg.norm(pts[:, :2] - centroid_xy, axis=1)
            if np.mean(dists_to_center) > compact_max: continue

            dist = math.hypot(track['centroid'][0], track['centroid'][1])
            candidates.append({'dist': dist, 'id': tid, 'track': track})

        selected_track = None
        candidates.sort(key=lambda x: x['dist'])

        if not candidates:
            self.locked_human_id = None
        else:
            best_candidate = candidates[0]
            if self.locked_human_id is None:
                selected_track = best_candidate['track']
                self.locked_human_id = best_candidate['id']
            else:
                locked_candidate = next((c for c in candidates if c['id'] == self.locked_human_id), None)
                if locked_candidate:
                    if best_candidate['dist'] < (locked_candidate['dist'] - 1.5):
                        selected_track = best_candidate['track']
                        self.locked_human_id = best_candidate['id']
                    else:
                        selected_track = locked_candidate['track']
                else:
                    selected_track = best_candidate['track']
                    self.locked_human_id = best_candidate['id']

        marker_array = MarkerArray()
        
        if selected_track:
            cx, cy = selected_track['centroid'][0], selected_track['centroid'][1]
            cz = selected_track['centroid'][2]
            dist_val = int(math.hypot(cx, cy) + 0.5)
            
            angle_rad = math.atan2(-cx, cy)
            angle_deg = int(math.degrees(angle_rad) + 0.5)
            if angle_deg < 0: angle_deg += 360

            pos_msg = Int32MultiArray()
            pos_msg.data = [dist_val, angle_deg]
            self.node.pub_pedestrian_pos.publish(pos_msg)

            info_marker = Marker()
            info_marker.header = header
            info_marker.ns = "human_info"
            info_marker.id = 999
            info_marker.type = Marker.TEXT_VIEW_FACING
            info_marker.action = Marker.ADD
            info_marker.pose.position.x = cx
            info_marker.pose.position.y = cy
            info_marker.pose.position.z = cz + 2.0 
            info_marker.text = f"HUMAN ID:{self.locked_human_id}\n{dist_val}m | {angle_deg}deg"
            info_marker.scale.z = 0.5
            info_marker.color.r, info_marker.color.g, info_marker.color.b, info_marker.color.a = 0.0, 1.0, 0.0, 1.0
            info_marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(info_marker)
            
            cyl = Marker()
            cyl.header = header
            cyl.ns = "human_highlight"
            cyl.id = 1000
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = cx
            cyl.pose.position.y = cy
            cyl.pose.position.z = cz
            cyl.scale.x = 0.8
            cyl.scale.y = 0.8
            cyl.scale.z = 1.8
            cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 0.0, 1.0, 0.0, 0.4
            cyl.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(cyl)
        # else:
        #     empty_msg = Int32MultiArray()
        #     empty_msg.data = [0, 0]
        #     self.node.pub_pedestrian_pos.publish(empty_msg)

        self.node.pub_human_debug.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()