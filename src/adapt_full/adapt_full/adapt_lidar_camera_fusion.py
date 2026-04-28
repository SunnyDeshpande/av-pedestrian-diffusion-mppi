#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import message_filters
import numpy as np
import math


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
