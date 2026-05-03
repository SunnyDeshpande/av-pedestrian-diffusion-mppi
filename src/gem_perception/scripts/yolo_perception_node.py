#!/usr/bin/env python3
"""ROS1 (noetic) wrapper: YOLO-World detection → LiDAR fusion → goal pose."""
import os
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Bool, Header, String
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

from gem_perception.geometry import (
    project_to_image,
    transform_points,
)
from gem_perception.pipeline import PipelineParams, run_pipeline
from gem_perception.ros_common import (
    GoalHold,
    K_from_camera_info,
    draw_detection_overlay,
    draw_lidar_projection,
    transform_to_matrix,
)
from gem_perception.yolo_detector import YoloWorldDetector


class YoloPerceptionNode:
    def __init__(self):
        rospy.init_node("gem_perception_yolo")

        self.image_topic = rospy.get_param("~image_topic", "/oak/rgb/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/oak/rgb/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.prompt_topic = rospy.get_param("~prompt_topic", "/perception/prompt")

        self.camera_frame = rospy.get_param("~camera_frame", "front_single_camera_optical_link")
        self.lidar_frame = rospy.get_param("~lidar_frame", "ouster")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")

        weight = rospy.get_param(
            "~yolo_weight",
            os.path.expanduser("~/host/gem_perception_models/yolov8s-worldv2.pt"),
        )
        device = rospy.get_param("~device", "cuda")
        conf = rospy.get_param("~conf", 0.05)
        default_prompt = rospy.get_param("~default_prompt", "")

        self.params = PipelineParams(
            z_min_base=rospy.get_param("~z_min_base", 0.15),
            z_max_base=rospy.get_param("~z_max_base", 5.0),
            dbscan_eps=rospy.get_param("~dbscan_eps", 0.4),
            dbscan_min_samples=rospy.get_param("~dbscan_min_samples", 3),
            min_cluster_points=rospy.get_param("~min_cluster_points", 3),
            estimated_goal_distance=rospy.get_param("~estimated_goal_distance", 15.0),
        )

        self.detector = YoloWorldDetector(weight, device=device, conf=conf)
        if default_prompt:
            self.detector.set_prompt(default_prompt)

        self.hold = GoalHold(hold_seconds=rospy.get_param("~goal_hold_seconds", 2.0))
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_ann = rospy.Publisher("/perception/image_annotated", Image, queue_size=1)
        self.pub_proj = rospy.Publisher("/perception/lidar_projected_image", Image, queue_size=1)
        self.pub_cluster = rospy.Publisher("/perception/object_cluster", PointCloud2, queue_size=1)
        self.pub_markers = rospy.Publisher("/perception/object_bbox_3d", MarkerArray, queue_size=1)
        self.pub_goal_map = rospy.Publisher("/perception/goal_pose", PoseStamped, queue_size=1)
        self.pub_goal_base = rospy.Publisher("/perception/goal_pose_base_link", PoseStamped, queue_size=1)
        self.pub_goal_est_flag = rospy.Publisher("/perception/goal_is_estimated", Bool, queue_size=1)

        rospy.Subscriber(self.prompt_topic, String, self._on_prompt, queue_size=1)
        self.sub_img = Subscriber(self.image_topic, Image)
        self.sub_info = Subscriber(self.camera_info_topic, CameraInfo)
        self.sub_pc = Subscriber(self.lidar_topic, PointCloud2)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_info, self.sub_pc], queue_size=5, slop=0.2
        )
        self.sync.registerCallback(self._on_frame)

        rospy.loginfo("gem_perception (YOLO-World) ready")

    def _on_prompt(self, msg: String):
        with self.lock:
            self.detector.set_prompt(msg.data)
        rospy.loginfo("Prompt updated: %s", msg.data)

    def _lookup_matrix(self, target_frame: str, source_frame: str, stamp) -> np.ndarray:
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.5))
        t = tf.transform.translation
        q = tf.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def _on_frame(self, img_msg: Image, info_msg: CameraInfo, pc_msg: PointCloud2):
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"cv_bridge: {e}")
            return

        with self.lock:
            det = self.detector.infer(image_bgr)

        K = K_from_camera_info(info_msg.K)

        try:
            T_cam_lidar = self._lookup_matrix(self.camera_frame, self.lidar_frame, img_msg.header.stamp)
            T_base_lidar = self._lookup_matrix(self.base_frame, self.lidar_frame, img_msg.header.stamp)
            T_base_cam = self._lookup_matrix(self.base_frame, self.camera_frame, img_msg.header.stamp)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
            return

        # LiDAR → numpy
        pc_list = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        points_lidar = np.asarray(pc_list, dtype=np.float64) if pc_list else np.empty((0, 3), dtype=np.float64)

        # LiDAR projection overlay (always published for calibration check)
        pts_cam_all = transform_points(points_lidar, T_cam_lidar)
        uv, idx = project_to_image(pts_cam_all, K)
        depths = pts_cam_all[idx, 2] if idx.size else np.empty((0,))
        proj_img = draw_lidar_projection(image_bgr, uv, depths)
        self.pub_proj.publish(self.bridge.cv2_to_imgmsg(proj_img, "bgr8"))

        now = rospy.Time.now().to_sec()
        goal_base = None
        is_estimated = False
        result = None

        if det is not None:
            result = run_pipeline(
                det, points_lidar, K, T_cam_lidar, T_base_lidar, T_base_cam, self.params
            )
            goal_base = result.goal_base
            is_estimated = result.is_estimated

            ann = draw_detection_overlay(
                image_bgr, det.bbox_xyxy, det.mask, result.pixel_centroid,
                is_estimated, det.prompt, det.score,
            )
            self.pub_ann.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))

            if result.cluster_cloud_base is not None:
                self._publish_cluster_cloud(result.cluster_cloud_base, img_msg.header.stamp)
            self._publish_markers(result, img_msg.header.stamp)
        else:
            # publish raw image through the annotated topic so rviz stays alive
            self.pub_ann.publish(self.bridge.cv2_to_imgmsg(image_bgr, "bgr8"))

        held = self.hold.update(now, goal_base, is_estimated)
        if held is None:
            self.pub_goal_est_flag.publish(Bool(data=False))
            return
        g, is_est = held
        self._publish_goal(g, is_est, img_msg.header.stamp)

    def _publish_cluster_cloud(self, pts_base: np.ndarray, stamp):
        header = Header(frame_id=self.base_frame, stamp=stamp)
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        cloud = pc2.create_cloud(header, fields, pts_base.astype(np.float32).tolist())
        self.pub_cluster.publish(cloud)

    def _publish_markers(self, result, stamp):
        markers = MarkerArray()
        if result.cluster is not None:
            c = result.cluster
            mn, mx = c.bbox_min_base, c.bbox_max_base
            mk = Marker()
            mk.header.frame_id = self.base_frame
            mk.header.stamp = stamp
            mk.ns = "object_bbox_3d"
            mk.id = 0
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x = float((mn[0] + mx[0]) / 2)
            mk.pose.position.y = float((mn[1] + mx[1]) / 2)
            mk.pose.position.z = float((mn[2] + mx[2]) / 2)
            mk.pose.orientation.w = 1.0
            mk.scale.x = max(float(mx[0] - mn[0]), 0.1)
            mk.scale.y = max(float(mx[1] - mn[1]), 0.1)
            mk.scale.z = max(float(mx[2] - mn[2]), 0.1)
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.0, 1.0, 0.0, 0.4
            markers.markers.append(mk)

        mk = Marker()
        mk.header.frame_id = self.base_frame
        mk.header.stamp = stamp
        mk.ns = "goal"
        mk.id = 1
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose.position.x = float(result.goal_base[0])
        mk.pose.position.y = float(result.goal_base[1])
        mk.pose.position.z = float(result.goal_base[2])
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.6
        if result.is_estimated:
            mk.color.r, mk.color.g, mk.color.b = 1.0, 1.0, 0.0
        else:
            mk.color.r, mk.color.g, mk.color.b = 0.0, 1.0, 0.0
        mk.color.a = 0.8
        markers.markers.append(mk)
        self.pub_markers.publish(markers)

    def _publish_goal(self, goal_base: np.ndarray, is_estimated: bool, stamp):
        # base_link frame
        pb = PoseStamped()
        pb.header.frame_id = self.base_frame
        pb.header.stamp = stamp
        pb.pose.position.x = float(goal_base[0])
        pb.pose.position.y = float(goal_base[1])
        pb.pose.position.z = float(goal_base[2])
        pb.pose.orientation.w = 1.0
        self.pub_goal_base.publish(pb)

        # Transform to map
        try:
            T_map_base = self._lookup_matrix(self.map_frame, self.base_frame, stamp)
            pt_map = T_map_base @ np.array([goal_base[0], goal_base[1], goal_base[2], 1.0])
            pm = PoseStamped()
            pm.header.frame_id = self.map_frame
            pm.header.stamp = stamp
            pm.pose.position.x = float(pt_map[0])
            pm.pose.position.y = float(pt_map[1])
            pm.pose.position.z = float(pt_map[2])
            pm.pose.orientation.w = 1.0
            self.pub_goal_map.publish(pm)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"map TF unavailable, skipping map-frame goal: {e}")

        self.pub_goal_est_flag.publish(Bool(data=bool(is_estimated)))


def main():
    YoloPerceptionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
