"""ROS2 (jazzy) wrapper: YOLO-World detection → LiDAR fusion → goal pose."""
import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Bool, Header, String
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration

from .geometry import project_to_image, transform_points
from .pipeline import PipelineParams, run_pipeline
from .ros_common import (
    D_from_camera_info,
    GoalHold,
    K_from_camera_info,
    draw_detection_overlay,
    draw_lidar_projection,
    transform_to_matrix,
)
from .yolo_detector import YoloWorldDetector


class YoloPerceptionNode(Node):
    def __init__(self):
        super().__init__("gem_perception_yolo")

        self.declare_parameter("image_topic", "/oak/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")
        self.declare_parameter("lidar_topic", "/ouster/points")
        self.declare_parameter("prompt_topic", "/perception/prompt")
        self.declare_parameter("camera_frame", "front_single_camera_optical_link")
        self.declare_parameter("lidar_frame", "ouster")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        # Default to "auto" which expands to <models_root>/yolov8s-worldv2.pt; the
        # detector picks models_root from env GEM_PERCEPTION_MODELS or platform defaults.
        self.declare_parameter("yolo_weight", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("conf", 0.05)
        self.declare_parameter("default_prompt", "")
        self.declare_parameter("z_min_base", 0.15)
        self.declare_parameter("z_max_base", 5.0)
        self.declare_parameter("dbscan_eps", 0.4)
        self.declare_parameter("dbscan_min_samples", 3)
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("estimated_goal_distance", 15.0)
        self.declare_parameter("goal_hold_seconds", 2.0)

        p = self.get_parameter
        self.image_topic = p("image_topic").value
        self.camera_info_topic = p("camera_info_topic").value
        self.lidar_topic = p("lidar_topic").value
        self.prompt_topic = p("prompt_topic").value
        self.camera_frame = p("camera_frame").value
        self.lidar_frame = p("lidar_frame").value
        self.base_frame = p("base_frame").value
        self.map_frame = p("map_frame").value

        self.params = PipelineParams(
            z_min_base=p("z_min_base").value,
            z_max_base=p("z_max_base").value,
            dbscan_eps=p("dbscan_eps").value,
            dbscan_min_samples=p("dbscan_min_samples").value,
            min_cluster_points=p("min_cluster_points").value,
            estimated_goal_distance=p("estimated_goal_distance").value,
        )

        from .sam_detector import _default_models_root
        weight = p("yolo_weight").value
        if not weight:
            weight = os.path.join(_default_models_root(), "yolov8s-worldv2.pt")
        self.detector = YoloWorldDetector(weight,
                                          device=p("device").value,
                                          conf=p("conf").value)
        dp = p("default_prompt").value
        if dp:
            self.detector.set_prompt(dp)

        self.hold = GoalHold(hold_seconds=p("goal_hold_seconds").value)
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_ann = self.create_publisher(Image, "/perception/image_annotated", 1)
        self.pub_proj = self.create_publisher(Image, "/perception/lidar_projected_image", 1)
        self.pub_cluster = self.create_publisher(PointCloud2, "/perception/object_cluster", 1)
        self.pub_markers = self.create_publisher(MarkerArray, "/perception/object_bbox_3d", 1)
        self.pub_goal_map = self.create_publisher(PoseStamped, "/perception/goal_pose", 1)
        self.pub_goal_base = self.create_publisher(PoseStamped, "/perception/goal_pose_base_link", 1)
        self.pub_goal_est_flag = self.create_publisher(Bool, "/perception/goal_is_estimated", 1)

        self.create_subscription(String, self.prompt_topic, self._on_prompt, 1)
        self.sub_img = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=qos)
        self.sub_info = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos)
        self.sub_pc = message_filters.Subscriber(self, PointCloud2, self.lidar_topic, qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_info, self.sub_pc], queue_size=5, slop=0.2
        )
        self.sync.registerCallback(self._on_frame)

        self.get_logger().info("gem_perception (YOLO-World) ready")

    def _on_prompt(self, msg: String):
        with self.lock:
            self.detector.set_prompt(msg.data)
        self.get_logger().info(f"Prompt updated: {msg.data}")

    def _lookup_matrix(self, target_frame, source_frame, stamp):
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp,
                                                 timeout=Duration(seconds=0.2))
        except Exception:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame,
                                                 rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.5))
        t = tf.transform.translation
        q = tf.transform.rotation
        return transform_to_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def _on_frame(self, img_msg, info_msg, pc_msg):
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        with self.lock:
            det = self.detector.infer(image_bgr)

        K = K_from_camera_info(info_msg.k)
        D = D_from_camera_info(info_msg.d)

        try:
            T_cam_lidar = self._lookup_matrix(self.camera_frame, self.lidar_frame, img_msg.header.stamp)
            T_base_lidar = self._lookup_matrix(self.base_frame, self.lidar_frame, img_msg.header.stamp)
            T_base_cam = self._lookup_matrix(self.base_frame, self.camera_frame, img_msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f"TF lookup: {e}")
            return

        pc_list = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        if pc_list:
            points_lidar = np.asarray([[p[0], p[1], p[2]] for p in pc_list], dtype=np.float64)
        else:
            points_lidar = np.empty((0, 3), dtype=np.float64)

        pts_cam_all = transform_points(points_lidar, T_cam_lidar)
        uv, idx = project_to_image(pts_cam_all, K, D)
        depths = pts_cam_all[idx, 2] if idx.size else np.empty((0,))
        self.pub_proj.publish(self.bridge.cv2_to_imgmsg(
            draw_lidar_projection(image_bgr, uv, depths), "bgr8"))

        now = self.get_clock().now().nanoseconds * 1e-9
        goal_base = None
        is_estimated = False
        result = None

        if det is not None:
            result = run_pipeline(det, points_lidar, K,
                                  T_cam_lidar, T_base_lidar, T_base_cam,
                                  self.params, D=D)
            goal_base = result.goal_base
            is_estimated = result.is_estimated
            ann = draw_detection_overlay(image_bgr, det.bbox_xyxy, det.mask, result.pixel_centroid,
                                         is_estimated, det.prompt, det.score)
            self.pub_ann.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
            if result.cluster_cloud_base is not None:
                self._publish_cluster_cloud(result.cluster_cloud_base, img_msg.header.stamp)
            self._publish_markers(result, img_msg.header.stamp)
        else:
            self.pub_ann.publish(self.bridge.cv2_to_imgmsg(image_bgr, "bgr8"))

        held = self.hold.update(now, goal_base, is_estimated)
        if held is None:
            self.pub_goal_est_flag.publish(Bool(data=False))
            return
        g, is_est = held
        self._publish_goal(g, is_est, img_msg.header.stamp)

    def _publish_cluster_cloud(self, pts_base, stamp):
        header = Header(frame_id=self.base_frame, stamp=stamp)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self.pub_cluster.publish(pc2.create_cloud(header, fields, pts_base.astype(np.float32).tolist()))

    def _publish_markers(self, result, stamp):
        ma = MarkerArray()
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
            ma.markers.append(mk)
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
        ma.markers.append(mk)
        self.pub_markers.publish(ma)

    def _publish_goal(self, goal_base, is_estimated, stamp):
        pb = PoseStamped()
        pb.header.frame_id = self.base_frame
        pb.header.stamp = stamp
        pb.pose.position.x = float(goal_base[0])
        pb.pose.position.y = float(goal_base[1])
        pb.pose.position.z = float(goal_base[2])
        pb.pose.orientation.w = 1.0
        self.pub_goal_base.publish(pb)
        try:
            T_map_base = self._lookup_matrix(self.map_frame, self.base_frame, stamp)
            pt = T_map_base @ np.array([goal_base[0], goal_base[1], goal_base[2], 1.0])
            pm = PoseStamped()
            pm.header.frame_id = self.map_frame
            pm.header.stamp = stamp
            pm.pose.position.x = float(pt[0])
            pm.pose.position.y = float(pt[1])
            pm.pose.position.z = float(pt[2])
            pm.pose.orientation.w = 1.0
            self.pub_goal_map.publish(pm)
        except Exception as e:
            self.get_logger().warn(f"map TF unavailable: {e}")
        self.pub_goal_est_flag.publish(Bool(data=bool(is_estimated)))


def main(args=None):
    rclpy.init(args=args)
    node = YoloPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
