#!/usr/bin/env python3
# ------- Imports -------
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Int32MultiArray, Bool

from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

import numpy as np
import cv2
import time

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# COCO class ID for 'person'
COCO_PERSON_CLASS_ID = 0

class RgbdPedestrianDetector(Node):
    def __init__(self):
        super().__init__('rgbd_pedestrian_detector')

        # ------- Parameters -------
        self.declare_parameteCOCO_PERSON_CLASS_IDr('publish_debug_image', True)
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('half', False)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('depth_topic', '/oak/stereo/image_raw')

        self.publish_debug = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.publish_debug = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.half = bool(self.get_parameter('half').value)
        self.max_det = int(self.get_parameter('max_detections').value)

        # ------- YOLO model -------
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`")

        self.get_logger().info(f"Loading YOLOv11 model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.overrides['conf'] = self.conf
        self.model.overrides['iou'] = self.iou
        self.model.overrides['device'] = self.device
        self.model.overrides['imgsz'] = self.imgsz
        self.model.overrides['half'] = self.half
        self.model.overrides['max_det'] = self.max_det
        self.model.overrides['classes'] = [COCO_PERSON_CLASS_ID]

        # ------- ROS I/O -------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.bridge = CvBridge()

        # ------- Inputs -------
        self.sub_rgb = self.create_subscription(Image, self.image_topic, self.image_cb, qos)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_cb, qos)

        # ------- Outputs -------
        self.pub_dets = self.create_publisher(Detection2DArray, 'detections', 10)
        self.pub_debug = self.create_publisher(Image, 'detections/image', 10) if self.publish_debug else None

        self.pub_rgbd_position = self.create_publisher(
            Int32MultiArray, 'rgbd_pedestrian_position', 10
        )
        self.pub_ped_sign_present = self.create_publisher(
            Bool, 'pedestrian_sign_present', 10
        )
        
        self.pub_person_marker = self.create_publisher(Marker, 'person_3d_marker', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.publish_camera_transform()

        # ---------------- State ----------------
        self.latest_depth = None
        self.last_fps_t = time.time()
        self.frame_count = 0

        self.get_logger().info("RGB-D Person 3D Extractor ready (distance+direction+presence).")

    # ------- TF -------
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

    # ------- Depth image callback -------
    def depth_cb(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = depth_img.astype(np.float32) / 1000.0  # mm -> m
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    # ------- RGB image callback -------
    def image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge RGB failed: {e}")
            return

        self.publish_camera_transform()

        h, w = cv_image.shape[:2]
        debug_img = cv_image.copy() if self.publish_debug else None

        # crude intrinsics from image size (replace with real K if you have)
        fx = float(w)
        fy = float(h)
        cx0 = w / 2.0
        cy0 = h / 2.0

        # YOLO inference
        results = self.model.predict(cv_image, verbose=False, stream=False)
        det_msg = Detection2DArray()
        det_msg.header = msg.header

        # Tracking closest pedestrian
        best_dist = float('inf')
        best_angle_deg = 0.0
        found_valid_ped = False

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)

            for i in range(xyxy.shape[0]):
                if clss[i] != COCO_PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                score = float(confs[i])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1)
                bh = (y2 - y1)

                # 2D detection message for visualization
                det = Detection2D()
                det.header = msg.header
                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = float(cx)
                det.bbox.center.position.y = float(cy)
                det.bbox.size_x = float(bw)
                det.bbox.size_y = float(bh)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = "person"
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                det_msg.detections.append(det)

                # Depth projection to 3D
                z = 0.0
                if self.latest_depth is not None:
                    xi = int(np.clip(cx, 0, self.latest_depth.shape[1] - 1))
                    yi = int(np.clip(cy, 0, self.latest_depth.shape[0] - 1))
                    z = float(self.latest_depth[yi, xi])
                if z <= 0.0:
                    # skip if no depth
                    continue

                # Camera optical frame (x right, y down, z forward)
                X_optical = (cx - cx0) * z / fx
                Y_optical = (cy - cy0) * z / fy
                Z_optical = z

                # Approximate transform to base_link:
                X_base = Y_optical 
                Y_base = -X_optical
                Z_base = Z_optical

                # Distance in ground plane (Y,Z)
                dist = float(np.sqrt(Y_base**2 + Z_base**2))

                # Direction:
                # 0° axis = right side of vehicle (-ve Y axis of ego frame),
                # CCW positive: 0° right, 90° front, 180° left, 270° back.
                angle_rad = np.arctan2(Z_base, -Y_base)
                angle_deg = float(np.degrees(angle_rad))
                if angle_deg < 0.0:
                    angle_deg += 360.0

                # Update closest pedestrian
                if dist < best_dist:
                    best_dist = dist
                    best_angle_deg = angle_deg
                    found_valid_ped = True

                # Debug info on image
                if debug_img is not None:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(debug_img, p1, p2, (0, 255, 0), 2)
                    label = f"person {score:.2f}"
                    cv2.putText(
                        debug_img,
                        label,
                        (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    info_str = f"{dist:.1f} m, {angle_deg:.1f} deg"
                    cv2.putText(
                        debug_img,
                        info_str,
                        (p1[0], min(h - 5, p2[1] + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # Optional 3D marker
                marker = Marker()
                marker.header.frame_id = 'base_link'
                marker.header.stamp = msg.header.stamp
                marker.ns = "person_3d"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = X_base
                marker.pose.position.y = Y_base
                marker.pose.position.z = Z_base
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                self.pub_person_marker.publish(marker)

        # Publish detections / debug images
        self.pub_dets.publish(det_msg)
        if self.pub_debug is not None and debug_img is not None:
            out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_debug.publish(out_msg)

        # Publish presence + (distance, direction)
        sign_msg = Bool()
        if found_valid_ped:
            sign_msg.data = True
            pos_msg = Int32MultiArray()
            pos_msg.data = [
                int(round(best_dist)),       # distance in meters (int)
                int(round(best_angle_deg))   # direction in degrees CCW from right
            ]
            self.pub_rgbd_position.publish(pos_msg)
        else:
            sign_msg.data = False

        self.pub_ped_sign_present.publish(sign_msg)

        # FPS logging
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now_t = time.time()
            fps = 30.0 / (now_t - self.last_fps_t + 1e-9)
            self.last_fps_t = now_t
            self.get_logger().info(f"~{fps:.1f} FPS")

# ------- Main -------
def main(args=None):
    rclpy.init(args=args)
    node = RgbdPedestrianDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RGB-D Person 3D Extractor")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
