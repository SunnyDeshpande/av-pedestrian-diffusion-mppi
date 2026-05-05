"""ROS2 BEV overlay (/motion_image + perception goal → /motion_image_with_goal)."""
import math
import threading

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


EARTH_R = 6378137.0


def enu_to_latlon(e, n, lat0, lon0):
    lat0r = math.radians(lat0)
    lon = lon0 + math.degrees(e / (EARTH_R * math.cos(lat0r)))
    lat = lat0 + math.degrees(n / EARTH_R)
    return lat, lon


class BevOverlayNode(Node):
    def __init__(self):
        super().__init__("bev_overlay_node")
        self.bridge = CvBridge()
        self.declare_parameter("ref_lat", 40.092722)
        self.declare_parameter("ref_lon", -88.236365)
        self.declare_parameter("lat_start_bt", 40.092722)
        self.declare_parameter("lon_start_l", -88.236365)
        self.declare_parameter("lat_scale", 0.00062)
        self.declare_parameter("lon_scale", 0.00136)
        self.declare_parameter("img_width", 2107)
        self.declare_parameter("img_height", 1313)
        p = self.get_parameter
        self.ref_lat = p("ref_lat").value; self.ref_lon = p("ref_lon").value
        self.lat_start_bt = p("lat_start_bt").value; self.lon_start_l = p("lon_start_l").value
        self.lat_scale = p("lat_scale").value; self.lon_scale = p("lon_scale").value
        self.img_width = p("img_width").value; self.img_height = p("img_height").value

        self.lock = threading.Lock()
        self.latest_goal = None
        self.is_estimated = False

        self.create_subscription(PoseStamped, "/perception/goal_pose", self._on_goal, 1)
        self.create_subscription(Bool, "/perception/goal_is_estimated", self._on_flag, 1)
        self.pub = self.create_publisher(Image, "/motion_image_with_goal", 1)
        self.create_subscription(Image, "/motion_image", self._on_image, 1)

    def _on_goal(self, msg):
        with self.lock:
            self.latest_goal = (msg.pose.position.x, msg.pose.position.y,
                                self.get_clock().now().nanoseconds * 1e-9)

    def _on_flag(self, msg):
        with self.lock:
            self.is_estimated = bool(msg.data)

    def _on_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        with self.lock:
            g = self.latest_goal
            is_est = self.is_estimated
        if g is not None:
            gx, gy, t = g
            if (self.get_clock().now().nanoseconds * 1e-9) - t < 3.0:
                lat, lon = enu_to_latlon(gx, gy, self.ref_lat, self.ref_lon)
                px = int(self.img_width * (lon - self.lon_start_l) / self.lon_scale)
                py = int(self.img_height - self.img_height * (lat - self.lat_start_bt) / self.lat_scale)
                if 0 <= px < self.img_width and 0 <= py < self.img_height:
                    color = (0, 255, 255) if is_est else (0, 255, 0)
                    cv2.circle(img, (px, py), 18, color, 3)
                    cv2.drawMarker(img, (px, py), color, cv2.MARKER_CROSS, 30, 2)
                    label = "GOAL [est]" if is_est else "GOAL"
                    cv2.putText(img, label, (px + 22, py + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"bev overlay publish: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BevOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
