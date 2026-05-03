"""ROS2 map→base_link broadcaster from a GPS topic (sensor_msgs/NavSatFix).

On the real car you would typically replace this with your EKF/localization
output; the perception node only needs map→base_link TF to produce goals in
the map frame.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import NavSatFix, Imu
from tf2_ros import TransformBroadcaster


EARTH_R = 6378137.0


def latlon_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    e = EARTH_R * math.cos(lat0) * (lon - lon0)
    n = EARTH_R * (lat - lat0)
    return e, n


class MapTfBroadcaster(Node):
    def __init__(self):
        super().__init__("map_tf_broadcaster")
        self.declare_parameter("ref_lat", 40.092722)
        self.declare_parameter("ref_lon", -88.236365)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("gps_topic", "/gps/fix")
        self.declare_parameter("imu_topic", "/imu")
        self.lat0 = self.get_parameter("ref_lat").value
        self.lon0 = self.get_parameter("ref_lon").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.br = TransformBroadcaster(self)
        self._last_yaw = 0.0

        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._imu_cb, 10)
        self.create_subscription(NavSatFix, self.get_parameter("gps_topic").value, self._gps_cb, 10)
        self.get_logger().info(f"map_tf_broadcaster: origin {self.lat0:.6f},{self.lon0:.6f}")

    def _imu_cb(self, msg):
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._last_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _gps_cb(self, msg):
        e, n = latlon_to_enu(msg.latitude, msg.longitude, self.lat0, self.lon0)
        half = self._last_yaw / 2.0
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = e
        t.transform.translation.y = n
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(half)
        t.transform.rotation.w = math.cos(half)
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MapTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
