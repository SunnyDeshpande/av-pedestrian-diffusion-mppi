#!/usr/bin/env python3
"""Draw perception goal onto /motion_image (BEV) and republish.

Uses the same pixel math as gem_gnss_image.py to convert lat/lon of the goal
back into the BEV image. The goal's map-frame XY is converted to lat/lon using
the ref anchor (same as map_tf_broadcaster).
"""
import math
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


EARTH_R = 6378137.0


def enu_to_latlon(e, n, lat0_deg, lon0_deg):
    lat0 = math.radians(lat0_deg)
    lon = lon0_deg + math.degrees(e / (EARTH_R * math.cos(lat0)))
    lat = lat0_deg + math.degrees(n / EARTH_R)
    return lat, lon


class BevOverlayNode:
    def __init__(self):
        rospy.init_node("bev_overlay_node")
        self.bridge = CvBridge()

        self.ref_lat = rospy.get_param("~ref_lat", 40.092722)
        self.ref_lon = rospy.get_param("~ref_lon", -88.236365)
        self.lat_start_bt = rospy.get_param("~lat_start_bt", 40.092722)
        self.lon_start_l = rospy.get_param("~lon_start_l", -88.236365)
        self.lat_scale = rospy.get_param("~lat_scale", 0.00062)
        self.lon_scale = rospy.get_param("~lon_scale", 0.00136)
        self.img_width = rospy.get_param("~img_width", 2107)
        self.img_height = rospy.get_param("~img_height", 1313)

        self.lock = threading.Lock()
        self.latest_goal_map = None
        self.is_estimated = False

        rospy.Subscriber("/perception/goal_pose", PoseStamped, self._on_goal, queue_size=1)
        rospy.Subscriber("/perception/goal_is_estimated", Bool, self._on_flag, queue_size=1)

        self.pub = rospy.Publisher("/motion_image_with_goal", Image, queue_size=1)
        rospy.Subscriber("/motion_image", Image, self._on_image, queue_size=1)

    def _on_goal(self, msg):
        with self.lock:
            self.latest_goal_map = (msg.pose.position.x, msg.pose.position.y,
                                    rospy.Time.now().to_sec())

    def _on_flag(self, msg):
        with self.lock:
            self.is_estimated = bool(msg.data)

    def _on_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        with self.lock:
            g = self.latest_goal_map
            is_est = self.is_estimated
        if g is not None:
            gx, gy, t = g
            if rospy.Time.now().to_sec() - t < 3.0:
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
            rospy.logwarn_throttle(5.0, f"bev overlay publish: {e}")


def main():
    BevOverlayNode()
    rospy.spin()


if __name__ == "__main__":
    main()
