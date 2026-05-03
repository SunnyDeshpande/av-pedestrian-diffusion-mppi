#!/usr/bin/env python3
"""Broadcast map → base_link TF from /septentrio_gnss/insnavgeod.

Uses a fixed GPS anchor (lat/lon) as the map origin; converts each incoming
INSNavGeod to a local ENU offset and publishes it. For the real car, replace
this with your localization stack — the downstream perception node only needs
map→base_link to exist.
"""
import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

try:
    from septentrio_gnss_driver.msg import INSNavGeod
except ImportError:
    INSNavGeod = None


EARTH_R = 6378137.0


def latlon_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    dlat = lat - lat0
    dlon = lon - lon0
    e = EARTH_R * math.cos(lat0) * dlon
    n = EARTH_R * dlat
    return e, n


class MapTfBroadcaster:
    def __init__(self):
        rospy.init_node("map_tf_broadcaster")
        self.lat0 = rospy.get_param("~ref_lat", 40.092722)
        self.lon0 = rospy.get_param("~ref_lon", -88.236365)
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.br = tf2_ros.TransformBroadcaster()
        if INSNavGeod is None:
            rospy.logerr("septentrio_gnss_driver not available. Cannot broadcast map TF.")
            return
        rospy.Subscriber("/septentrio_gnss/insnavgeod", INSNavGeod, self._cb, queue_size=10)
        rospy.loginfo("map_tf_broadcaster: origin lat=%.6f lon=%.6f", self.lat0, self.lon0)

    def _cb(self, msg):
        e, n = latlon_to_enu(msg.latitude, msg.longitude, self.lat0, self.lon0)
        # heading in INSNavGeod is typically deg, measured clockwise from north; convert to ENU yaw (CCW from east)
        heading_deg = float(msg.heading)
        yaw_enu = math.radians(90.0 - heading_deg)
        half_yaw = yaw_enu / 2.0
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = e
        t.transform.translation.y = n
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.br.sendTransform(t)


def main():
    MapTfBroadcaster()
    rospy.spin()


if __name__ == "__main__":
    main()
