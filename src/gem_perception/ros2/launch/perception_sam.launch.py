import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("gem_perception_ros2")
    cfg = [pkg, "/config/perception.yaml"]
    rviz = [pkg, "/rviz/perception.rviz"]

    return LaunchDescription([
        DeclareLaunchArgument("default_prompt", default_value=""),
        Node(
            package="gem_perception_ros2",
            executable="sam_perception_node",
            name="gem_perception_sam",
            output="screen",
            parameters=[cfg, {"default_prompt": LaunchConfiguration("default_prompt")}],
        ),
        Node(
            package="gem_perception_ros2",
            executable="map_tf_broadcaster",
            name="map_tf_broadcaster",
            output="screen",
            parameters=[cfg],
        ),
        Node(
            package="gem_perception_ros2",
            executable="bev_overlay_node",
            name="bev_overlay_node",
            output="screen",
            parameters=[cfg],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_perception",
            output="screen",
            arguments=["-d", rviz],
        ),
    ])
