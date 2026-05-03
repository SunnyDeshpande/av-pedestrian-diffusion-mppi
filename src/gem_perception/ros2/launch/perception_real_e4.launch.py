"""Launch perception on the real GEM e4 (ROS2 humble).

Differences from `perception_yolo.launch.py`:
  - Loads `config/perception_real_e4.yaml` instead of the default sim config.
  - Optionally publishes a static TF from `front_single_camera_link`
    (or `oak-d-base-frame`) to `<camera>_optical_link` if the URDF on the
    real car doesn't include the optical-frame child. Toggle with the
    launch arg `publish_optical_tf:=true`.

Usage:
  # YOLO-World, RViz on, optical-TF off (URDF has it already):
  ros2 launch gem_perception_ros2 perception_real_e4.launch.py default_prompt:="red sign"

  # If the URDF lacks the optical frame:
  ros2 launch gem_perception_ros2 perception_real_e4.launch.py \\
    publish_optical_tf:=true \\
    parent_camera_frame:=front_single_camera_link \\
    optical_frame_name:=front_single_camera_optical_link
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("gem_perception_ros2")
    cfg = [pkg, "/config/perception_real_e4.yaml"]
    rviz = [pkg, "/rviz/perception.rviz"]
    detector = LaunchConfiguration("detector")
    is_yolo = PythonExpression(["'", detector, "' == 'yolo'"])
    is_sam = PythonExpression(["'", detector, "' == 'sam'"])

    return LaunchDescription([
        DeclareLaunchArgument("default_prompt", default_value=""),
        DeclareLaunchArgument("run_rviz", default_value="true"),
        DeclareLaunchArgument("detector", default_value="yolo",
                              description="yolo | sam"),
        DeclareLaunchArgument("publish_optical_tf", default_value="false",
                              description="Publish a body→optical static TF"),
        DeclareLaunchArgument("parent_camera_frame",
                              default_value="front_single_camera_link"),
        DeclareLaunchArgument("optical_frame_name",
                              default_value="front_single_camera_optical_link"),

        # Body→optical static TF (-π/2, 0, -π/2) — only if URDF doesn't have it.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="cam_optical_static_tf",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "-1.5707963", "--pitch", "0", "--yaw", "-1.5707963",
                "--frame-id", LaunchConfiguration("parent_camera_frame"),
                "--child-frame-id", LaunchConfiguration("optical_frame_name"),
            ],
            condition=IfCondition(LaunchConfiguration("publish_optical_tf")),
            output="screen",
        ),

        # YOLO node (active when detector == "yolo")
        Node(
            package="gem_perception_ros2",
            executable="yolo_perception_node",
            name="gem_perception_yolo",
            output="screen",
            parameters=[cfg, {"default_prompt": LaunchConfiguration("default_prompt")}],
            condition=IfCondition(is_yolo),
        ),
        # SAM node (active when detector == "sam")
        Node(
            package="gem_perception_ros2",
            executable="sam_perception_node",
            name="gem_perception_sam",
            output="screen",
            parameters=[cfg, {"default_prompt": LaunchConfiguration("default_prompt")}],
            condition=IfCondition(is_sam),
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
            condition=IfCondition(LaunchConfiguration("run_rviz")),
        ),
    ])
