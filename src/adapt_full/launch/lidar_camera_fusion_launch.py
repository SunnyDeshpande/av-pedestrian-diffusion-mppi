#!/usr/bin/env python3
# launch/lidar_camera_fusion_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value='',
        description='Vehicle identifier (e.g., e2, e4)'
    )

    matching_threshold_arg = DeclareLaunchArgument(
        'matching_threshold',
        default_value='2.0',
        description='Maximum distance (meters) for matching Lidar and Camera detections'
    )

    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'sensor_fusion_params.yaml'
    ])

    # Lidar-Camera fusion node
    fusion_node = Node(
        package='adapt_full',
        executable='lidar_camera_fusion',
        name='sensor_fusion_node',
        output='screen',
        parameters=[
            config_file,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'matching_threshold': LaunchConfiguration('matching_threshold'),
            }
        ],
    )

    return LaunchDescription([
        vehicle_name_arg,
        matching_threshold_arg,
        fusion_node,
    ])
