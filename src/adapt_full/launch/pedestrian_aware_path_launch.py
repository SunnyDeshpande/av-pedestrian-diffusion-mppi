#!/usr/bin/env python3
# launch/pedestrian_aware_path_launch.py
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

    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed',
        default_value='3.0',
        description='Desired vehicle speed in m/s (max: 5.0)'
    )

    min_danger_angle_arg = DeclareLaunchArgument(
        'min_danger_angle',
        default_value='45',
        description='Minimum angle of danger zone in degrees'
    )

    max_danger_angle_arg = DeclareLaunchArgument(
        'max_danger_angle',
        default_value='85',
        description='Maximum angle of danger zone in degrees'
    )

    max_danger_distance_arg = DeclareLaunchArgument(
        'max_danger_distance',
        default_value='10',
        description='Maximum distance for danger zone in meters'
    )

    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'pedestrian_aware_params.yaml'
    ])

    # Pedestrian-aware path controller node
    pedestrian_aware_node = Node(
        package='adapt_full',
        executable='pedestrian_aware_path',
        name='pedestrian_aware_path',
        output='screen',
        parameters=[
            config_file,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'desired_speed': LaunchConfiguration('desired_speed'),
                'pedestrian/min_danger_angle': LaunchConfiguration('min_danger_angle'),
                'pedestrian/max_danger_angle': LaunchConfiguration('max_danger_angle'),
                'pedestrian/max_danger_distance': LaunchConfiguration('max_danger_distance'),
            }
        ],
    )

    return LaunchDescription([
        vehicle_name_arg,
        desired_speed_arg,
        min_danger_angle_arg,
        max_danger_angle_arg,
        max_danger_distance_arg,
        pedestrian_aware_node,
    ])
