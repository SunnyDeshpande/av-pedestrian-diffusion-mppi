#!/usr/bin/env python3
# launch/straight_path_launch.py
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
        default_value='2.0',
        description='Desired vehicle speed in m/s (max: 5.0)'
    )

    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'straight_path_params.yaml'
    ])

    # Straight path controller node
    straight_path_node = Node(
        package='adapt_full',
        executable='straight_path',
        name='adapt_straight_path',
        output='screen',
        parameters=[
            config_file,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'desired_speed': LaunchConfiguration('desired_speed'),
            }
        ],
    )

    return LaunchDescription([
        vehicle_name_arg,
        desired_speed_arg,
        straight_path_node,
    ])
