#!/usr/bin/env python3
# launch/high_level_command_launch.py
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

    decision_rate_arg = DeclareLaunchArgument(
        'decision_rate_hz',
        default_value='20.0',
        description='Decision loop frequency in Hz'
    )

    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'high_level_decision_params.yaml'
    ])

    # High-level decision node
    decision_node = Node(
        package='adapt_full',
        executable='high_level_command',
        name='high_level_decision_node',
        output='screen',
        parameters=[
            config_file,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'decision_rate_hz': LaunchConfiguration('decision_rate_hz'),
            }
        ],
    )

    return LaunchDescription([
        vehicle_name_arg,
        decision_rate_arg,
        decision_node,
    ])
