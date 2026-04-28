#!/usr/bin/env python3
# launch/safety_controller_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value='',
        description='Vehicle identifier (e.g., e2, e4)'
    )

    # Safety controller node
    safety_controller_node = Node(
        package='adapt_full',
        executable='safety_controller',
        name='safety_controller',
        output='screen',
        parameters=[{
            'vehicle_name': LaunchConfiguration('vehicle_name'),
        }],
    )

    return LaunchDescription([
        vehicle_name_arg,
        safety_controller_node,
    ])
