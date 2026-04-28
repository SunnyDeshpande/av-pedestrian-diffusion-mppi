#!/usr/bin/env python3
# launch/stanley_controller_launch.py
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

    waypoint_file_arg = DeclareLaunchArgument(
        'waypoint_file',
        default_value='track.csv',
        description='Waypoint file name'
    )

    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'stanley_controller_params.yaml'
    ])

    # Stanley controller node
    stanley_controller_node = Node(
        package='adapt_full',
        executable='stanley_controller',
        name='stanley_controller_node',
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
        waypoint_file_arg,
        stanley_controller_node,
    ])
