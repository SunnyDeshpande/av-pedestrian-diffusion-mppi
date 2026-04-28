#!/usr/bin/env python3
# launch/adapt_full_launch.py
"""
Full Adapt system launch file
Launches all Adapt components with individual enable/disable flags
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

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

    controller_mode_arg = DeclareLaunchArgument(
        'controller_mode',
        default_value='stanley',
        description='Controller mode: stanley, straight_path'
    )

    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar',
        default_value='true',
        description='Enable lidar preprocessing node'
    )

    enable_stanley_arg = DeclareLaunchArgument(
        'enable_stanley',
        default_value='false',
        description='Enable Stanley path tracking controller'
    )

    enable_straight_path_arg = DeclareLaunchArgument(
        'enable_straight_path',
        default_value='false',
        description='Enable straight path controller'
    )

    enable_safety_arg = DeclareLaunchArgument(
        'enable_safety',
        default_value='false',
        description='Enable safety controller'
    )

    enable_fusion_arg = DeclareLaunchArgument(
        'enable_fusion',
        default_value='false',
        description='Enable lidar-camera fusion'
    )

    enable_high_level_arg = DeclareLaunchArgument(
        'enable_high_level',
        default_value='false',
        description='Enable high-level command interface'
    )

    # Get paths to config files
    lidar_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'lidar_params.yaml'
    ])

    stanley_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'stanley_controller_params.yaml'
    ])

    straight_path_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'straight_path_params.yaml'
    ])

    fusion_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'sensor_fusion_params.yaml'
    ])

    high_level_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'),
        'config',
        'high_level_decision_params.yaml'
    ])

    # Lidar preprocessing node
    lidar_node = Node(
        package='lidar_person_detection',
        executable='lidar_preprocessor',
        name='lidar_preprocessor',
        output='screen',
        parameters=[lidar_config],
        condition=IfCondition(LaunchConfiguration('enable_lidar'))
    )

    # Stanley controller node
    stanley_node = Node(
        package='adapt_full',
        executable='stanley_controller',
        name='stanley_controller_node',
        output='screen',
        parameters=[
            stanley_config,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'desired_speed': LaunchConfiguration('desired_speed'),
            }
        ],
        condition=IfCondition(LaunchConfiguration('enable_stanley'))
    )

    # Straight path controller node
    straight_path_node = Node(
        package='adapt_full',
        executable='straight_path',
        name='adapt_straight_path',
        output='screen',
        parameters=[
            straight_path_config,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
                'desired_speed': LaunchConfiguration('desired_speed'),
            }
        ],
        condition=IfCondition(LaunchConfiguration('enable_straight_path'))
    )

    # Safety controller node
    safety_node = Node(
        package='adapt_full',
        executable='safety_controller',
        name='safety_controller',
        output='screen',
        parameters=[{
            'vehicle_name': LaunchConfiguration('vehicle_name'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_safety'))
    )

    # Lidar-Camera fusion node
    fusion_node = Node(
        package='adapt_full',
        executable='lidar_camera_fusion',
        name='sensor_fusion_node',
        output='screen',
        parameters=[
            fusion_config,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
            }
        ],
        condition=IfCondition(LaunchConfiguration('enable_fusion'))
    )

    # High-level decision node
    high_level_node = Node(
        package='adapt_full',
        executable='high_level_command',
        name='high_level_decision_node',
        output='screen',
        parameters=[
            high_level_config,
            {
                'vehicle_name': LaunchConfiguration('vehicle_name'),
            }
        ],
        condition=IfCondition(LaunchConfiguration('enable_high_level'))
    )

    return LaunchDescription([
        vehicle_name_arg,
        desired_speed_arg,
        controller_mode_arg,
        enable_lidar_arg,
        enable_stanley_arg,
        enable_straight_path_arg,
        enable_safety_arg,
        enable_fusion_arg,
        enable_high_level_arg,
        lidar_node,
        stanley_node,
        straight_path_node,
        safety_node,
        fusion_node,
        high_level_node,
    ])
    
    
