#!/usr/bin/env python3
# launch/adapt_mppi_launch.py
"""
Adapt autonomy stack launch file - MPPI replaces Stanley.

Starts `adapt_mppi_node` (pkg `mppi_controller`), which speaks the
AutoShield topic contract (NavSatFix + INSNavGeod + VehicleSpeedRpt +
/pacmod/enabled) and consumes /fusion_pedestrian_position as the MPPI
obstacle source. The rest of the AutoShield stack (fusion, high-level
decision, safety) stays available behind enable flags.

Also auto-launches RViz with adapt_main.rviz (sensor viz + MPPI
chosen/sampled trajectories + obstacle/pedestrian markers). Disable
with `enable_rviz:=false`.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name', default_value='e4',
        description='Vehicle identifier (e.g., e2, e4)',
    )
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='2.0',
        description='Desired cruise speed, m/s (MPPI v_ref).',
    )
    enable_lidar_arg = DeclareLaunchArgument(
        'enable_lidar', default_value='true',
    )
    enable_mppi_arg = DeclareLaunchArgument(
        'enable_mppi', default_value='true',
        description='Enable Adapt MPPI lateral+longitudinal controller.',
    )
    enable_safety_arg = DeclareLaunchArgument(
        'enable_safety', default_value='false',
    )
    enable_fusion_arg = DeclareLaunchArgument(
        'enable_fusion', default_value='false',
    )
    enable_high_level_arg = DeclareLaunchArgument(
        'enable_high_level', default_value='false',
    )
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz', default_value='true',
        description='Auto-launch RViz2 with adapt_main.rviz',
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='',
        description="Torch device for MPPI. '' = auto (cuda if available else cpu); 'cuda:0' forces GPU; 'cpu' forces CPU.",
    )

    lidar_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'lidar_params.yaml',
    ])
    fusion_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'sensor_fusion_params.yaml',
    ])
    high_level_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'high_level_decision_params.yaml',
    ])

    # Note: adapt's original launch referenced a non-existent
    # `lidar_person_detection` package. The real node lives inside
    # adapt_full itself - this corrected entry points at it.
    lidar_node = Node(
        package='adapt_full',
        executable='lidar_processing',
        name='lidar_processing',
        output='screen',
        parameters=[lidar_config],
        condition=IfCondition(LaunchConfiguration('enable_lidar')),
    )

    # Adapt MPPI replaces Stanley. Native adapt topic contract -
    # no bridge nodes needed.
    mppi_node = Node(
        package='mppi_controller',
        executable='adapt_mppi_node',
        name='adapt_mppi_node',
        output='screen',
        parameters=[{
            'vehicle_name': LaunchConfiguration('vehicle_name'),
            'desired_speed': LaunchConfiguration('desired_speed'),
            'rate_hz': 10.0,
            'mppi/K': 600,
            'mppi/H': 30,
            'mppi/dt': 0.1,
            'mppi/device': LaunchConfiguration('device'),
        }],
        condition=IfCondition(LaunchConfiguration('enable_mppi')),
    )

    safety_node = Node(
        package='adapt_full',
        executable='safety_controller',
        name='safety_controller',
        output='screen',
        parameters=[{'vehicle_name': LaunchConfiguration('vehicle_name')}],
        condition=IfCondition(LaunchConfiguration('enable_safety')),
    )

    fusion_node = Node(
        package='adapt_full',
        executable='lidar_camera_fusion',
        name='sensor_fusion_node',
        output='screen',
        parameters=[fusion_config,
                    {'vehicle_name': LaunchConfiguration('vehicle_name')}],
        condition=IfCondition(LaunchConfiguration('enable_fusion')),
    )

    high_level_node = Node(
        package='adapt_full',
        executable='high_level_command',
        name='high_level_decision_node',
        output='screen',
        parameters=[high_level_config,
                    {'vehicle_name': LaunchConfiguration('vehicle_name')}],
        condition=IfCondition(LaunchConfiguration('enable_high_level')),
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare('mppi_controller'), 'rviz', 'adapt_main.rviz',
    ])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
    )

    return LaunchDescription([
        vehicle_name_arg,
        desired_speed_arg,
        enable_lidar_arg,
        enable_mppi_arg,
        enable_safety_arg,
        enable_fusion_arg,
        enable_high_level_arg,
        enable_rviz_arg,
        device_arg,
        lidar_node,
        mppi_node,
        safety_node,
        fusion_node,
        high_level_node,
        rviz_node,
    ])
