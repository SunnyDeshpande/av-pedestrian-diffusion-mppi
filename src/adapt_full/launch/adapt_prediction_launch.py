#!/usr/bin/env python3
# launch/adapt_prediction_launch.py
"""
Unified launch file with 3 selectable prediction modes and controller choice.

Prediction modes:
  single-default   - Original constant-velocity predictor (multi-ped extended)
  single-diffusion - Per-pedestrian diffusion model (TrajectoryDenoiser)
  multi-diffusion  - Joint multi-agent diffusion model (JointTrajectoryDenoiser)

Controllers:
  mppi    - MPPI with velocity-aware pedestrian obstacle cost
  stanley - Stanley cross-track + PID speed (no obstacle avoidance)

Usage examples:
  ros2 launch adapt_full adapt_prediction_launch.py prediction_mode:=single-default controller:=mppi
  ros2 launch adapt_full adapt_prediction_launch.py prediction_mode:=multi-diffusion controller:=mppi \\
      weights:=/path/to/ema_best.pt
  ros2 launch adapt_full adapt_prediction_launch.py prediction_mode:=single-diffusion controller:=stanley \\
      weights:=/path/to/ema_best.pt
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def _launch_setup(context, *args, **kwargs):
    """Resolve prediction_mode and controller at launch time to build the
    right set of nodes."""
    prediction_mode = LaunchConfiguration('prediction_mode').perform(context)
    controller = LaunchConfiguration('controller').perform(context)
    vehicle_name = LaunchConfiguration('vehicle_name').perform(context)
    desired_speed = LaunchConfiguration('desired_speed').perform(context)
    weights = LaunchConfiguration('weights').perform(context)
    device = LaunchConfiguration('device').perform(context)
    lowlevel_mode = LaunchConfiguration('lowlevel_mode').perform(context)
    enable_lidar = LaunchConfiguration('enable_lidar').perform(context)
    enable_fusion = LaunchConfiguration('enable_fusion').perform(context)
    enable_safety = LaunchConfiguration('enable_safety').perform(context)
    enable_high_level = LaunchConfiguration('enable_high_level').perform(context)
    enable_rviz = LaunchConfiguration('enable_rviz').perform(context)

    nodes = []

    # --- Perception nodes (conditional) ---
    lidar_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'lidar_params.yaml',
    ])
    fusion_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'sensor_fusion_params.yaml',
    ])
    high_level_config = PathJoinSubstitution([
        FindPackageShare('adapt_full'), 'config', 'high_level_decision_params.yaml',
    ])

    nodes.append(Node(
        package='adapt_full',
        executable='lidar_processing',
        name='lidar_processing',
        output='screen',
        parameters=[lidar_config],
        condition=IfCondition(enable_lidar),
    ))
    nodes.append(Node(
        package='adapt_full',
        executable='lidar_camera_fusion',
        name='sensor_fusion_node',
        output='screen',
        parameters=[fusion_config,
                    {'vehicle_name': vehicle_name}],
        condition=IfCondition(enable_fusion),
    ))
    nodes.append(Node(
        package='adapt_full',
        executable='safety_controller',
        name='safety_controller',
        output='screen',
        parameters=[{'vehicle_name': vehicle_name}],
        condition=IfCondition(enable_safety),
    ))
    nodes.append(Node(
        package='adapt_full',
        executable='high_level_command',
        name='high_level_decision_node',
        output='screen',
        parameters=[high_level_config,
                    {'vehicle_name': vehicle_name}],
        condition=IfCondition(enable_high_level),
    ))

    # --- Prediction node ---
    if prediction_mode == 'single-default':
        # Original constant-velocity predictor (multi-ped extended)
        nodes.append(Node(
            package='yolo_person_detector',
            executable='pedestrian_behaviour_predictor',
            name='pedestrian_behaviour_predictor',
            output='screen',
        ))
    elif prediction_mode == 'single-diffusion':
        # Per-pedestrian diffusion model
        nodes.append(Node(
            package='diffusion_prediction',
            executable='infer_node',
            name='diffusion_predictor_node',
            output='screen',
            parameters=[{
                'weights': weights,
                'device': device,
                'prediction_mode': 'single',
                'K': 20,
                'ddim_steps': 10,
            }],
        ))
    elif prediction_mode == 'multi-diffusion':
        # Joint multi-agent diffusion model
        nodes.append(Node(
            package='diffusion_prediction',
            executable='infer_node',
            name='diffusion_predictor_node',
            output='screen',
            parameters=[{
                'weights': weights,
                'device': device,
                'prediction_mode': 'joint',
                'max_agents': 16,
                'K': 20,
                'ddim_steps': 10,
            }],
        ))
    else:
        raise ValueError(
            f"Unknown prediction_mode '{prediction_mode}'. "
            "Use: single-default, single-diffusion, multi-diffusion"
        )

    # --- Controller ---
    if controller == 'mppi':
        # When any predictor is active, MPPI reads from the tensor topic
        prediction_source = 'predicted' if prediction_mode != 'raw-only' else 'raw'
        nodes.append(Node(
            package='mppi_controller',
            executable='adapt_mppi_node',
            name='adapt_mppi_node',
            output='screen',
            parameters=[{
                'vehicle_name': vehicle_name,
                'desired_speed': float(desired_speed),
                'rate_hz': 10.0,
                'mppi/K': 600,
                'mppi/H': 30,
                'mppi/dt': 0.1,
                'mppi/device': device,
                'prediction_source': prediction_source,
            }],
        ))
    elif controller == 'mppi-split':
        # Split mode: planner publishes to /mppi/control_output,
        # bridge node converts to PACMod commands
        prediction_source = 'predicted' if prediction_mode != 'raw-only' else 'raw'
        nodes.append(Node(
            package='mppi_controller',
            executable='mppi_planner_node',
            name='mppi_planner_node',
            output='screen',
            parameters=[{
                'vehicle_name': vehicle_name,
                'desired_speed': float(desired_speed),
                'rate_hz': 10.0,
                'mppi/K': 600,
                'mppi/H': 30,
                'mppi/dt': 0.1,
                'mppi/device': device,
                'prediction_source': prediction_source,
            }],
        ))
        nodes.append(Node(
            package='mppi_controller',
            executable='pacmod_bridge_node',
            name='pacmod_bridge_node',
            output='screen',
            parameters=[{
                'mode': lowlevel_mode,
                'desired_speed': float(desired_speed),
            }],
        ))
    elif controller == 'stanley':
        stanley_config = PathJoinSubstitution([
            FindPackageShare('adapt_full'), 'config',
            'stanley_controller_params.yaml',
        ])
        nodes.append(Node(
            package='adapt_full',
            executable='stanley_controller',
            name='stanley_controller_node',
            output='screen',
            parameters=[stanley_config, {
                'vehicle_name': vehicle_name,
                'desired_speed': float(desired_speed),
            }],
        ))
    else:
        raise ValueError(
            f"Unknown controller '{controller}'. Use: mppi, mppi-split, stanley"
        )

    # --- RViz ---
    if enable_rviz.lower() == 'true':
        rviz_config = PathJoinSubstitution([
            FindPackageShare('mppi_controller'), 'rviz', 'adapt_main.rviz',
        ])
        nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'prediction_mode', default_value='single-default',
            description='Prediction mode: single-default, single-diffusion, multi-diffusion',
        ),
        DeclareLaunchArgument(
            'controller', default_value='mppi',
            description='Controller: mppi, mppi-split, stanley',
        ),
        DeclareLaunchArgument(
            'lowlevel_mode', default_value='pid',
            description='Low-level throttle mode for mppi-split: pid, linear',
        ),
        DeclareLaunchArgument(
            'vehicle_name', default_value='e4',
            description='Vehicle identifier (e.g., e2, e4)',
        ),
        DeclareLaunchArgument(
            'desired_speed', default_value='2.0',
            description='Desired cruise speed, m/s',
        ),
        DeclareLaunchArgument(
            'weights', default_value='',
            description='Path to diffusion model weights (.pt). Required for diffusion modes.',
        ),
        DeclareLaunchArgument(
            'device', default_value='',
            description="Torch device. '' = auto (cuda if available else cpu).",
        ),
        DeclareLaunchArgument('enable_lidar', default_value='true'),
        DeclareLaunchArgument('enable_fusion', default_value='false'),
        DeclareLaunchArgument('enable_safety', default_value='false'),
        DeclareLaunchArgument('enable_high_level', default_value='false'),
        DeclareLaunchArgument(
            'enable_rviz', default_value='true',
            description='Auto-launch RViz2 with adapt_main.rviz',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
