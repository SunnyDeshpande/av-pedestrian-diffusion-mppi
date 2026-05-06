# Copyright 2023 Ouster, Inc.
#

"""Launch a sensor node along with os_cloud and os_"""

from pathlib import Path
import launch
import lifecycle_msgs.msg
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import LifecycleNode, ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, EmitEvent, LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution, PythonExpression
from launch.events import matches_action
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


import os


def generate_launch_description():
    """
    Generate launch description for running ouster_ros components separately each
    component will run in a separate process).
    """
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value=EnvironmentVariable('VEHICLE_NAME'),
        description='Name of the vehicle used to locate the parameter file '
    )
    vehicle_name = LaunchConfiguration('vehicle_name')

    init_gps_arg = DeclareLaunchArgument(
        'gps_init',
        default_value='true',
        description='Set to true to startup GPS'
    )
    gps_init = LaunchConfiguration('gps_init')
    init_zed_arg = DeclareLaunchArgument(
        'zed_init',
        default_value='true',
        description='Set to true to startup GPS'
    )
    zed_init = LaunchConfiguration('zed_init')

    ouster_ros_pkg_dir = get_package_share_directory('basic_launch')
    ouster_params_file = PathJoinSubstitution([
        ouster_ros_pkg_dir,
        'config',
        vehicle_name,
        'ouster',
        'driver_params.yaml'
    ])
    

    os_driver = LifecycleNode(
        package='ouster_ros',
        executable='os_driver',
        name='os_driver',
        namespace='ouster',
        parameters=[ouster_params_file],
        output='screen',
    )

    os_sensor_configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(os_driver),
            transition_id=lifecycle_msgs.msg.Transition.TRANSITION_CONFIGURE,
        )
    )

    os_sensor_activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=os_driver, goal_state='inactive',
            entities=[
                LogInfo(msg="os_driver activating..."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(os_driver),
                    transition_id=lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
                )),
            ],
            handle_once=True
        )
    )

    os_sensor_finalized_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=os_driver, goal_state='finalized',
            entities=[
                LogInfo(
                    msg="Failed to communicate with the sensor in a timely manner."),
                EmitEvent(event=launch.events.Shutdown(
                    reason="Couldn't communicate with sensor"))
            ],
        )
    )
    lucid_cam_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('basic_launch'),
                    'launch',
                    'perception',
                    'corner_cameras.launch.py'
                ])
            )
    )
    front_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('depthai_ros_driver'), 'launch'),
            '/rgbd_pcl.launch.py']),
        launch_arguments={
            'rectify_rgb': 'true',
            'parent_frame': 'front_camera_link',
            'publish_tf_from_calibration': 'true',
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", vehicle_name, "' == 'e4'"])
        )
    )
    tf2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('basic_launch'), 'launch', 'tf2.launch.py'
            ])
        ),
        launch_arguments={'vehicle_name': vehicle_name}.items(),
    )
    zed_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('basic_launch'),
                'launch',
                'perception',
                'zed_camera.launch.py'
            ])
        ),
        launch_arguments={
            'camera_model' : 'zed2'}.items(),
        condition=IfCondition(
            PythonExpression(["'", vehicle_name, "' == 'e2' and ",
                              "'", zed_init,"' == 'true'"
                              ]),
        )
    )
    rviz_display_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch'),
            '/rviz_display.launch.py'])
    )
    gnss_config=PathJoinSubstitution([
        get_package_share_directory('basic_launch'),
        'config',
        vehicle_name,
        'septentrio_gnss/',
        'ins.yaml'
        ])
    gnss_node = ComposableNode(
        name='septentrio_gnss_driver',
        package='septentrio_gnss_driver', 
        plugin='rosaic_node::ROSaicNode',
        parameters=[gnss_config],
        condition=IfCondition(gps_init)
    )
    
    gem_gnss_image_node = Node(
        package='gem_gnss_image',
        executable='gem_gnss_image',
        output='screen',
        name='gem_gnss_image_node',
        condition=IfCondition(gps_init)
    )

    gnss_container = ComposableNodeContainer(
        name='septentrio_gnss_driver_container',
        namespace='septentrio_gnss_driver',
        package='rclcpp_components',
        executable='component_container',
        emulate_tty=True,
        sigterm_timeout = '20',
        composable_node_descriptions=[gnss_node],
        output='screen',
        condition=IfCondition(gps_init)
    )
    launch_lucid_on_active = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=os_driver, goal_state='active',
            entities=[
                LogInfo(msg="os_driver is active. Launching Lucid corner cameras..."),
                rviz_display_launch,
                gnss_container,
                gem_gnss_image_node,
                lucid_cam_launch,
                front_camera_launch,
                zed_camera_launch,
                tf2_launch,
            ],
            handle_once=True
        )
    )

    
    return launch.LaunchDescription([
        vehicle_name_arg,
        init_gps_arg,
        init_zed_arg,
        os_driver,
        os_sensor_configure_event,
        os_sensor_activate_event,
        os_sensor_finalized_event,
        launch_lucid_on_active
    ])
