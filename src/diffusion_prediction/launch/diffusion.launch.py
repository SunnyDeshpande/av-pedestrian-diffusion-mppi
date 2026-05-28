"""Launch the diffusion pedestrian-prediction inference node.

Resolves the default weights path from this launch file's source location so it
keeps working with `colcon build --symlink-install` (the install symlink is
realpath'd back to src/).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_weights() -> str:
    here = os.path.realpath(__file__)
    pkg_src = os.path.dirname(os.path.dirname(here))  # src/diffusion_prediction/
    return os.path.join(pkg_src, 'models', 'diffusion', 'av2_joint_v2', 'ema_best.pt')


def generate_launch_description():
    weights_arg = DeclareLaunchArgument(
        'weights', default_value=_default_weights(),
        description='Path to the diffusion model weights (.pt).',
    )
    mode_arg = DeclareLaunchArgument(
        'prediction_mode', default_value='joint',
        description='"joint" or "single" - must match the weights checkpoint.',
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='cuda:0',
        description='Torch device. Falls back to CPU if CUDA unavailable.',
    )
    use_fusion_paths_arg = DeclareLaunchArgument(
        'use_fusion_paths', default_value='true',
        description='Build history from /fusion_pedestrian_paths instead of internal tracker.',
    )
    temporal_alpha_arg = DeclareLaunchArgument(
        'temporal_alpha', default_value='0.55',
        description='Inter-frame EMA blend for predictions (0=all previous, 1=all current). '
                    'Higher = less lag, more jitter.',
    )
    anchor_arg = DeclareLaunchArgument(
        'anchor_to_current_position', default_value='true',
        description="If true, shift the predicted trajectory so its first point coincides "
                    "with the pedestrian's current position. Removes start-of-trajectory noise "
                    "and fusion-path smoothing lag while preserving predicted shape.",
    )

    infer_node = Node(
        package='diffusion_prediction',
        executable='infer_node',
        name='diffusion_predictor_node',
        output='screen',
        parameters=[{
            'weights': LaunchConfiguration('weights'),
            'prediction_mode': LaunchConfiguration('prediction_mode'),
            'device': LaunchConfiguration('device'),
            'use_fusion_paths': LaunchConfiguration('use_fusion_paths'),
            'temporal_alpha': LaunchConfiguration('temporal_alpha'),
            'anchor_to_current_position': LaunchConfiguration('anchor_to_current_position'),
        }],
    )

    return LaunchDescription([
        weights_arg,
        mode_arg,
        device_arg,
        use_fusion_paths_arg,
        temporal_alpha_arg,
        anchor_arg,
        infer_node,
    ])
