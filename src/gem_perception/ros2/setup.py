from setuptools import find_packages, setup
from glob import glob

package_name = 'gem_perception_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ACRL',
    maintainer_email='ogawa3@illinois.edu',
    description='ROS2 (humble) text-promptable perception.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_perception_node = gem_perception_ros2.yolo_perception_node:main',
            'sam_perception_node  = gem_perception_ros2.sam_perception_node:main',
            'map_tf_broadcaster   = gem_perception_ros2.map_tf_broadcaster:main',
            'bev_overlay_node     = gem_perception_ros2.bev_overlay_node:main',
            'download_models      = gem_perception_ros2.download_models:main',
        ],
    },
)
