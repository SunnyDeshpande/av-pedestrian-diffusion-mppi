from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'adapt_full'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'waypoints'), glob('waypoints/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sunny Deshpande',
    maintainer_email='sunny@example.com',
    description='Adapt full integration package with lidar processing',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_processing = adapt_full.adapt_lidar_processing:main',
            'straight_path = adapt_full.adapt_straight_path:main',
            'stanley_controller = adapt_full.adapt_stanley_controller:main',
            'safety_controller = adapt_full.adapt_safety_controller:main',
            'lidar_camera_fusion = adapt_full.adapt_lidar_camera_fusion:main',
            'high_level_command = adapt_full.adapt_high_level_command:main',
            'camera_position_spoof = adapt_full.adapt_camera_position:main',
            'pedestrian_aware_path = adapt_full.adapt_pedestrian_aware_path:main',
        ],
    },
)
