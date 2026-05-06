from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo_person_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sunny Deshpande',
    maintainer_email='you@example.com',
    description='YOLOv11 person detector for ROS 2',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'rgbd_pedestrian_detector = yolo_person_detector.rgbd_pedestrian_detector:main',
            'pedestrian_behaviour_predictor = yolo_person_detector.pedestrian_behaviour_predictor:main',
        ],
    },
)
