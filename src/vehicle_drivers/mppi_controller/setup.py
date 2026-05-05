from setuptools import setup
from glob import glob

package_name = 'mppi_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Sunny Deshpande',
    maintainer_email='sunny@example.com',
    description='MPPI motion planner for the GEM e4 (Adapt Phase 1).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'adapt_mppi_node = mppi_controller.adapt_mppi_node:main',
            'adapt_mppi_generic_node = mppi_controller.adapt_mppi_generic_node:main',
            'mppi_planner_node = mppi_controller.mppi_planner_node:main',
            'pacmod_bridge_node = mppi_controller.pacmod_bridge_node:main',
        ],
    },
)
