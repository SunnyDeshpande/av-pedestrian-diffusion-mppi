from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'diffusion_prediction'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'scripts'),
            glob(os.path.join('scripts', '*.py'))),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sunny Deshpande',
    maintainer_email='sunnydeshpande9900@gmail.com',
    description='Conditional diffusion model for pedestrian trajectory prediction',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'infer_node = diffusion_prediction.infer_node:main',
            'train = diffusion_prediction.train:main',
            'finetune = diffusion_prediction.finetune:main',
            'eval = diffusion_prediction.eval:main',
        ],
    },
)
