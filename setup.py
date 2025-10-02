import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ros2_lerobot'

def get_data_files_top_level(dir_name):
    base_dir = os.path.dirname(__file__)
    source_dir = os.path.join(base_dir, dir_name)
    data_files = []
    for root, dirs, files in os.walk(source_dir):
        if files:
            install_path = os.path.join(
                'share',
                package_name,
                os.path.relpath(root, base_dir)
            )
            file_paths = [os.path.join(root, f) for f in files]
            data_files.append((install_path, file_paths))
    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        *get_data_files_top_level('demo_data'),
        *get_data_files_top_level('ckpt'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dognwoo',
    maintainer_email='sondongwoo2024@gmail.com',
    description='ROS2 package with dataset and checkpoints',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collect_data = ros2_lerobot.collect_data:main',
            'keyboard_listener = ros2_lerobot.keyboard_listener:main',
            'kkm_keyboard_listener = ros2_lerobot.kkm_keyboard_listener:main',
            'inference_act = ros2_lerobot.inference_act:main',
            'inference_act_parallel = ros2_lerobot.inference_act_parallel:main',
            'control_parallel = ros2_lerobot.control_parallel:main',
            'inference_server = ros2_lerobot.inference_server:main',
            'control_client = ros2_lerobot.control_client:main',
            'kkm_control_client = ros2_lerobot.kkm_control_client:main',
            'realsense_towel_metrics = ros2_lerobot.realsense_towel_metrics:main',
        ],
    },
)
