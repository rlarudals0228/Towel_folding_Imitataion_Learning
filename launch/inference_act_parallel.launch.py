from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_lerobot',
            executable='control_parallel',
            name='control_parallel_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='inference_act_parallel',
            name='inference_act_parallel_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='keyboard_listener',
            name='keyboard_listener_node',
            output='screen'
        )
    ])