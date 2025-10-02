from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_lerobot',
            executable='keyboard_listener',
            name='keyboard_listener_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='inference_act',
            name='inference_act_node',
            output='screen'
        )
    ])