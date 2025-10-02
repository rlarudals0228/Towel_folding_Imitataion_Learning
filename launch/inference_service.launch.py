from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_lerobot',
            executable='control_client',
            name='control_client_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='inference_server',
            name='inference_server_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='keyboard_listener',
            name='keyboard_listener_node',
            output='screen'
        )
    ])