from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_lerobot',
            executable='kkm_control_client',
            name='control_client_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='inference_server',
            name='inference_server_node_flatten',
            parameters=[
                {'config_name': 'towel_flattening_config.yaml'},
                {'service_name': 'run_inference_towel_flattening'},
            ],
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='inference_server',
            name='inference_node_folding',  
            parameters=[
                {'config_name': 'towel_folding_config.yaml'},
                {'service_name': 'run_inference_towel_folding'},
            ],
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='kkm_keyboard_listener',
            name='keyboard_listener_node',
            output='screen'
        ),
        Node(
            package='ros2_lerobot',
            executable='realsense_towel_metrics',
            output='screen',
            arguments=[
                '--color', '/camera/external_camera/color/image_rect_raw',
                '--depth', '/camera/external_camera/depth/image_rect_raw',
                '--info', '/camera/external_camera/depth/camera_info',
                '--rect-thr', '0.85',
                '--std-thr-mm', '7.0',
                '--range-thr-mm', '17.0',
                '--overlay-topic', '/towel/overlay',
                '--metric-topic', '/towel/metrics',
                '--decision-topic', '/towel/decision'
            ]
        ),
    ])
