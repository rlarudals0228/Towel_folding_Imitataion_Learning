import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory
from cv_bridge import CvBridge
from std_msgs.msg import String
import traceback

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import os
import sys
import cv2
import datetime
import numpy as np

import threading

class CreateDataset(Node):
    def __init__(self):
        super().__init__('collect_data_node')
        
        self.data_mutex = threading.Lock()
        # ROS 2 Topics
        self.wrist_image_sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_rect_raw/compressed',
            self.wrist_image_callback,
            10
        )
        self.wrist_image_sub
        
        self.follower_joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.follower_joint_callback,
            10
        )
        self.follower_joint_sub

        self.leader_joint_sub = self.create_subscription(
            JointTrajectory,
            '/leader/joint_trajectory',
            self.leader_joint_callback,
            10
        )
        self.leader_joint_sub
        
        self.keyboard_sub = self.create_subscription(
            String,
            '/keyboard_command',
            self.keyboard_callback,
            10
        )
        self.keyboard_sub
        
        timer_period = 0.0333  # seconds 
        self.call_timer = self.create_timer(timer_period, self.collect_data)
        
        '''LeRobot Init'''
        self.NUM_DEMO = 10 # Number of demonstrations to collect
        REPO_NAME = 'omy_real'
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        current_file_path = os.path.abspath(__file__)
        ros2_lerobot_src_path = os.path.join(os.path.dirname(current_file_path), '..')
        self.ROOT = os.path.join(
            ros2_lerobot_src_path,
            'demo_data',
            f'{timestamp}_episode_{self.NUM_DEMO}'
        )
        
        # Task name
        self.TASK_NAME = 'Put Object in the area'
        
        self.dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = self.ROOT, 
                robot_type="omy",
                fps=30,
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["state"], # 6 joint angles and 1 gripper
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 joint angles and 1 gripper
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
        
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'rh_r1_joint']
        
        self.bridge = CvBridge()
        self.top_view_img = None
        self.wrist_img = None
        self.follower_joint_positions = None
        self.leader_joint_positions = None
        self.follower_joint_velocitys = None
        self.observation_flag = {'wrist_img':False,
                                 'follower_joint':False,
                                 'leader_joint':False,
                                 'keyboard':False}
        
        self.record_flag = False
        self.episode_id = 0
        
        self.start_episod = False
        self.done = False
        self.reset = False

        self.get_logger().info('=== Ready Collect Data ===')
    
    def wrist_image_callback(self, msg):
        try:
            with self.data_mutex:
                np_arr = np.frombuffer(msg.data, np.uint8)
                self.wrist_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # self.get_logger().info(f'wrist image test: {self.wrist_img.shape}')
            self.observation_flag['wrist_img'] = True
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            
            self.observation_flag['wrist_img'] = False

    def follower_joint_callback(self, msg):
        try:
            if set(self.joint_names).issubset(set(msg.name)):
                with self.data_mutex:
                    self.follower_joint_positions = [msg.position[msg.name.index(j)] for j in self.joint_names]
                    # self.follower_joint_velocitys = [msg.velocity[msg.name.index(j)] for j in self.joint_names]
                    
                    self.observation_flag['follower_joint'] = True
                
                # self.get_logger().info(f'Received follower joint positions: {self.follower_joint_positions}')
                # self.get_logger().info(f'Received follower joint velocitys: {self.follower_joint_velocitys}')
        except Exception as e:
            self.get_logger().error(f"Failed to sub follower joint: {e}")
            
            self.observation_flag['follower_joint'] = False
        
    def leader_joint_callback(self, msg):
        try:
            if set(self.joint_names).issubset(set(msg.joint_names)):
                with self.data_mutex:
                    self.leader_joint_positions = [msg.points[0].positions[msg.joint_names.index(j)] for j in self.joint_names]
                    # self.leader_joint_velocitys = [msg.points[0].velocities[msg.joint_names.index(j)] for j in self.joint_names]
                    
                    self.observation_flag['leader_joint'] = True
                    
                # self.get_logger().info(f'Received leader joint positions: {self.leader_joint_positions}')
                # self.get_logger().info(f'Received leader joint velocities: {self.leader_joint_velocities}')
                
        except Exception as e:
            self.get_logger().error(f"Failed to sub leader joint: {e}")
            
            self.observation_flag['leader_joint'] = False
    
    def keyboard_callback(self, msg):
        try:
            if self.episode_id < self.NUM_DEMO:
                if msg.data == 'start':
                    self.start_episod = True
                    self.get_logger().info(f"Start. Episode: {self.episode_id}")
                    
                elif msg.data == 'reset':
                    self.reset = True
                    self.get_logger().info(f"Reset. Episode: {self.episode_id}")
                    
                elif msg.data == 'done':
                    self.done = True
                    self.get_logger().info(f"Done. Episode: {self.episode_id}")

                self.observation_flag['keyboard'] = True
            
        except Exception as e:
            self.get_logger().error(f"Failed to sub keyboard: {e}")
    
            self.observation_flag['keyboard'] = False

    def collect_data(self):
        if self.episode_id < self.NUM_DEMO:
            
            # Wait for the data
            for item in self.observation_flag.items():
                if item is False: return            
            
            try:
                if self.start_episod:
                    if self.done:
                        
                        # Save the episode data and reset the environment
                        self.done = False
                        
                        if self.dataset.episode_buffer["size"] == 0:
                            self.get_logger().info(f"Episode Buffer Empty! Nothing has been saved.")
                            return
                        
                        self.dataset.save_episode()
                        self.get_logger().info(f"[Save {self.episode_id} Episode]")
                        self.episode_id += 1
                        self.start_episod = False
                        
                        if self.episode_id >= self.NUM_DEMO:
                            self.get_logger().info("All demonstrations collected. Shutting down...")
                            self.get_logger().info(f"Save as: {self.ROOT}")
                            self.get_logger().info("!! Press Ctrl+C to exit. !!")
                            self.shutdown_node()
                    
                    if not self.record_flag:
                        self.record_flag = True

                    if self.reset: 
                        self.dataset.clear_episode_buffer()
                        self.start_episod = False
                        self.record_flag = False
                        self.reset = False
                        
                    with self.data_mutex:
                        follower_positions_np = np.array(self.follower_joint_positions, dtype=np.float32)
                        leader_positions_np = np.array(self.leader_joint_positions, dtype=np.float32)
                        
                    state = follower_positions_np.copy()
                    action = leader_positions_np.copy()
                    
                    # image flip (for stereo vision using a mirror)
                    with self.data_mutex:
                        flip_image = self.wrist_img.copy()
                    region = flip_image[:, 470:].copy()
                    region = cv2.flip(region, 1)
                    flip_image[:, 470:] = region
                    flip_image = cv2.resize(flip_image, (256, 256))
                    flip_image = cv2.cvtColor(flip_image, cv2.COLOR_RGB2BGR)
                    image = flip_image
                    
                    # not to flip
                    # image = self.wrist_img                
                    
                    if self.record_flag:
                        # Add the frame to the dataset
                        self.dataset.add_frame( {
                                "observation.image": image,
                                "observation.state": state, 
                                "action": action,
                            }, task=self.TASK_NAME,
                        )
                        
            except Exception as e:
                traceback.print_exc()
                
    def shutdown_node(self):
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = CreateDataset()
 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt detected. Shutting down...")
    finally:
        # cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)
        
if __name__ == '__main__':
    main()
