import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from ros2_lerobot_interfaces.srv import Inference

import torchvision
import torch
import os
import time
import cv2
import yaml
import datetime
import numpy as np
from collections import deque

from ros2_lerobot.action_optimizer import ActionOptimizer
from ros2_lerobot.math_func import finite_difference_derivative

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

class InferenceACT(Node):
    def __init__(self):
        super().__init__('inference_act_parallel_node')
        
        self.get_logger().info('Starting Inference ACT Node...')
        # ROS parameter get (config path)
        
        self.declare_parameter('config_name', "towel_flattening_config.yaml")
        self.declare_parameter('service_name', 'run_inference')
        
        config_name = self.get_parameter('config_name').get_parameter_value().string_value
        service_name = self.get_parameter('service_name').get_parameter_value().string_value
        
        
        # === Init Configs ====
        current_file_path = os.path.abspath(__file__)
        self.pkg_root_path = os.path.join(os.path.dirname(current_file_path), '..')
        self.get_logger().info(f'Using config path from ROS parameter: {config_name}')
            
        config_path = os.path.join(self.pkg_root_path, 'config', config_name)
        
            #config_path = os.path.join(self.pkg_root_path, 'config', "experiment_config.yaml")
            #config_path = os.path.join(self.pkg_root_path, 'config', "towel_folding_config.yaml")
            #config_path = os.path.join(self.pkg_root_path, 'config', "towel_flattening_DAgger_config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.enable_optimizer = self.config["enable_optimizer"]
        self.enable_actions_concat = self.config["enable_actions_concat"]
        self.enable_spline = self.config["enable_spline"]
        self.enable_temporal_ensemble = self.config["enable_temporal_ensemble"]

        policy_dir = self.config["policy_dir"]
        dataset_dir = self.config["dataset_dir"]

        self.chunk_size = self.config["chunk_size"]
        self.n_action_step = self.config["n_action_step"]
        self.blending_horizon = self.config["blending_horizon"]
        
        self.action_dim = self.config["action_dim"]
        self.len_delay_time = self.config["len_delay_time"]

        # === Execution State Flags ===
        self.observation_flag = {
            'wrist_img': False,
            'follower_joint': False,
            'keyboard': False
        }

        # === ROS Interface ===
        self.wrist_image_sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_rect_raw/compressed',
            self.wrist_image_callback,
            10
        )
        self.follower_joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.follower_joint_callback,
            10
        )
        
        self.srv = self.create_service(Inference, service_name, self.inference_callback)
        self.get_logger().info('Inference service is ready.')
        
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'rh_r1_joint']

        # === Path Configurations ===
        pretrained_folder_name = os.path.join(
            self.pkg_root_path, 
            'ckpt', 
            policy_dir
        )
        dataset_path = os.path.join(
            self.pkg_root_path,
            'demo_data',
            dataset_dir
        )

        # === Utility ===
        self.bridge = CvBridge()
        self.device = torch.device("cuda:0")
        
        self.action_queue = deque([], maxlen=self.n_action_step)
        self.pre_actions = np.array([], dtype=np.int8)
        
        self.log = {"raw_actions": [],
                    "ref.value": [],
                    "solved": [],
                    "TE_pre_actions": [],
                    "solve_time": []}
        
        # === Policy Configuration & Loading ===
        # dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=dataset_path)
        dataset_metadata = LeRobotDatasetMetadata("OMY", root=dataset_path)
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

        self.temporal_ensemble_coeff = 0.9
        if self.enable_temporal_ensemble:
            cfg = ACTConfig(input_features=input_features, 
                            output_features=output_features, 
                            chunk_size=self.chunk_size,
                            n_action_steps=1,
                            temporal_ensemble_coeff=self.temporal_ensemble_coeff)
        else:
            cfg = ACTConfig(input_features=input_features, 
                            output_features=output_features, 
                            chunk_size=self.chunk_size,
                            n_action_steps=self.n_action_step)

        self.policy = ACTPolicy.from_pretrained(pretrained_name_or_path=pretrained_folder_name,
                                                config=cfg,
                                                dataset_stats=dataset_metadata.stats)
        self.policy.to(self.device)
        self.policy.reset()
        self.policy.eval()
        
        # === Action Optimizer Parameters ===
        if self.enable_optimizer:
            self.solver = "CLARABEL"
            self.action_optimizer = ActionOptimizer(
                solver=self.solver,
                chunk_size=self.n_action_step,
                blending_horizon=self.blending_horizon,
                action_dim=self.action_dim,
                len_time_delay=self.len_delay_time
            )

            self.solve_times = []

    def wrist_image_callback(self, msg):
        try:
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
                self.follower_joint_positions = [msg.position[msg.name.index(j)] for j in self.joint_names]
                self.follower_joint_velocities = [msg.velocity[msg.name.index(j)] for j in self.joint_names]
                
                self.observation_flag['follower_joint'] = True
                
                # self.get_logger().info(f'🦿 Received leader joint positions: {self.follower_joint_positions}')
                # self.get_logger().info(f'🦿 Received leader joint velocitys: {self.follower_joint_velocities}')
                
        except Exception as e:
            self.get_logger().error(f"Failed to sub follower joint: {e}")
            
            self.observation_flag['follower_joint'] = False
            
    def inference_callback(self, request, response):
        
        img_transform = torchvision.transforms.ToTensor()

        wrist_img = cv2.resize(self.wrist_img, (256, 256))
        
        region = self.wrist_img[:, 470:]
        region = cv2.flip(region, 1)
        flip_img = self.wrist_img.copy()
        flip_img[:, 470:] = region
        flip_img = cv2.resize(flip_img, (256, 256))
        
        wrist_img = img_transform(wrist_img)
        flip_img = img_transform(flip_img)
        
        follower_positions_np = np.array(self.follower_joint_positions, dtype=np.float32)
        follower_velocitys_np = np.array(self.follower_joint_velocities, dtype=np.float32)
        state = np.concatenate((follower_positions_np, follower_velocitys_np), axis=0)
        
        # # throwing ball
        #batch = {
        #     'observation.state': torch.from_numpy(np.array([state], dtype=np.float32)).to(self.device),
        #     'observation.wrist_image': wrist_img.unsqueeze(0).to(self.device),
        #     'observation.flip_image': flip_img.unsqueeze(0).to(self.device),
        #}
    
        # # towel folding/flattening
        batch = {
            'observation.state': torch.from_numpy(np.array([state[:self.action_dim]], dtype=np.float32)).to(self.device),
            'observation.image': flip_img.unsqueeze(0).to(self.device),
        }
        
        if self.enable_temporal_ensemble:
            self.raw_actions, TE_pre_actions = self.policy.select_action(batch)
            self.log["TE_pre_actions"].extend(TE_pre_actions.detach().cpu().numpy())
            self.raw_actions = self.raw_actions.detach().cpu().numpy()
        else:
            self.raw_actions = self.policy.get_action_chuck(batch, self.n_action_step).detach().cpu().numpy()[:, :self.action_dim]
        
        self.get_logger().info(f"raw_actions: {self.raw_actions}")
        
        self.log["raw_actions"].extend(self.raw_actions)
        self.actions = self.raw_actions
        
        pre_actions = np.array(request.pre_actions.data, dtype=np.float64)
        
        self.action_queue.clear()
        
        if self.enable_optimizer:            
            self.get_logger().info(f"pre_actions: {pre_actions.shape}")
            if np.isnan(pre_actions).any():
                temp_pre_actions = np.tile(follower_positions_np, (self.n_action_step, 1))
                self.actions, ref_value = self.action_optimizer.solve(self.actions, temp_pre_actions, self.blending_horizon)
            else:
                s = time.time()
                pre_actions = pre_actions.reshape((self.n_action_step, 3, self.action_dim))
                self.actions, ref_value = self.action_optimizer.solve(self.actions, pre_actions[:, 0, :], self.blending_horizon)
                elapsed_time = time.time() - s
                self.solve_times.append(elapsed_time)
                
                self.get_logger().info(f"Solved Time: {elapsed_time:.6f} sec")
                self.get_logger().info(f"Min: {min(self.solve_times):.6f}, Max: {max(self.solve_times):.6f}, Mean: {sum(self.solve_times)/len(self.solve_times):.6f}")

                self.log["solve_time"].extend((self.solve_times, max(self.solve_times), sum(self.solve_times)/len(self.solve_times)))
                
            self.log["solved"].extend(self.actions)
            self.log["ref.value"].extend(ref_value)

            if type(self.actions) != np.ndarray:
                self.get_logger().info(f"=== Optimizer solved error ===: \n{ref_value}")

        if self.enable_actions_concat:
            if self.pre_actions.dtype == np.float32:
                self.actions = np.concatenate((self.pre_actions[:-self.blending_horizon], self.actions), axis=0)
            else:
                self.actions = np.concatenate(([self.actions[0], self.actions[0]], self.actions), axis=0)
        
            actions_dot = finite_difference_derivative(self.actions, 1, 0.0333, 0)
            actions_ddot = finite_difference_derivative(actions_dot, 1, 0.0333, 0)
            
            actions_full = np.array([self.actions[-self.n_action_step:], actions_dot[-self.n_action_step:], actions_ddot[-self.n_action_step:]])
            
            actions_full = actions_full.transpose((1,0,2))
            
            self.action_queue.extend(actions_full)
            
            flatten_actions = actions_full.flatten()
            
            msg = Float64MultiArray()
            msg.data = flatten_actions.tolist()
            
            msg.layout.dim = [
                                MultiArrayDimension(label="chunk", size=self.n_action_step, stride=3 * self.action_dim),
                                MultiArrayDimension(label="phase", size=3, stride=self.action_dim),
                                MultiArrayDimension(label="joint", size=self.action_dim, stride=1)
                            ]
            
            self.get_logger().info(f"[Inference] 📤 Response ({self.n_action_step},3,{self.action_dim}) array -> {actions_full.shape}")

            response.cur_actions = msg
            return response

        flatten_actions = self.actions.flatten()
            
        msg = Float64MultiArray()
        msg.data = flatten_actions.tolist()
        
        msg.layout.dim = [
                            MultiArrayDimension(label="chunk", size=self.n_action_step, stride=7),
                            MultiArrayDimension(label="joint", size=7, stride=1)
                        ]

        self.get_logger().info(f"[Inference] 📤 Response ({self.n_action_step}, {self.action_dim}) array -> {self.actions.shape}")
        
        response.cur_actions = msg
        return response

    def save_log_with_timestamp(self, name, log_list, save_dir):
        log_array = np.array(log_list, dtype=object)
        np.save(os.path.join(save_dir, f"log_{name}.npy"), log_array)
        
    def save_logs(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        log_dir = os.path.join(self.pkg_root_path, 'logs', f"logs_{timestamp}")

        os.makedirs(log_dir, exist_ok=True)
        
        config_path = os.path.join(log_dir, "experiment_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        for k, v in self.log.items():
            if k != "experiment_config":
                self.save_log_with_timestamp(k, v, log_dir)
                
        self.get_logger().info(f"📁 Save logs as: {log_dir}")
        
    def shutdown_node(self):
        self.save_logs()
        
        if rclpy.ok():
            rclpy.shutdown()
        
        self.destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = InferenceACT()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
        node.shutdown_node()
        
if __name__ == '__main__':
    main()