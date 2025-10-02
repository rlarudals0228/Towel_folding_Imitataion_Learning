import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

import torchvision
import torch
import os
import sys
import cv2
import yaml
import time
import math
import datetime
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ros2_lerobot.action_optimizer import ActionOptimizer
from ros2_lerobot.math_func import cubic_spline, quintic_spline_multi, finite_difference_derivative

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

class InferenceACT(Node):
    def __init__(self):
        super().__init__('inference_act_node')

        # === Execution State Flags ===
        self.start_episod = False
        self.observation_flag = {
            'top_view_img': True,
            'wrist_img': False,
            'follower_joint': False,
            'keyboard': False
        }
        
        self.goal_handle = None
        self.flag_inference = False
        self.move_init_positions = False
        self.event_finish = False
        
        self.enable_optimizer = True
        self.enable_actions_concat = True
        self.enable_spline = True
        self.enable_temporal_ensemble = False


        # === Path Configurations ===
        ros2_lerobot_pkg_path = get_package_share_directory('ros2_lerobot')
        pretrained_folder_name = os.path.join(
            ros2_lerobot_pkg_path, 
            'ckpt', 
            # 'throwing_with_mirror_flip-act_chunk-size-100_epoch-200000',
            'throwing_ball-act_chunk-size-100_epoch-200000',
            # 'step-10000'
            'step-30000'
        )
        dataset_path = os.path.join(
            ros2_lerobot_pkg_path,
            'demo_data',
            'throwing_ball'
            # 'throwing_with_mirror_flip'
        )

        # === Utility ===
        self.bridge = CvBridge()
        self.device = torch.device("cuda:0")
        self.chunk_size = 100
        self.n_action_step = 50
        self.action_queue = deque([], maxlen=self.n_action_step)
        self.pre_actions = np.array([], dtype=np.int8)
        self.x0 = np.array([], dtype=np.int8)
        self.timestamp = None
        
        self.start_inference_time = 0.0
        self.start_spline_time = 0.0
        self.start_time = time.time()

        self.log = {"experiment_config": None,
                    "inference_actions": [],
                    "ref.value": [],
                    "solved": [],
                    "actions_full": [],
                    "spline_actions": [],
                    "publish_action": [],
                    "subscribe_joint": []}

        # === Policy Configuration & Loading ===
        dataset_metadata = LeRobotDatasetMetadata("throwing_ball", root=dataset_path)
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
        self.keyboard_sub = self.create_subscription(
            String,
            '/keyboard_command',
            self.keyboard_callback,
            10
        )
        self.joint_trajectory_publisher_ = self.create_publisher(JointTrajectory, '/leader/joint_trajectory', 10)

        timer_cb_group = None
        self.call_timer1 = self.create_timer(0.033, self.inference, callback_group=timer_cb_group) # 30 HZ
        self.call_timer2 = self.create_timer(0.0025, self.pub_interpolation_action, callback_group=timer_cb_group) # 400 HZ

        self.action_topic = '/arm_controller/follow_joint_trajectory'
        self.action_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
        self.get_logger().info('Waiting for action server...')
        self.action_client.wait_for_server()
        self.get_logger().info('Action server available')        

        # === Joint Positions ===
        self.init_positions = [
            0.0,
            -math.pi / 2,
            self.angle_to_radian(152),
            self.angle_to_radian(-62),
            math.pi / 2,
            0.0,
            0.0
        ]
        self.target_positions = [
            -0.00152199656124115, 
            -0.046043392033767706, 
            1.554066347051239, 
            -0.05522330830078125, 
            1.481010512111664, 
            0.0122718462890625, 
            0.0
        ]
        self.pre_follower_joint_positions = self.target_positions.copy()
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'rh_r1_joint']

        # === Trajectory Parameters ===
        self.duration = 5.0
        self.epsilon = 0.01
        self.num_points = 100

        # === Action Optimizer Parameters ===
        # self.spline = 'cubic'
        self.spline = 'quintic'
        
        self.blending_horizon = 10
        self.solver = "CLARABEL"
        self.action_optimizer = ActionOptimizer(
            solver=self.solver,
            chunk_size=self.n_action_step,
            blending_horizon=self.blending_horizon,
            action_dim=7
        )
        
        self.log["experiment_config"] = {'enable_optimizer': self.enable_optimizer,
                                         'enable_actions_concat': self.enable_actions_concat,
                                         'enable_spline': self.enable_spline,
                                         'enable_temporal_ensemble': self.enable_temporal_ensemble,
                                         'temporal_ensemble_coeff': self.temporal_ensemble_coeff,
                                         'chuck_size': self.chunk_size,
                                         'n_action_step': self.n_action_step,
                                         'blending_horizon': self.blending_horizon,
                                         'spline': self.spline,
                                         'optimizer_solver': self.solver,
                                         'dataset': dataset_path,
                                         'policy': pretrained_folder_name
                                         }
        
        self.get_logger().info("🧪 Experiment Configuration:")
        for k, v in self.log["experiment_config"].items():
            self.get_logger().info(f"  - {k}: {v}")
    
    def top_view_image_callback(self, msg):
        try:
            self.top_view_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # self.get_logger().info(f'top view image test: {self.top_view_img.shape}')
            self.observation_flag['top_view_img'] = True
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
    
            self.observation_flag['top_view_img'] = False

    def wrist_image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.wrist_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv2.imwrite('/home/dognwoo/colcon_ws/src/ros2_lerobot/data/wrist_img.png', self.wrist_img)
            
            # self.get_logger().info(f'wrist image test: {self.wrist_img.shape}')
            self.observation_flag['wrist_img'] = True
            
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            
            self.observation_flag['wrist_img'] = False
                
    def follower_joint_callback(self, msg):
        try:
            if set(self.joint_names).issubset(set(msg.name)):
                self.follower_joint_positions = [msg.position[msg.name.index(j)] for j in self.joint_names]
                self.follower_joint_velocitys = [msg.velocity[msg.name.index(j)] for j in self.joint_names]
                
                self.observation_flag['follower_joint'] = True
                
                # self.get_logger().info(f'🦿 Received leader joint positions: {self.follower_joint_positions}')
                # self.get_logger().info(f'🦿 Received leader joint velocitys: {self.follower_joint_velocitys}')
                
            if self.goal_handle is None and not self.event_finish:
                goal_msg = FollowJointTrajectory.Goal()
                goal_msg.trajectory = self.create_smooth_trajectory(
                    self.follower_joint_positions, self.target_positions
                )
                
                goal_msg.path_tolerance = []
                goal_msg.goal_tolerance = []
                goal_msg.goal_time_tolerance.sec = 0
                goal_msg.goal_time_tolerance.nanosec = 0
                
                self.get_logger().info('Sending goal...')
                    
                self._send_goal_future = self.action_client.send_goal_async(
                    goal_msg, feedback_callback=self.feedback_callback
                )
                self._send_goal_future.add_done_callback(
                    self.goal_response_callback
                )
            
            if self.check_step_completion() and not self.move_init_positions:
                self.get_logger().info('Succeses to move initial positions!')
                self.move_init_positions = True
                
        except Exception as e:
            self.get_logger().error(f"Failed to sub follower joint: {e}")
            
            self.observation_flag['follower_joint'] = False

    def keyboard_callback(self, msg):
        try:            
            if msg.data == 'start':
                self.start_episod = True
                self.get_logger().info(f"✅ Start. Inference")
            else:
                self.start_episod = False
            
            self.observation_flag['keyboard'] = True
        except Exception as e:
            self.get_logger().error(f"Failed to sub keyboard: {e}")
    
            self.observation_flag['keyboard'] = False
            
    def inference(self):
        if self.event_finish:
            return
        
        if self.start_episod:
            for item in self.observation_flag.items():
                if not item: return
            
            # if len(self.action_queue) == 0:
            if len(self.action_queue) < self.action_optimizer.B:
                img_transform = torchvision.transforms.ToTensor()

                # top_view_img = cv2.resize(self.top_view_img, (256, 256))
                # top_view_img = np.zeros((256, 256, 3), np.uint8)
                wrist_img = cv2.resize(self.wrist_img, (256, 256))
                
                region = self.wrist_img[:, 470:]
                region = cv2.flip(region, 1)
                flip_img = self.wrist_img.copy()
                flip_img[:, 470:] = region
                flip_img = cv2.resize(flip_img, (256, 256))
                
                # top_view_img = img_transform(top_view_img)
                wrist_img = img_transform(wrist_img)
                flip_img = img_transform(flip_img)
                
                follower_positions_np = np.array(self.follower_joint_positions, dtype=np.float32)
                follower_velocitys_np = np.array(self.follower_joint_velocitys, dtype=np.float32)
                state = np.concatenate((follower_positions_np, follower_velocitys_np), axis=0)
                
                batch = {
                    'observation.state': torch.tensor([state]).to(self.device),
                    # 'observation.image': top_view_img.unsqueeze(0).to(self.device),
                    'observation.wrist_image': wrist_img.unsqueeze(0).to(self.device),
                    'observation.flip_image': flip_img.unsqueeze(0).to(self.device),
                }
                
                if not self.enable_optimizer and \
                   not self.enable_actions_concat and \
                   not self.enable_spline and \
                   not self.enable_temporal_ensemble:
                    self.raw_action = self.policy.select_action(batch).detach().cpu().numpy()[:, :7]
                    self.log["inference_actions"].append(self.raw_action)
                    self.action_queue.extend(self.raw_action)
                elif self.enable_temporal_ensemble:
                    self.raw_action = self.policy.select_action(batch).detach().cpu().numpy()[:, :7]
                    self.log["inference_actions"].append(self.raw_action)
                    self.action_queue.extend(self.raw_action)
                else:
                    self.raw_actions = self.policy.get_action_chuck(batch, self.n_action_step).detach().cpu().numpy()[:, :7] # float32
                    
                    self.log["inference_actions"].extend(self.raw_actions[:-self.blending_horizon])
                    
                    past_actions = self.pre_actions
                    # past_actions = np.array(self.action_queue)
                    len_past_actions = len(self.action_queue)
                    if self.enable_optimizer:
                        if len(past_actions.shape) > 2:
                            past_actions = past_actions[:,0,:] #pos only
                        self.actions, ref_value = self.action_optimizer.solve(self.raw_actions, past_actions, len_past_actions) # (100, 7), float64
                        
                        self.log["solved"].extend(self.actions[:-self.blending_horizon])
                        self.log["ref.value"].extend(ref_value[:-self.blending_horizon])

                        if type(self.actions) != np.ndarray:
                            self.get_logger().info(f"=== Optimizer solved error ===: \n{self.actions}")
                        
                        self.action_queue.clear()
                    
                    if self.enable_spline:
                        if self.enable_actions_concat:
                            if self.pre_actions.dtype == np.float32:
                                self.actions = np.concatenate((self.pre_actions[:-len_past_actions], self.actions), axis=0)
                            else:
                                self.actions = np.concatenate(([self.actions[0], self.actions[0]], self.actions), axis=0)
                        
                        actions_dot = finite_difference_derivative(self.actions, 1, 0.0333, 0)
                        actions_ddot = finite_difference_derivative(actions_dot, 1, 0.0333, 0)
                        
                        if self.enable_actions_concat:
                            actions_full = np.array([self.actions[-self.n_action_step:], actions_dot[-self.n_action_step:], actions_ddot[-self.n_action_step:]])
                        else:
                            actions_full = np.array([self.actions, actions_dot, actions_ddot])
                        
                        actions_full = actions_full.transpose((1,0,2)) # (100, 3, 7), float64
                        
                        self.log["actions_full"].extend(actions_full[:-self.blending_horizon])
                        
                        self.action_queue.extend(actions_full)
                    else:
                        self.action_queue.extend(self.actions)
                    
                    if self.enable_actions_concat:
                        self.pre_actions = self.actions[-self.n_action_step:]
                        self.actions = self.actions[-self.n_action_step:]
                    else:
                        self.pre_actions = self.actions
                    
                self.flag_inference = True

            if self.flag_inference:

                new_x1 = self.action_queue.popleft()
                # self.log["solved"].append(new_x1)
                
                if self.x0.dtype == np.float64:
                    self.x0 = self.x1.copy()
                else:
                    self.x0 = new_x1.copy()
                self.x1 = new_x1.copy()
                self.last_inf_time = time.time()
            
    def pub_interpolation_action(self):
        if self.event_finish:
            self.get_logger().info('Event finish')
            self.goal_handle = None
            self.shutdown_node()
            return
        
        for item in self.observation_flag.items():
            if item is False: return
        
        if not self.start_episod:
            return
        
        if not self.flag_inference:
            return
        
        if self.enable_spline and not self.enable_temporal_ensemble:
            t_0 = 0.0
            t_f = 0.0333
            self.timestamp = time.time()
            t = min(self.timestamp - self.last_inf_time, 0.0333)
            x_0 = self.x0[0]
            x_dot_0 = self.x0[1]
            x_ddot_0 = self.x0[2]
            x_f = self.x1[0]
            x_dot_f = self.x1[1]
            x_ddot_f = self.x1[2]
            
            if self.spline == 'cubic':
                spline_action = cubic_spline(t, t_0, t_f, x_0, x_f, x_dot_0, x_dot_f) # (7, )
            
            if self.spline == 'quintic':
                spline_action = quintic_spline_multi(t, t_0, t_f, x_0, x_dot_0, x_ddot_0, x_f, x_dot_f, x_ddot_f) # (3, 7)
                spline_action = spline_action[0]
        else:
            spline_action = self.x1
        
        if self.enable_spline:
            self.log["spline_actions"].append((self.timestamp, spline_action))
            
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        
        if self.enable_temporal_ensemble:
            point.positions = self.x0
        else:
            point.positions = spline_action
        point.time_from_start.sec = 0
        
        msg.points.append(point)
        self.joint_trajectory_publisher_.publish(msg)
        
        try:
            self.log["publish_action"].append((self.timestamp, point.positions))
            self.log["subscribe_joint"].append((self.timestamp, self.follower_joint_positions))
        except:
            self.log["publish_action"].append(point.positions)
            self.log["subscribe_joint"].append(self.follower_joint_positions)
            
        # self.get_logger().info(f'pub actions: {spline_action}')
        
        error_joint = np.linalg.norm(self.pre_follower_joint_positions - np.array(self.follower_joint_positions))
        # self.get_logger().info(f"error_joint: {error_joint}")
        
        if error_joint > 0.3:
            # self.get_logger().info(f"target pos: {action[:7]}")
            self.get_logger().info(f"current pos: {self.follower_joint_positions}")
            self.get_logger().info(f"error: {error_joint}")
            self.get_logger().info("🛑 Joint Error is too high!")
            self.event_finish = True
        
        self.pre_follower_joint_positions = self.follower_joint_positions
        
        # self.get_logger().info(f"spline elapsed time: {time.time() - self.start_spline_time}")
        # self.start_spline_time = time.time()
                    
    def angle_to_radian(self, angle):
        return angle * math.pi / 180   
    
    def radian_to_angle(self, radian):
        return radian * 180 / math.pi
    
    def create_smooth_trajectory(self, start_pos, end_pos):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        times = np.linspace(0, self.duration, self.num_points)

        for i in range(self.num_points):
            point = JointTrajectoryPoint()
            t = times[i]

            # Quintic polynomial coefficients
            t_norm = t / self.duration
            t_norm2 = t_norm * t_norm
            t_norm3 = t_norm2 * t_norm
            t_norm4 = t_norm3 * t_norm
            t_norm5 = t_norm4 * t_norm

            # Quintic polynomial coefficients for position
            pos_coeff = 10 * t_norm3 - 15 * t_norm4 + 6 * t_norm5

            # Velocity coefficients (derivative of position)
            vel_coeff = (30 * t_norm2 - 60 * t_norm3 + 30 * t_norm4) / self.duration

            # Acceleration coefficients (derivative of velocity)
            acc_coeff = (60 * t_norm - 180 * t_norm2 + 120 * t_norm3) / (
                self.duration * self.duration
            )

            positions = []
            velocities = []
            accelerations = []

            for j in range(len(self.joint_names)):
                pos = start_pos[j] + (end_pos[j] - start_pos[j]) * pos_coeff
                vel = (end_pos[j] - start_pos[j]) * vel_coeff
                acc = (end_pos[j] - start_pos[j]) * acc_coeff

                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)

            point.positions = positions
            point.velocities = velocities
            point.accelerations = accelerations
            point.time_from_start.sec = int(times[i])
            point.time_from_start.nanosec = int((times[i] % 1) * 1e9)

            traj.points.append(point)

        return traj

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'Feedback: {feedback.actual.positions}')
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self.goal_handle = goal_handle
    
    def check_step_completion(self):
        return all(
            abs(curr - target) < self.epsilon
            for curr, target in zip(self.follower_joint_positions, self.target_positions)
        )
    
    def save_log_with_timestamp(self, name, log_list, save_dir):
        log_array = np.array(log_list, dtype=object)  # object type: 각 row = (timestamp, vector)
        np.save(os.path.join(save_dir, f"log_{name}.npy"), log_array)
        
    def save_logs(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        current_file_path = os.path.abspath(__file__)
        pkg_root_path = os.path.join(os.path.dirname(current_file_path), '..')
        log_dir = os.path.join(pkg_root_path, 'logs', f"logs_{timestamp}")

        os.makedirs(log_dir, exist_ok=True)
        
        config_path = os.path.join(log_dir, "experiment_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.log["experiment_config"], f, default_flow_style=False, sort_keys=False)
        
        for k, v in self.log.items():
            if k != "experiment_config":
                self.save_log_with_timestamp(k, v, log_dir)
                
        self.get_logger().info(f"📁 Save logs as: {log_dir}")
        
    def shutdown_node(self):
        self.save_logs()
        
        self.event_finish = True
        
        init_positions = [0.0, -1.5708, 2.6529, -1.0821, 1.5708, 0.0, 0.0]
        
        if self.goal_handle is None:
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory = self.create_smooth_trajectory(
                self.follower_joint_positions, init_positions
            )
            
            goal_msg.path_tolerance = []
            goal_msg.goal_tolerance = []
            goal_msg.goal_time_tolerance.sec = 0
            goal_msg.goal_time_tolerance.nanosec = 0
            
            self.get_logger().info('🟢 Sending goal for shutdown...')
            self._send_goal_future = self.action_client.send_goal_async(
                goal_msg, feedback_callback=self.feedback_callback
            )
            self._send_goal_future.add_done_callback(
                self.goal_response_callback
            )
        else:        
            if self.goal_handle:
                self.goal_handle.cancel_goal_async()
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(0)
    
def main(args=None):
    rclpy.init(args=args)
    node = InferenceACT()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.goal_handle = None
        node.shutdown_node()
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
        
if __name__ == '__main__':
    main()