import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from ros2_lerobot_interfaces.srv import Inference

import os
import sys
import yaml
import time
import math
import datetime
import numpy as np
from collections import deque

from ros2_lerobot.math_func import cubic_spline, quintic_spline_multi

class InferenceACT(Node):
    def __init__(self):
        super().__init__('inference_act_node')

        # === Init Configs ====
        current_file_path = os.path.abspath(__file__)
        self.pkg_root_path = os.path.join(os.path.dirname(current_file_path), '..')
        config_path = os.path.join(self.pkg_root_path, 'config', "experiment_config.yaml")
        # config_path = os.path.join(self.pkg_root_path, 'config', "towel_folding_config.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.enable_optimizer = self.config["enable_optimizer"]
        self.enable_spline = self.config["enable_spline"]
        self.enable_actions_concat = self.config["enable_actions_concat"]
        self.enable_temporal_ensemble = self.config["enable_temporal_ensemble"]

        self.n_action_step = self.config["n_action_step"]
        self.blending_horizon = self.config["blending_horizon"]

        self.spline = self.config["spline"]
        self.action_dim = self.config["action_dim"]
        self.enable_actions_publish = self.config["enable_actions_publish"]

        # === Execution State Flags ===
        self.start_episod = False
        self.observation_flag = {
            'follower_joint': False,
            'keyboard': False
        }
        
        self.goal_handle = None
        self.move_init_positions = False
        self.event_finish = False
        
        self.flag_response = False
        # self.flag_

        # === ROS Interface ===
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
        
        self.cli = self.create_client(Inference, 'run_inference')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for inference service...')

        self.req = Inference.Request()

        timer_cb_group = None
        self.timer = self.create_timer(0.0025, self.response_callback)
        self.call_timer2 = self.create_timer(0.0025, self.pub_interpolation_action, callback_group=timer_cb_group) # 400 HZ

        self.action_topic = '/arm_controller/follow_joint_trajectory'
        self.action_client = ActionClient(self, FollowJointTrajectory, self.action_topic)
        self.get_logger().info('Waiting for action server...')
        self.action_client.wait_for_server()
        self.get_logger().info('Action server available')        

        # === Utility ===
        self.action_queue = deque([], maxlen=100)
        self.cur_actions = np.full((3, 7), np.nan)
        self.x0 = np.array([], dtype=np.int8)
        self.timestamp = None
        self.last_inf_time = 0
        self.start_spline_time = 0.0   
        self.cnt_timer = 0
        self.index_action = 0
        self.last_chunk_index = 0
        self.current_chunk_index = 0

        self.log = {
                    "spline_actions": [],
                    "publish_action": [],
                    "subscribe_joint_pos": [],
                    "subscribe_joint_vel": [],
                    "current_chunk_index": [],
                    "x1": [],
                    "x_f": [],
                    "x_dot_f": [],
                    "x_ddot_f": []}
        
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
        
    def send_request(self):
        self.get_logger().info('Request sent.')
        msg = Float64MultiArray()
        
        pre_actions = self.cur_actions.flatten().tolist()
        msg.data = pre_actions
        
        msg.layout.dim = [
                            MultiArrayDimension(label="chunk", size=self.n_action_step, stride=3 * self.action_dim),
                            MultiArrayDimension(label="phase", size=3, stride=self.action_dim),
                            MultiArrayDimension(label="joint", size=self.action_dim, stride=1)
                        ]
        
        self.req.pre_actions = msg
        self.future = self.cli.call_async(self.req)
        self.flag_response = True

    def response_callback(self, update_current = False):
        if self.flag_response:
            try:
                if self.future.done():
                    response = self.future.result()
                    self.new_actions = np.array(response.cur_actions.data, dtype=np.float64)
                    
                    try:
                        self.new_actions = self.new_actions.reshape((self.n_action_step, 3, self.action_dim))
                    except:
                        if self.enable_temporal_ensemble:
                            self.new_actions = self.new_actions.reshape((1, 14))
                        else:
                            self.new_actions = self.new_actions.reshape((self.n_action_step, self.action_dim))
                        
                    self.get_logger().info(f'Received result: {self.new_actions.shape}')
                    if update_current:
                        self.cur_actions = self.new_actions.copy()
                    self.flag_response = False
            except Exception as e:
                self.get_logger().error(f'Service call failed: {e}')
        
    def follower_joint_callback(self, msg):
        try:
            if set(self.joint_names).issubset(set(msg.name)):
                self.follower_joint_positions = [msg.position[msg.name.index(j)] for j in self.joint_names]
                self.follower_joint_velocities = [msg.velocity[msg.name.index(j)] for j in self.joint_names]
                
                self.observation_flag['follower_joint'] = True
                
                # self.get_logger().info(f'🦿 Received leader joint positions: {self.follower_joint_positions}')
                # self.get_logger().info(f'🦿 Received leader joint velocitys: {self.follower_joint_velocities}')
                
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
                time.sleep(1)
                self.send_request()
                self.response_callback(update_current=True)
                
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

    def pub_interpolation_action(self):
        if self.event_finish:
            self.goal_handle = None
            self.shutdown_node()
            return
        
        for item in self.observation_flag.items():
            if item is False: return
        
        if not self.start_episod:
            return
        
        # for the first time
        if np.isnan(self.cur_actions).any():
            self.cur_actions = self.new_actions.copy()
        
        self.cnt_timer += 1
        if  30 * self.cnt_timer >  400 * self.index_action:
            self.index_action += 1
            
            if self.enable_temporal_ensemble:
                self.send_request()
            
            else:
                # STEP
                self.current_chunk_index = self.index_action - self.last_chunk_index
                
                if self.current_chunk_index == self.n_action_step - self.blending_horizon:
                    self.send_request()
                    
                elif self.current_chunk_index > self.n_action_step - self.blending_horizon:
                    # check 
                    if not self.flag_response:
                        self.last_chunk_index += self.n_action_step - self.blending_horizon
                        self.cur_actions = self.new_actions.copy()
                        
                        self.current_chunk_index = self.index_action - self.last_chunk_index
                        
                x_new = self.cur_actions[min(self.current_chunk_index, 
                                            self.n_action_step-1)].copy()

                if self.x0.dtype == np.float64:
                    self.x0 = self.x1.copy()
                else:
                    self.x0 = x_new.copy()
                self.x1 = x_new.copy()

        self.log["current_chunk_index"].append((self.cnt_timer / 400.0, self.current_chunk_index))
        self.log["x1"].append((self.cnt_timer / 400.0, self.x1[0]))
        
        if not self.enable_temporal_ensemble:
            if self.enable_spline:
                t_0 = 0.0
                t_f = 0.0333
                self.timestamp = self.cnt_timer / 400.0 - (self.index_action-1) / 30.0
                t = min(self.timestamp, 0.0333)
                self.last_inf_time = time.time()
                
                if self.enable_actions_concat:
                    x_0 = self.x0[0]
                    x_dot_0 = self.x0[1]
                    x_ddot_0 = self.x0[2]
                    x_f = self.x1[0]
                    x_dot_f = self.x1[1]
                    x_ddot_f = self.x1[2]
                    
                    self.log["x_f"].append((self.cnt_timer / 400.0, x_f))
                    self.log["x_dot_f"].append((self.cnt_timer / 400.0, x_dot_f))
                    self.log["x_ddot_f"].append((self.cnt_timer / 400.0, x_ddot_f))
                    
                else:
                    x_0 = self.x0
                    x_f = self.x1
                
                if self.spline == 'cubic':
                    spline_action = cubic_spline(t, t_0, t_f, x_0, x_f, x_dot_0, x_dot_f)
                
                elif self.spline == 'quintic':
                    spline_action = quintic_spline_multi(t, t_0, t_f, x_0, x_dot_0, x_ddot_0, x_f, x_dot_f, x_ddot_f)
                    spline_action = spline_action[0]
                
                elif self.spline == 'linear':
                    ratio = t * 30
                    ratio = max(0.0, min(1.0, ratio))
                    spline_action = (1 - ratio) * x_0 + ratio * x_f 
            else:
                spline_action = self.x1
            
            if self.enable_spline:
                self.log["spline_actions"].append((self.cnt_timer / 400.0, spline_action))
        
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        
        # self.get_logger().info(f"new action: {self.new_actions[0, :self.action_dim]}")
        if self.enable_temporal_ensemble:
            point.positions = self.new_actions[0, :self.action_dim]
        else:
            point.positions = spline_action
        point.time_from_start.sec = 0
        
        msg.points.append(point)
        if self.enable_actions_publish:
            self.joint_trajectory_publisher_.publish(msg)
        
        try:
            self.log["publish_action"].append((self.cnt_timer / 400.0, point.positions))
            self.log["subscribe_joint_pos"].append((self.cnt_timer / 400.0, self.follower_joint_positions))
            self.log["subscribe_joint_vel"].append((self.cnt_timer / 400.0, self.follower_joint_velocities))
        except:
            self.log["publish_action"].append(point.positions)
            self.log["subscribe_joint_pos"].append(self.follower_joint_positions)
            self.log["subscribe_joint_vel"].append(self.follower_joint_velocities)
            
        error_joint = np.linalg.norm(self.pre_follower_joint_positions - np.array(self.follower_joint_positions))
        
        if error_joint > 0.2:
            self.get_logger().info(f"current pos: {self.follower_joint_positions}")
            self.get_logger().info(f"error: {error_joint}")
            self.get_logger().info("🛑 Joint Error is too high!")
            self.event_finish = True
        
        self.pre_follower_joint_positions = self.follower_joint_positions
                    
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
            
            self._send_goal_future = self.action_client.send_goal_async(
                goal_msg, feedback_callback=self.feedback_callback
            )
            self._send_goal_future.add_done_callback(
                self.goal_response_callback
            )
        else:        
            if self.goal_handle:
                self.goal_handle.cancel_goal_async()
            
            if rclpy.ok():
                rclpy.shutdown()
            
            self.destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = InferenceACT()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.goal_handle = None
        node.get_logger().info('🟢 Keyboard interrupt, shutting down.\n')
        node.shutdown_node()
        
if __name__ == '__main__':
    main()