import os
import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class ActionOptimizer:
    def __init__(self, solver="CLARABEL", chunk_size=100, blending_horizon=10, action_dim=7, len_time_delay=0):
        self.solver = solver
        self.N = chunk_size
        self.B = blending_horizon
        self.D = action_dim
        self.TD = len_time_delay

        self.dt = 0.0333
        self.epsilon_start = 0.0001
        self.epsilon_blending = 0.02
        self.epsilon_goal = 0.002
        self.epsilon_path = 0.003
        
        self.q = cp.Variable((self.N+3, self.D)) # previous + 3 to consider previous vel/acc/jrk
        self.ref = cp.Parameter((self.N+3, self.D),value=np.zeros((self.N+3, self.D))) # previous + 3
        
        D_j = np.zeros((self.N+3, self.N+3))
        for i in range(self.N - 2):
            D_j[i, i]     = -1
            D_j[i, i+1]   = 3
            D_j[i, i+2]   = -3
            D_j[i, i+3]   = 1
        D_j = D_j / self.dt**3

        q_total = self.q + self.ref  # (N, D)
        cost = cp.sum([cp.sum_squares(D_j @ q_total[:, d]) for d in range(self.D)])

        constraints = []

        constraints += [self.q[self.B+3:] <= self.epsilon_path]
        constraints += [self.q[self.B+3:] >= - self.epsilon_path]
        constraints += [self.q[4+self.TD:self.B+3] <= self.epsilon_blending]
        constraints += [self.q[4+self.TD:self.B+3] >= - self.epsilon_blending]
        constraints += [self.q[0:4+self.TD] == 0.0]
        # constraints += [self.q[0:3] <= self.epsilon_start]
        # constraints += [self.q[0:3] >= - self.epsilon_start]

        np.set_printoptions(precision=3, suppress=True, linewidth=100)
        # original = q.value.copy()

        self.p = cp.Problem(cp.Minimize(cost), constraints)

        # 로그와 큐 초기화
        self.log_actions = []     # list of shape-(D,) vectors over time
        self.log_solveds = []
        
        self.clear_logs()

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        # 로그 기록
        self.log_actions.extend(actions.transpose(0, 1))

        blend_len = len_past_actions
        
        # 1) Blending
        self.ref.value[3:] = actions.copy()
        
        # import pdb
        # pdb.set_trace()
        
        if blend_len > 0:
            # update last actions
            self.ref.value[:3+self.TD] = past_actions[-blend_len-3:-blend_len + self.TD].copy()
            
            ratio_space = np.linspace(0, 1, blend_len-self.TD) # (B,1)    
            self.ref.value[3+self.TD:blend_len+3] = ratio_space[:, None] * actions[self.TD:blend_len] + (1 - ratio_space[:, None]) * past_actions[-blend_len+self.TD:]
        
        else: # blend_len == 0
            # update last actions
            self.ref.value[:3] = actions[0]
            
        # 2) QP solve
        t0 = time.time()
        try:
            # p.solve(warm_start=True, verbose=False, solver='OSQP')
            self.p.solve(warm_start=True, verbose=False, solver=self.solver,time_limit=0.1) #, tol_gap_abs=1e-5, tol_feas=1e-5)
        except Exception as e:
            return None, e

        t1 = time.time()
        # print("Time taken for second solve:", t1 - t0)

        self.solved = self.q.value.copy() + self.ref.value.copy()

        # 3) 로그 & 큐에 저장
        self.log_solveds.extend(self.solved.transpose(0, 1).copy())

        return self.solved[3:].copy(), self.ref.value[3:].copy()

    def clear_logs(self):
        self.log_actions.clear()
        self.log_solveds.clear()

    def get_logs(self):
        return {
            'actions': np.array(self.log_actions),   # (T_all, D) → (D, T_all)
            'solved': np.array(self.log_solveds)
        }
        
    def plot_logs(self, show=False, save=False, path=''):
        fig, axes = plt.subplots(4, 2, figsize=(20, 15), num="Actions vs Solved")
        axes = axes.flatten()

        log_actions = np.array(self.log_actions)
        log_solveds = np.array(self.log_solveds)
        
        for i in range(self.D):
            ax = axes[i]
            ax.plot(log_actions[:, i], label='Actions', linestyle='--')
            ax.plot(log_solveds[:, i], label='Solved', linestyle=':')
            ax.set_title(f'Joint {i+1} (Actions vs Solved)')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Action value')
            ax.grid(True)
            ax.legend()

        # 불필요한 subplot 제거
        for j in range(self.D, len(axes)):
            fig.delaxes(axes[j])
            
        fig.suptitle("Actions vs Solved per Joint", fontsize=16)
        fig.subplots_adjust(top=0.92, bottom=0.05, hspace=0.6)
        
        if save:
            path = os.path.join(path, "plot_actions_vs_solved")
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, "actions_vs_solved_plot.png"), dpi=300)
        
        if show:
            plt.show()
        plt.close()

        if save:
            for i in range(self.D):
                fig_single, ax_single = plt.subplots(figsize=(8, 6))
                ax_single.plot(log_actions[:, i], label='Actions', linestyle='--')
                ax_single.plot(log_solveds[:, i], label='Solved', linestyle=':')
                ax_single.set_title(f'Joint {i+1} (Actions vs Solved)')
                ax_single.set_xlabel('Time step')
                ax_single.set_ylabel('Action value')
                ax_single.grid(True)
                ax_single.legend()
                
                single_path = os.path.join(path, f"joint_{i+1}_plot.png")
                fig_single.savefig(single_path, dpi=300)
                plt.close(fig_single)
                print(f"Saved Joint {i+1} plot to {single_path}")
        
    def save_logs(self, path="/home/dognwoo/colcon_ws/src/ros2_lerobot/data"):
        if len(self.log_actions) != 0:
            np.save(os.path.join(path, "log_actions.npy"), self.log_actions)
        if len(self.log_solveds) != 0:
            np.save(os.path.join(path, "log_solveds.npy"), self.log_solveds)

if __name__ == "__main__":
    action_optimizer = ActionOptimizer(solver="CLARABEL", 
                            chunk_size=100, 
                            blending_horizon=20,
                            action_dim=7)
    
    actions = np.load('/home/dognwoo/colcon_ws/src/ros2_lerobot/data/log_inference_actions-temp.npy')
    print(actions.shape)
    solved_traj = action_optimizer.step(actions[100:200, :], actions[:100, :])
    action_optimizer.plot_logs(save=True)
    action_optimizer.save_logs()