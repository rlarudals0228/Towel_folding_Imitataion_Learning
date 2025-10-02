import os
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math_func import finite_difference_derivative

def get_latest_log_folder(logs_root_dir):
    subdirs = [d for d in os.listdir(logs_root_dir)
               if os.path.isdir(os.path.join(logs_root_dir, d)) and d.startswith("logs_")]
    if not subdirs:
        raise FileNotFoundError("No log folders found in the logs directory.")
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(logs_root_dir, d)))
    return latest_subdir

def load_all_logs(log_dir):
    logs = {}
    for filename in os.listdir(log_dir):
        if filename.startswith("log_") and filename.endswith(".npy"):
            log_name = filename.replace("log_", "").replace(".npy", "")
            log_path = os.path.join(log_dir, filename)
            log_data = np.load(log_path, allow_pickle=True)
            logs[log_name] = log_data
    return logs

def plot_pos_vel_acc(config, all_logs, logs_path, idx_start=0, idx_end=None, joint_index=1):
    
    chunk = config["n_action_step"]
    blend = config["blending_horizon"]
    time_delay = config["len_delay_time"]
    dt = 0.0333
    
    # Times New Roman 폰트 설정
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # 또는 ['Times'] for compatibility
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'pdf.fonttype': 42,  # To embed fonts properly in PDF
        'ps.fonttype': 42
    })
    
    cmap = plt.get_cmap('tab10')
    
    timestamps_target_pos = np.array([x[0] for x in all_logs["publish_action"]])
    timestamps_target_pos -= timestamps_target_pos[0]  # 정규화
    target_pos = np.array([x[1] for x in all_logs["publish_action"]])
    t0 = timestamps_target_pos[idx_start]  # 기준 시간 저장
    
    timestamps_current_pos = np.array([x[0] for x in all_logs["subscribe_joint_pos"]])
    timestamps_current_pos -= timestamps_current_pos[0]  # 정규화
    current_pos = np.array([x[1] for x in all_logs["subscribe_joint_pos"]])
    
    timestamps_current_vel = np.array([x[0] for x in all_logs["subscribe_joint_vel"]])
    timestamps_current_vel -= timestamps_current_vel[0]  # 정규화
    current_vel = np.array([x[1] for x in all_logs["subscribe_joint_vel"]])
    
    timestamps_vel = np.array([x[0] for x in all_logs["x_dot_f"]])
    timestamps_vel -= timestamps_vel[0]  # 정규화
    # vel = np.array([x[1] for x in all_logs["x_dot_f"]])
    
    # timestamps_acc = np.array([x[0] for x in all_logs["x_ddot_f"]])
    # timestamps_acc -= timestamps_acc[0]  # 정규화
    # acc = np.array([x[1] for x in all_logs["x_ddot_f"]])
    
    # vel = finite_difference_derivative(current_pos, 1, 0.0025, 0)
    target_vel = finite_difference_derivative(target_pos, 1, 0.0025, 0)
    target_acc = finite_difference_derivative(target_vel, 1, 0.0025, 0)
    current_acc = finite_difference_derivative(current_vel, 1, 0.0025, 0)

    if idx_end is None:
        idx_end = len(all_logs["subscribe_joint_pos"])

    # === Plot using subplots ===
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Position subplot
    axes[0].plot(timestamps_target_pos[idx_start:idx_end] - t0, target_pos[idx_start:idx_end, joint_index-1], color=cmap(0), label='Target Position', alpha=0.5)
    axes[0].plot(timestamps_current_pos[idx_start:idx_end] - t0, current_pos[idx_start:idx_end, joint_index-1], color=cmap(3), label='Current Position', alpha=0.5)
    axes[0].set_ylabel("Position [rad]")
    # axes[0].set_title(f"Joint {joint_index} - Position")
    # axes[0].grid(True)
    
    handles = [
        plt.Line2D([], [], color=cmap(0), label='Target Position', alpha=0.5),
        plt.Line2D([], [], color=cmap(3), label='Current Position', alpha=0.5),
        # plt.Line2D([], [], color='gray', alpha=0.3, linewidth=10, label='Inference Delay'),
        # plt.Line2D([], [], color='yellow', alpha=0.3, linewidth=10, label='Blending Zone')
    ]
    axes[0].legend(handles=handles, loc='upper left')
    

    # Velocity subplot
    axes[1].plot(timestamps_current_pos[idx_start:idx_end] - t0, target_vel[idx_start:idx_end, joint_index-1], color=cmap(0), label='Target Velocity', alpha=0.5)
    axes[1].plot(timestamps_current_pos[idx_start:idx_end] - t0, current_vel[idx_start:idx_end, joint_index-1], color=cmap(3), label='Current Velocity', alpha=0.5)
    axes[1].set_ylabel("Velocity [rad/s]")
    # axes[1].set_title(f"Joint {joint_index} - Velocity")
    axes[1].legend(loc='upper left')
    # axes[1].grid(True)

    handles = [
        plt.Line2D([], [], color=cmap(0), label='Target Velocity', alpha=0.5),
        plt.Line2D([], [], color=cmap(3), label='Current  Velocity', alpha=0.5),
        # plt.Line2D([], [], color='gray', alpha=0.3, linewidth=10, label='Inference Delay'),
        # plt.Line2D([], [], color='yellow', alpha=0.3, linewidth=10, label='Blending Zone')
    ]
    axes[1].legend(handles=handles, loc='upper left')

    # Acceleration subplot
    axes[2].plot(timestamps_target_pos[idx_start:idx_end] - t0, target_acc[idx_start:idx_end, joint_index-1], color=cmap(0), label='Target Acceleration', alpha=0.5)
    axes[2].plot(timestamps_current_pos[idx_start:idx_end] - t0, current_acc[idx_start:idx_end, joint_index-1], color=cmap(3), label='Current Acceleration', alpha=0.5)
    axes[2].set_ylabel("Acceleration [rad/s²]")
    axes[2].set_xlabel("Time [s]")
    # axes[2].set_title(f"Joint {joint_index} - Acceleration")
    axes[2].legend(loc='upper left')
    # axes[2].grid(True)
    
    handles = [
        plt.Line2D([], [], color=cmap(0), label='Target Acceleration', alpha=0.5),
        plt.Line2D([], [], color=cmap(3), label='Current Acceleration', alpha=0.5),
        # plt.Line2D([], [], color='gray', alpha=0.3, linewidth=10, label='Inference Delay'),
        # plt.Line2D([], [], color='yellow', alpha=0.3, linewidth=10, label='Blending Zone')
    ]
    axes[2].legend(handles=handles, loc='upper left')

    # bias_time = 4.5 * (chunk-blend) * dt
    
    # visible_time = timestamps_target_pos[idx_start:idx_end]  # 실제 plotting 시간 범위
    # t_end = visible_time[-1]

    # k = 0
    # while True:
    #     # start_time = k * (chunk - blend) * dt - bias_time
    #     start_time = k * (chunk - blend) * dt + idx_start * 0.0025
    #     if start_time > t_end:
    #         break

    #     delay_time = time_delay * dt
    #     blending_time = blend * dt

    #     for ax in axes:
    #         ax.axvspan(start_time, start_time + delay_time, color='gray', alpha=0.2)
    #         ax.axvspan(start_time + delay_time, start_time + blending_time, color='yellow', alpha=0.2)
    #         ax.axvline(x=start_time, color='gray', linestyle='--', linewidth=0.5)

    #     k += 1

    # 전체 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(os.path.join(logs_path, 'pos_vel_acc_subplot.pdf'), dpi=300)
    plt.savefig(os.path.join(logs_path, 'pos_vel_acc_subplot.png'), dpi=300)
    plt.show()
    plt.close(fig)

def two_subplot(all_logs, logs_path, idx_start=0, idx_end=None, joint_index=1):
    # Times New Roman 폰트 설정
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # 또는 ['Times'] for compatibility
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'pdf.fonttype': 42,  # To embed fonts properly in PDF
        'ps.fonttype': 42
    })
    
    cmap = plt.get_cmap('tab10')
    
    timestamps_target_pos = np.array([x[0] for x in all_logs["publish_action"]])
    timestamps_target_pos -= timestamps_target_pos[0]  # 정규화
    target_pos = np.array([x[1] for x in all_logs["publish_action"]])
    t0 = timestamps_target_pos[idx_start]  # 기준 시간 저장
    
    timestamps_current_pos = np.array([x[0] for x in all_logs["subscribe_joint_pos"]])
    timestamps_current_pos -= timestamps_current_pos[0]  # 정규화
    current_pos = np.array([x[1] for x in all_logs["subscribe_joint_pos"]])
    
    timestamps_current_vel = np.array([x[0] for x in all_logs["subscribe_joint_vel"]])
    timestamps_current_vel -= timestamps_current_vel[0]  # 정규화
    current_vel = np.array([x[1] for x in all_logs["subscribe_joint_vel"]])
    
    timestamps_vel = np.array([x[0] for x in all_logs["x_dot_f"]])
    timestamps_vel -= timestamps_vel[0]  # 정규화
    
    target_vel = finite_difference_derivative(target_pos, 1, 0.0025, 0)
    target_acc = finite_difference_derivative(target_vel, 1, 0.0025, 0)
    current_acc = finite_difference_derivative(current_vel, 1, 0.0025, 0)

    if idx_end is None:
        idx_end = len(all_logs["subscribe_joint_pos"])
    
    # === Plot using subplots ===
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    # --- 1. Position subplot ---
    axes[0].plot(timestamps_current_pos[idx_start:idx_end] - t0, current_pos[idx_start:idx_end, joint_index-1], color=cmap(3), label='Position')
    axes[0].set_ylabel("Position (rad)")
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0.4, 2.0)
    axes[0].grid(True)  # 🔹 격자 추가

    # --- 2. Velocity + Acceleration subplot ---
    ax_vel = axes[1]
    ax_acc = ax_vel.twinx()  # 오른쪽 y축 생성

    # 속도 (왼쪽 y축, 파란색 계열)
    ax_vel.plot(timestamps_current_pos[idx_start:idx_end] - t0, target_vel[idx_start:idx_end, joint_index-1], color=cmap(0), label='Velocity', alpha=0.5)
    ax_vel.set_ylabel("Velocity (rad/s)", color='black')
    ax_vel.tick_params(axis='y', labelcolor='black')
    ax_vel.set_ylim(-5.0, 1.5)
    ax_vel.grid(True)  # 🔹 왼쪽 y축 기준 격자 추가

    # 가속도 (오른쪽 y축, 주황색 계열)
    ax_acc.plot(timestamps_target_pos[idx_start:idx_end] - t0, target_acc[idx_start:idx_end, joint_index-1], color=cmap(3), label='Acceleration', alpha=0.5)
    ax_acc.set_ylabel("Acceleration (rad/s²)", color='black')
    ax_acc.tick_params(axis='y', labelcolor='black')
    ax_acc.set_ylim(-100, 100)

    # x축 라벨
    axes[1].set_xlabel("Time (s)")

    # 범례 통합
    lines_1, labels_1 = ax_vel.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # 전체 레이아웃 정리 및 저장
    plt.tight_layout()
    plt.savefig(os.path.join(logs_path, 'pos_velacc_subplot.pdf'), dpi=300)
    plt.savefig(os.path.join(logs_path, 'pos_velacc_subplot.png'), dpi=300)
    plt.show()
    plt.close(fig)




if __name__ == "__main__":
    # logs_dir_name = None
    logs_dir_name = 'logs_2025-08-17_23-28-21'
    
    current_file_path = os.path.abspath(__file__)
    logs_root_path = os.path.join(os.path.dirname(current_file_path), '..', 'logs')
        
    if logs_dir_name:
        logs_path = os.path.join(logs_root_path, logs_dir_name)
    else:
        logs_path = get_latest_log_folder(logs_root_path)

    logs_path = os.path.join(logs_root_path, logs_path)
    config_path = os.path.join(logs_path, "experiment_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"folder names:", logs_path)
    all_logs = load_all_logs(logs_path)
    
    # # logs_2025-05-29_23-42-05 Enable Optimizer, Quintic
    # plot_pos_vel_acc(config, all_logs, logs_path, idx_start=4*400, idx_end=14*400, joint_index=2)
    
    # # logs_2025-05-29_23-43-50 Enable Optimizer, Linear
    # plot_pos_vel_acc(config, all_logs, logs_path, idx_start=10*400, idx_end=20*400, joint_index=2)
    
    # # logs_2025-05-29_23-50-44 Unenable Optimizer, Linear
    # plot_pos_vel_acc(config, all_logs, logs_path, idx_start=28*400, idx_end=38*400, joint_index=2)
    
    # logs_2025-05-31_00-17-09 주스 던지기: disable optimizer
    # plot_pos_vel_acc(config, all_logs, logs_path, idx_start=0, idx_end=17*400, joint_index=5)
    # two_subplot(all_logs, logs_path, idx_start=23*400, idx_end=34*400, joint_index=5)
    # two_subplot(all_logs, logs_path, idx_start=29*400, idx_end=34*400, joint_index=5)
    
    # logs_2025-05-31_00-18-39 주스 던지기: enable optimizer
    # plot_pos_vel_acc(config, all_logs, logs_path, idx_start=0, idx_end=17*400, joint_index=5)
    # two_subplot(all_logs, logs_path, idx_start=18*400, idx_end=29*400, joint_index=5)
    two_subplot(all_logs, logs_path, idx_start=24*400, idx_end=29*400, joint_index=5)
    
    # plot_pos_vel_acc(config, all_logs, logs_path, joint_index=5)
    
    