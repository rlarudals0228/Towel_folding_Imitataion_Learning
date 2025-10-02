import os
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math_func import finite_difference_derivative

def _compute_series(all_logs, idx_start, idx_end, joint_index):
    # 타임스탬프 & 시그널 추출
    timestamps_target_pos = np.array([x[0] for x in all_logs["publish_action"]])
    timestamps_target_pos -= timestamps_target_pos[0]
    target_pos = np.array([x[1] for x in all_logs["publish_action"]])

    timestamps_current_pos = np.array([x[0] for x in all_logs["subscribe_joint_pos"]])
    timestamps_current_pos -= timestamps_current_pos[0]
    current_pos = np.array([x[1] for x in all_logs["subscribe_joint_pos"]])

    timestamps_current_vel = np.array([x[0] for x in all_logs["subscribe_joint_vel"]])
    timestamps_current_vel -= timestamps_current_vel[0]
    current_vel = np.array([x[1] for x in all_logs["subscribe_joint_vel"]])

    # 파생량
    target_vel = finite_difference_derivative(target_pos, 1, 0.0025, 0)
    target_acc = finite_difference_derivative(target_vel, 1, 0.0025, 0)
    current_acc = finite_difference_derivative(current_vel, 1, 0.0025, 0)

    # 슬라이스 범위 보정
    N = len(all_logs["subscribe_joint_pos"])
    if idx_end is None or idx_end > N:
        idx_end = N

    # 기준 시간(t0) — 파일별 상대시간 정렬
    t0 = timestamps_target_pos[idx_start]

    return {
        "t0": t0,
        "t_pos_target": timestamps_target_pos[idx_start:idx_end] - t0,
        "t_pos_curr":   timestamps_current_pos[idx_start:idx_end] - t0,
        "pos_target":   target_pos[idx_start:idx_end, joint_index-1],
        "pos_curr":     current_pos[idx_start:idx_end, joint_index-1],
        "t_vel":        timestamps_current_pos[idx_start:idx_end] - t0,  # 속도는 current 타임에 맞춤
        "vel_target":   target_vel[idx_start:idx_end, joint_index-1],
        "vel_curr":     current_vel[idx_start:idx_end, joint_index-1],
        "t_acc":        timestamps_target_pos[idx_start:idx_end] - t0,  # 가속도는 target 타임에 맞춤
        "acc_target":   target_acc[idx_start:idx_end, joint_index-1],
        "acc_curr":     current_acc[idx_start:idx_end, joint_index-1],
        "idx_end":      idx_end
    }

def plot_pos_vel_acc_overlay(configA, all_logs_A, configB, all_logs_B, save_dir,
                             idx_start=0, idx_end=None, joint_index=1,
                             labelA="A", labelB="B"):
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'pdf.fonttype': 42, 'ps.fonttype': 42
    })
    cmap = plt.get_cmap('tab10')

    A = _compute_series(all_logs_A, idx_start, idx_end, joint_index)
    B = _compute_series(all_logs_B, idx_start, idx_end, joint_index)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- Position ---
    # A: solid, B: dashed
    axes[0].plot(A["t_pos_target"], A["pos_target"], color=cmap(0), alpha=0.7, linestyle='-',  label=f'{labelA} Target Pos')
    # axes[0].plot(A["t_pos_curr"],   A["pos_curr"],   color=cmap(3), alpha=0.7, linestyle='-',  label=f'{labelA} Current Pos')
    axes[0].plot(B["t_pos_target"], B["pos_target"], color=cmap(0), alpha=0.7, linestyle='--', label=f'{labelB} Target Pos')
    # axes[0].plot(B["t_pos_curr"],   B["pos_curr"],   color=cmap(3), alpha=0.7, linestyle='--', label=f'{labelB} Current Pos')
    axes[0].set_ylabel("Position [rad]")
    axes[0].legend(loc='upper left')

    # --- Velocity ---
    axes[1].plot(A["t_vel"], A["vel_target"], color=cmap(0), alpha=0.7, linestyle='-',  label=f'{labelA} Target Vel')
    # axes[1].plot(A["t_vel"], A["vel_curr"],   color=cmap(3), alpha=0.7, linestyle='-',  label=f'{labelA} Current Vel')
    axes[1].plot(B["t_vel"], B["vel_target"], color=cmap(0), alpha=0.7, linestyle='--', label=f'{labelB} Target Vel')
    # axes[1].plot(B["t_vel"], B["vel_curr"],   color=cmap(3), alpha=0.7, linestyle='--', label=f'{labelB} Current Vel')
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[1].legend(loc='upper left')

    # --- Acceleration ---
    axes[2].plot(A["t_acc"], A["acc_target"], color=cmap(0), alpha=0.7, linestyle='-',  label=f'{labelA} Target Acc')
    # axes[2].plot(A["t_acc"], A["acc_curr"],   color=cmap(3), alpha=0.7, linestyle='-',  label=f'{labelA} Current Acc')
    axes[2].plot(B["t_acc"], B["acc_target"], color=cmap(0), alpha=0.7, linestyle='--', label=f'{labelB} Target Acc')
    # axes[2].plot(B["t_acc"], B["acc_curr"],   color=cmap(3), alpha=0.7, linestyle='--', label=f'{labelB} Current Acc')
    axes[2].set_ylabel("Acceleration [rad/s²]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc='upper left')

    plt.tight_layout()
    out_pdf = os.path.join(save_dir, f'pos_vel_acc_overlay_J{joint_index}.pdf')
    out_png = os.path.join(save_dir, f'pos_vel_acc_overlay_J{joint_index}.png')
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close(fig)


def two_subplot_overlay(all_logs_A, all_logs_B, save_dir,
                        idx_start_A=0, idx_end_A=None,
                        idx_start_B=0, idx_end_B=None, 
                        joint_index=1,
                        labelA="A", labelB="B",
                        pos_ylim=None, vel_ylim=None, acc_ylim=None):
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'pdf.fonttype': 42, 'ps.fonttype': 42
    })
    cmap = plt.get_cmap('tab10')

    A = _compute_series(all_logs_A, idx_start_A, idx_end_A, joint_index)
    B = _compute_series(all_logs_B, idx_start_B, idx_end_B, joint_index)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # --- (1) Position ---
    axes[0].plot(A["t_pos_curr"], A["pos_curr"], color=cmap(0), linestyle='-',  label=f'{labelA} pos')
    axes[0].plot(B["t_pos_curr"], B["pos_curr"], color=cmap(3), linestyle='--', label=f'{labelB} pos')
    axes[0].set_ylabel("Position (rad)")
    if pos_ylim is not None:
        axes[0].set_ylim(*pos_ylim)
    axes[0].grid(True)
    axes[0].legend(loc='upper left')

    # --- (2) Velocity + Acceleration (twin y) ---
    ax_vel = axes[1]
    ax_acc = ax_vel.twinx()

    # Velocity (left y)
    ax_vel.plot(A["t_vel"], A["vel_target"], color=cmap(0), linestyle='-',  alpha=0.7, label=f'{labelA} vel')
    ax_vel.plot(B["t_vel"], B["vel_target"], color=cmap(3), linestyle='--', alpha=0.5, label=f'{labelB} vel')
    ax_vel.set_ylabel("Velocity (rad/s)")
    if vel_ylim is not None:
        ax_vel.set_ylim(*vel_ylim)
    ax_vel.grid(True)

    # Acceleration (right y)
    ax_acc.plot(A["t_acc"], A["acc_target"], color=cmap(2), linestyle='-',  alpha=0.7, label=f'{labelA} acc')
    ax_acc.plot(B["t_acc"], B["acc_target"], color=cmap(6), linestyle='--', alpha=0.5, label=f'{labelB} acc')
    ax_acc.set_ylabel("Acceleration (rad/s²)")
    if acc_ylim is not None:
        ax_acc.set_ylim(*acc_ylim)

    axes[1].set_xlabel("Time (s)")

    # 범례 통합
    lines_1, labels_1 = ax_vel.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    out_pdf = os.path.join(save_dir, f'pos_velacc_overlay_J{joint_index}.pdf')
    out_png = os.path.join(save_dir, f'pos_velacc_overlay_J{joint_index}.png')
    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close(fig)

def load_all_logs(log_dir):
    logs = {}
    for filename in os.listdir(log_dir):
        if filename.startswith("log_") and filename.endswith(".npy"):
            log_name = filename.replace("log_", "").replace(".npy", "")
            log_path = os.path.join(log_dir, filename)
            log_data = np.load(log_path, allow_pickle=True)
            logs[log_name] = log_data
    return logs

if __name__ == "__main__":
    # 예: A, B 두 실험 폴더
    logs_dir_A = 'logs_2025-08-17_23-28-21'
    logs_dir_B = 'logs_2025-08-17_23-32-42'

    current_file_path = os.path.abspath(__file__)
    logs_root_path = os.path.join(os.path.dirname(current_file_path), '..', 'logs')

    logs_path_A = os.path.join(logs_root_path, logs_dir_A)
    logs_path_B = os.path.join(logs_root_path, logs_dir_B)

    with open(os.path.join(logs_path_A, "experiment_config.yaml"), "r") as f:
        configA = yaml.safe_load(f)
    with open(os.path.join(logs_path_B, "experiment_config.yaml"), "r") as f:
        configB = yaml.safe_load(f)

    all_logs_A = load_all_logs(logs_path_A)
    all_logs_B = load_all_logs(logs_path_B)

    # 저장은 A 폴더에
    save_dir = logs_path_A

    # 공통 인덱스/조인트 선택
    idx_start_A = 20*400
    idx_end_A   = 24*400
    idx_start_B = 20*400
    idx_end_B   = 24*400
    joint_index = 5

    # (1) 3-서브플롯 오버레이
    # plot_pos_vel_acc_overlay(configA, all_logs_A, configB, all_logs_B, save_dir,
    #                          idx_start=idx_start, idx_end=idx_end,
    #                          joint_index=joint_index,
    #                          labelA="LiPo", labelB="Raw")

    # (2) 2-서브플롯(위치 / 속도+가속도 2축) 오버레이
    two_subplot_overlay(all_logs_A, all_logs_B, save_dir,
                        idx_start_A=idx_start_A, idx_end_A=idx_end_A,
                        idx_start_B=idx_start_B, idx_end_B=idx_end_B,
                        joint_index=joint_index,
                        labelA="Quintic", labelB="Cubic",
                        pos_ylim=(0.4, 2.0), vel_ylim=(-4.0, 2.0), acc_ylim=(-100, 100))
