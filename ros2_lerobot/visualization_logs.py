import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ros2_lerobot.math_func import finite_difference_derivative

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

def plot(logs, keys, joint_index=0):

    # plt.figure(figsize=(6, 4.5))
    # plt.figure(figsize=(4.5, 3.5))
    plt.figure(figsize=(6, 4.5))
    
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    font_prop = FontProperties(fname=font_path)
    
    mpl.rcParams.update({
        'font.family': 'serif',
        # 'font.serif': ['Times New Roman'],  # 또는 ['Times'] for compatibility
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'pdf.fonttype': 42,  # To embed fonts properly in PDF
        'ps.fonttype': 42
    })
    
    fontdict={
              'fontsize': 18,
              'style': 'italic', # 'oblique'
              'fontweight': 'light'}  # 'heavy', 'light', 'ultrabold', 'ultralight'
    
    color_map = {"inference_actions": "blue",
                 "spline_actions": "red",
                 "solved": "purple",
                 "ref.value": "green",
                 "publish_action": "orange",
                 "subscribe_joint": "black",
                 "x1": "red",
                 "current_chunk_index": "blue",
                 "raw_actions": "blue",
                 "actions_dot": "red",
                 "pos": "blue",
                 "vel": "red",
                 "TE_pre_actions": "green"}
    
    for key in keys:
        if key in logs:
            print(f"{key} shape: {logs[key].shape}")
            
            log = np.array(logs[key], dtype=object)
            
            if log.shape[1] == 2:
                timestamps = np.array([x[0] for x in log])
                timestamps -= timestamps[0]  # 정규화
                values = np.array([x[1] for x in log])
            else:
                timestamps = np.array([i * 0.0333 for i in range(len(log))])
                values = np.array([x for x in log])
            
            if key == "current_chunk_index":
                plt.plot(timestamps, values / 10000.0, color=color_map.get(key, "gray"), alpha=0.5, label=key)
            else:
                if key == "pos":
                    plt.plot(timestamps, values[:, joint_index], color=color_map.get(key, "gray"), alpha=0.5, label="Position (rad)")
                if key == "vel":
                    plt.plot(timestamps, values[:, joint_index], color=color_map.get(key, "gray"), alpha=0.5, label="Velocity (rad/s)")
                    
    plt.ylim(-3.14, 3.14)
    # plt.xlim(0.0, 3.14)
    # plt.xlabel("Time (s)", fontproperties=font_prop, **fontdict)
    # plt.ylabel(f"Joint {joint_index+1} Value", fontproperties=font_prop, **fontdict)
    # plt.title(f"Joint {joint_index+1} Trajectories", fontproperties=font_prop, **fontdict)
    plt.xlabel("Time (s)", fontproperties=font_prop, size=18)
    plt.ylabel(f"Value", fontproperties=font_prop, size=18)
    # plt.title(f"Joint {joint_index+1} Trajectories")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    logs_root_path = os.path.join(os.path.dirname(current_file_path), '..', 'logs')
    # latest_logs_folder = get_latest_log_folder(logs_root_path)
    # logs_path = os.path.join(logs_root_path, latest_logs_folder)
    # print(f"folder names:", latest_logs_folder)

    logs_path = logs_root_path + "/Only TE4 with TE pre actions"
    # logs_path = logs_root_path + "/logs_2025-08-17_23-28-21"

    all_logs = load_all_logs(logs_path)

    all_logs["actions_dot"] = finite_difference_derivative(all_logs["raw_actions"], 1, 0.0333, 0)

    all_logs["pos"] = all_logs["raw_actions"]
    all_logs["vel"] = all_logs["actions_dot"]

    print(all_logs["TE_pre_actions"].shape)
    print("Max:", np.max(all_logs["vel"]))
    plot(all_logs, keys=("pos", "vel"), joint_index=3)
    # plot(all_logs, keys=("raw_actions", "solved", "ref.value"), joint_index=3)
    # plot(all_logs, keys=("spline_actions", "x1"))
    # plot(all_logs, keys=("publish_action", "subscribe_joint"))
