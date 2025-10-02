import os
import numpy as np
import matplotlib.pyplot as plt

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

def plot_chunks_with_te(pre_chunks, te_actions, joint_index=0, sample_every=1):
    """
    pre_chunks : (N, H, A)  # 매 스텝별 예측 청크 (Pre-TE)
    te_actions : (N, A)     # TE로 블렌딩된 최종 액션(시계열)
    joint_index : 시각화할 조인트 인덱스 (0..A-1)
    sample_every: 청크 라인 샘플링 간격(밀도 조절)
    """
    # torch.Tensor면 numpy로
    if hasattr(pre_chunks, "detach"):
        pre_chunks = pre_chunks.detach().cpu().numpy()
    if hasattr(te_actions, "detach"):
        te_actions = te_actions.detach().cpu().numpy()

    Np, H, A = pre_chunks.shape
    Nt, At   = te_actions.shape
    assert A == At, f"dim mismatch: pre_chunks A={A}, te_actions A={At}"
    N = min(Np, Nt)  # 안전하게 공통 길이만 사용

    plt.figure(figsize=(12, 4))

    # 1) 매 시점 t에서 시작점 x=t로 청크 라인 오버레이
    for t in range(0, N, sample_every):
        x = t + np.arange(H)                  # x=t에서 시작
        y = pre_chunks[t, :, joint_index]     # 길이 H
        m = (x >= 0) & (x < N)                # x축 [0..N-1]만 표시
        if m.any():
            plt.plot(x[m], y[m], linewidth=0.8, alpha=0.25)

    # 2) 청크 시작점(각 t의 offset=0) 포인트
    start_y = pre_chunks[:N, 0, joint_index]
    plt.scatter(np.arange(N), start_y, s=10, zorder=3, label="chunk start", marker='o')

    # 3) TE 결과(블렌딩 후) 시계열
    te_y = te_actions[:N, joint_index]
    plt.scatter(np.arange(N), te_y, s=16, zorder=3, label="TE action", marker='*')
    
    plt.xlim(0, N - 1)
    plt.xlabel("time step (t)")
    plt.ylabel(f"joint {joint_index} value")
    plt.title(f"Per-step chunks + TE action — joint {joint_index}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    logs_root_path = os.path.join(os.path.dirname(current_file_path), '..', 'logs')
    # latest_logs_folder = get_latest_log_folder(logs_root_path)
    # logs_path = os.path.join(logs_root_path, latest_logs_folder)
    # print(f"folder names:", latest_logs_folder)

    logs_path = logs_root_path + "/Only TE4 with TE pre actions"

    all_logs = load_all_logs(logs_path)
    
    print(all_logs["TE_pre_actions"].shape)
    print(all_logs["raw_actions"].shape)
    pre_chunks = all_logs["TE_pre_actions"]  # (288, 100, 14)
    te_actions = all_logs["raw_actions"]      # (288, 14)
    plot_chunks_with_te(pre_chunks, te_actions, joint_index=3, sample_every=1)
    
