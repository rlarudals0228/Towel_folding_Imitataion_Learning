import os
import numpy as np

def load_all_logs(log_dir):
    logs = {}
    for filename in os.listdir(log_dir):
        if filename.startswith("log_") and filename.endswith(".npy"):
            log_name = filename.replace("log_", "").replace(".npy", "")
            log_path = os.path.join(log_dir, filename)
            log_data = np.load(log_path, allow_pickle=True)
            logs[log_name] = log_data
    return logs

def _as_list(x):
    # numpy object array일 수 있으므로 리스트로 변환
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def extract_runs_and_times(solve_log_obj):
    """
    solve_time 로그가 [times1, max1, mean1, times2, max2, mean2, ...] 형태(extend로 저장)라고 가정.
    - 여러 회차의 times를 모두 합쳐 overall 통계를 계산
    - 각 회차별(세 개가 한 묶음) 요약도 제공
    """
    seq = _as_list(solve_log_obj)
    if not isinstance(seq, list):
        raise ValueError("solve_time 로그 형식이 리스트가 아닙니다.")

    runs = []
    all_times = []

    i = 0
    while i < len(seq):
        item = seq[i]
        # times 묶음(리스트/ndarray) 탐지
        if isinstance(item, (list, np.ndarray)):
            times = np.array(item, dtype=float)
            run = {"times": times}
            # 뒤에 이어지는 max, mean(숫자) 있으면 함께 기록
            if i + 2 < len(seq) and isinstance(seq[i+1], (int, float, np.floating)) and isinstance(seq[i+2], (int, float, np.floating)):
                run["max_saved"] = float(seq[i+1])
                run["mean_saved"] = float(seq[i+2])
                i += 3
            else:
                i += 1
            runs.append(run)
            all_times.extend(times.tolist())
        else:
            # 형식이 다르거나 잔여 숫자만 있는 경우 건너뜀
            i += 1

    all_times = np.array(all_times, dtype=float)
    return runs, all_times

def print_summary(all_times, runs, time_unit="s"):
    def f(x): return f"{x:.6f}"

    print("\n=== Solve Time Summary (Overall) ===")
    if all_times.size == 0:
        print("데이터가 없습니다.")
        return

    p50, p90, p95, p99 = np.percentile(all_times, [50, 90, 95, 99])

    # 각 항목을 한 줄씩 출력하기 위한 행 데이터 구성
    rows = [
        ("count", str(all_times.size), ""),
        ("min",   f(all_times.min()),  f"[{time_unit}]"),
        ("p50",   f(p50),              f"[{time_unit}]"),
        ("mean",  f(all_times.mean()), f"[{time_unit}]"),
        ("std",   f(all_times.std(ddof=1)), f"[{time_unit}]"),
        ("p90",   f(p90),              f"[{time_unit}]"),
        ("p95",   f(p95),              f"[{time_unit}]"),
        ("p99",   f(p99),              f"[{time_unit}]"),
        ("max",   f(all_times.max()),  f"[{time_unit}]"),
    ]

    # 예쁘게 정렬(좌측 키 폭 고정)
    left_w = max(len(name) for name, _, _ in rows)
    for name, val, unit in rows:
        line = f"{name:<{left_w}} : {val} {unit}".rstrip()
        print(line)

    if all_times.mean() > 0:
        print(f"throughput ≈ {1.0 / all_times.mean():.2f} iters/sec")

    # 회차별 요약은 기존 형식 유지(원하면 이것도 행 단위로 바꿔줄 수 있어)
    if runs:
        print("\n=== Per-Run Quick View ===")
        print("idx | count | mean | max  (saved_mean/max 있을 경우 함께 표기)")
        for idx, r in enumerate(runs):
            times = r["times"]
            mean_c = times.mean() if times.size else float("nan")
            max_c  = times.max()  if times.size else float("nan")
            msg = f"{idx:>3} | {times.size:>5} | {f(mean_c)} | {f(max_c)}"
            if "mean_saved" in r or "max_saved" in r:
                msg += "   (saved:"
                if "mean_saved" in r:
                    msg += f" mean={f(r['mean_saved'])}"
                if "max_saved" in r:
                    if "mean_saved" in r: msg += ","
                    msg += f" max={f(r['max_saved'])}"
                msg += ")"
            print(msg)

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    logs_root_path = os.path.join(os.path.dirname(current_file_path), '..', 'logs')

    # 예시: 특정 로그 폴더를 직접 지정
    logs_path = os.path.join(logs_root_path, "logs_2025-08-18_01-51-26")
    all_logs = load_all_logs(logs_path)

    print("log path", logs_path)

    if "solve_time" not in all_logs:
        raise FileNotFoundError("log_solve_time.npy를 찾을 수 없습니다. 저장 로직을 확인하세요.")

    runs, all_times = extract_runs_and_times(all_logs["solve_time"])
    print_summary(all_times, runs, time_unit="s")
