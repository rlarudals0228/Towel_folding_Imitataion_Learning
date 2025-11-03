# 🧺 Towel Folding Imitation Learning  
### Tunable Terminal Condition Classification (TTCC) 기반 장기 비정형체 조작 모방학습

---

## 📘 개요  
본 프로젝트는 *외부 RGB-D 카메라 기반 정량적 시각 지표**를 활용해  로봇이 수건을 **자율적으로 평탄화(Flatten)하고 접기(Fold)** 하는  
**2단계 모방학습 프레임워크**를 구현한 연구입니다.

기존 연구처럼 “시간이나 단순 시각 피처”가 아니라  **정량 지표(metric)** 를 이용해 각 단계의 **종료 조건을 실시간 판단**하고,  
이를 조정 가능한 형태(**Tunable Terminal Condition**)로 설계한 점입니다.

> 🧩 목표: 로봇이 실시간으로 수건의 상태를 평가하여  
> ‘아직 펴야 하는가(FLATTEN)’ 또는 ‘접을 때인가(FOLD)’를 스스로 판단  

---


## 🧠 Key Contributions
- **Two-Stage Imitation Learning**  
  - Stage 1 – Flattening policy learned from teleoperation + DAgger corrections  
  - Stage 2 – Folding policy learned from roughly flattened towel states  
- **TTCC Model (Tunable Threshold Decision)**  
  - Real-time evaluation of towel geometry to determine transition timing  
  - Thresholds adjustable for different task environments (home vs industrial)
- **Integrated System (Vision + Control + Learning)**  
  - RealSense D415 (top view) + D405 (wrist view)  
  - ROS 2 Jazzy + MoveIt2 + ACT inference pipeline  
  - Compatible with OpenManipulator-Y hardware  

---

### 📊 정량 지표 (Quantitative Visual Metrics)
| 지표 | 설명 | 의미 |
|------|------|------|
| **Rectangularity Fit (Rfit)** | 수건 외곽의 사각형 정합도 | 1에 가까울수록 평탄 |
| **Height Std (σₕ)** | 표면 높이의 표준편차 | 작을수록 균일 |
| **Height Range (Δh)** | 표면 전체의 높이 차이 | 작을수록 평평 |

TTCC 모델은 이 세 가지 지표를 임계값(threshold)과 비교하여 상태를 분류합니다:
- `FLATTEN`: 아직 평탄화 필요  
- `FOLD`: 접기 정책으로 전환  

---

### Development Environment 
- OS: Ubuntu 24.04 LTS (Detail: Linux ubuntu 6.11.0-25-generic #25~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 15 17:20:50 UTC 2 x86_64 x86_64 x86_64 GNU/Linux)
- ROS2: Jazzy
- Python: 3.12.3

### Reference
- https://github.com/huggingface/lerobot
- https://github.com/ROBOTIS-GIT/open_manipulator

## 1. 모방학습 데이터 수집

### Terminal 1
```
ssh root@192.168.0.138
docker exec -it open_manipulator bash
source /workspace/colcon_ws/install/setup.bash
ros2 launch open_manipulator_bringup ai_teleoperation.launch.py
```

### Terminal 2
```
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py config_file:="realsense_config.yaml"
```

### Terminal 3
```
source /opt/ros/jazzy/setup.bash
rqt
```

### Terminal 4
```
source /opt/ros/jazzy/setup.bash
cd ~/colcon_ws
source install/setup.bash
ros2 launch ros2_lerobot create_datasheet.launch.py
```

## 2. 수집한 데이터 train & evaluation

### Visualize
```
conda activate mani
python -m lerobot.scripts.visualize_dataset  
--repo-id omy_real  -- root /home/dam/colcon_ws/src/ros2_lerobot/demo_data/towel_folding -- episode- index 0
```

### Train & Evalaution
```
conda activate mani
python train.py 
python evaluation.py
```

## 3. 모방학습 실행

### Terminal 1
```
ssh root@192.168.0.138
docker exec -it open_manipulator bash
source /workspace/colcon_ws/install/setup.bash
ros2 launch open_manipulator_bringup ai_inference.launch.py
```

### Terminal 2 (D415 camera ON)
```
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=external_camera \
  device_type:=d415 \
  align_depth:=true \
  enable_color:=true \
  enable_depth:=true \
  pointcloud.enable:=true
```

### Termianl 3 (수건 탐지 실행)
```
cd colcon_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run ros2_lerobot realsense_towel_metrics -- \
  --color /camera/external_camera/color/image_raw \
  --depth /camera/external_camera/depth/image_rect_raw \
  --rect-thr 0.85 --std-thr-mm 7.0 --range-thr-mm 18.0
```

### Terminal 4 (D405 camera ON)
```
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=arm_camera \
  device_type:=d405 \
  align_depth:=true \
  enable_depth:=true \
  pointcloud.enable:=true
```

### Terminal 5
```
source /opt/ros/jazzy/setup.bash
source install/setup.bash
rqt
```

### Terminal 6
```
source /opt/ros/jazzy/setup.bash
cd ~/colcon_ws
source install/setup.bash
ros2 launch ros2_lerobot inference_service_towel.launch.py
```
## 📊 Results (결과)

![Result_1](result/result_1.gif)
![Result_2](result/result_2.gif)
