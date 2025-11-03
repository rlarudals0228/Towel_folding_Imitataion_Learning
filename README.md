# 🧺 Towel Folding Imitation Learning  
**Two-Stage Imitation-Learning Framework for Deformable-Object Manipulation**

---

## 🌟 Overview  
This project implements a **two-stage imitation-learning system** that enables a robotic arm to **flatten and fold a towel autonomously**.  
The framework integrates **ROS 2**, **RealSense depth sensing**, and **Action Chunking Transformer (ACT)**-based visuomotor policy learning.

The system is designed to handle **long-horizon deformable-object tasks**, where a robot must decide when to continue flattening or when to switch to folding.  
To achieve this, a **Tunable Terminal Condition Classification (TTCC)** module analyzes quantitative visual metrics (rectangularity fit, height std, height range) from RGB-D data to make adaptive policy-switching decisions.

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
