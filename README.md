# 🧺 Towel Folding Robot using Imitation Learning  
### Vision-based Two-Stage Policy Switching Framework (TTCC)

---

## 📘 프로젝트 개요  
본 프로젝트는 **수건을 자율적으로 평탄화하고 접는 로봇 시스템**을 개발한 연구로,  
로봇이 외부 카메라로부터 수건의 상태를 인식하고,  
상태에 따라 스스로 “펴기(Flatten)” 또는 “접기(Fold)” 정책을 전환하도록 설계되었습니다.  

이를 위해 **2단계 모방학습 기반 프레임워크**를 구성하고,  
정량적 시각 지표를 활용한 **Tunable Terminal Condition Classification (TTCC)** 모델을 도입했습니다.  
즉, 사람이 개입하지 않아도 로봇이 스스로 “지금 접을 때인지”를 판단할 수 있는 구조입니다.

---

## 🦾 시스템 구성  
<p align="center">
  <img width="1575" height="1181" alt="Image" src="https://github.com/user-attachments/assets/b499651c-3f75-4e4b-9aa8-5ddee834624d" />
</p>

- **로봇:** ROBOTIS OpenManipulator-Y (6자유도 + 1자유도 그리퍼)  
- **카메라:** Intel RealSense D405 (손목 장착) / D415 (상단 뷰)  
- **학습 프레임워크:** HuggingFace LeRobot 기반 ACT(Action Chunking Transformer)  
- **ROS2:** Jazzy 환경 (Ubuntu 24.04)  

---

## 👁️ 비전 기반 상태 판단 파이프라인  
<p align="center">
  <img width="3150" height="1476" alt="Image" src="https://github.com/user-attachments/assets/330836ed-2784-49e8-b8eb-14b59c57d0f7" />
</p>

외부 RGB-D 카메라에서 획득한 깊이 데이터를 이용해  
수건의 **평탄화 정도를 정량적으로 분석**하고 정책 전환 여부를 판단합니다.

1. **Realsense D415 카메라 실행**
2. **3D 포인트 재구성** (RealSense 깊이 영상)  
3. **평면 추정 (RANSAC)** – 작업대 기준면 계산  
4. **잔차맵 생성** – 수건 표면의 높이 변화 시각화  
5. **수건 영역 추출** – 가장 큰 윤곽선으로 마스크 정의  
6. **지표 계산 및 TTCC 판단**  
   - Rectangularity Fit  
   - Height Standard Deviation  
   - Height Range  

이 세 지표가 임계값을 만족하면 “접기(Fold)”로 전환,  
그렇지 않으면 “평탄화(Flatten)”를 반복 수행합니다.

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

## 🎛️ 조정 가능한 임계값 (Tunable Thresholds)
| 적용 환경 | Rect Fit (> ) | Height Std (< mm) | Height Range (< mm) |
|------------|---------------|--------------------|---------------------|
| 산업 환경 (엄격) | 0.85 | 7 | 18 |
| 가정/실험 환경 (완화) | 0.77 | 15 | 30 |

→ 사용자는 환경과 목적에 따라 전환 기준 민감도를 조정할 수 있습니다.  
예: 공장 환경은 정밀 기준, 일반 환경은 완화된 기준 적용

---

## 🤖 모방학습 기반 제어 구조  
| 단계 | 설명 | 데이터 수 | 학습 방법 |
|------|------|------------|-----------|
| **Stage 1. Flattening** | 구겨진 수건을 펴고 대칭 정렬 | 60개 (DAgger 포함) | ACT |
| **Stage 2. Folding** | 평탄화된 수건을 반으로 접기 | 30개 | ACT |

- 텔레오퍼레이션 기반 시연 데이터 수집 (Leader–Follower 방식)  
- HuggingFace `LeRobot` 프레임워크로 RGB-D + 관절 데이터 동기화  
- ROS2 환경에서 두 정책을 개별 학습 후, TTCC 모델로 통합 실행  

---

## 📈 실험 결과  
| 조건 | 성공률 | 설명 |
|------|---------|------|
| 평탄화만 수행 | 76.7% | 구김 제거 정확도 |
| 접기만 수행 | 93.3% | 단일 동작 안정적 |
| 수동 전환 (Flatten → Fold) | 63.3% | 전환 타이밍 불안정 |
| **TTCC 자율 전환** | **80.0%** ✅ | 정량 지표 기반 안정적 전환 |

> TTCC 모델 적용 시, 잘못된 전환으로 인한 실패가 16%p 감소하고  
> 장기(long-horizon) 작업의 안정성이 개선됨을 확인했습니다.

---

## 🔧 개발 환경  
| 항목 | 내용 |
|------|------|
| OS | Ubuntu 24.04 LTS |
| ROS | ROS2 Jazzy |
| Python | 3.12 |
| GPU | RTX 5070 Ti |
| Framework | HuggingFace LeRobot |
| Depth Camera | Intel RealSense D415 / D405 |
| Manipulator | ROBOTIS OpenManipulator-Y |

---

## 📬 Contact  
**김경민** (Kwangwoon University, Dept. of Robotics)  
📧 rlarudals0228@naver.com  
🔗 [GitHub Repository](https://github.com/rlarudals0228/Towel_folding_Imitataion_Learning)

---

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
