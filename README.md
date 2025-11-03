## 1. 모방학습 데이터 수집 


### Terminal 1
```
ssh root@192.168.0.138
docker exec -it open_manipulator bash
source /workspace/colcon_ws/install/setup.bash
ros2 launch open_manipulator_bringup ai_teleoperation.launch.py
```

### Terminal 2
```source /opt/ros/jazzy/setup.bash
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
