# Machine Perception System

## Runtime Flow

`EuRoC Input -> LocalProcessingIngress -> YOLO + Native ORB Adapter -> ServerPacketBridge -> Preview`

- client entrypoint: `client/main.py`
- local services:
  - `client/local_services/DetectionService.py`
  - `client/local_services/SlamService.py`
- native ORB adapter:
  - `integration/orb_native/orb_native_adapter.cpp`

## Native ORB Adapter

`SlamService` sends TCP requests to a native C++ adapter (no ROS):

- request payload: frame timestamp + JPEG image + IMU samples
- adapter calls `ORB_SLAM3::System::TrackMonocular(...)`
- response payload: tracking state + camera/body pose + tracking image

### Build Adapter

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
integration/orb_native/build_native_adapter.sh
```

### Run Adapter

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
integration/orb_native/run_native_adapter.sh
```

Useful env overrides:

- `MPS_ORB_ROOT` (ORB-SLAM3 root with `lib/libORB_SLAM3.so`)
- `MPS_ORB_VOC_FILE` (defaults to `<ORB_ROOT>/Vocabulary/ORBvoc.txt`; override only if your fork supports binary voc)
- `MPS_ORB_SETTINGS_FILE` (explicit YAML path; overrides profile/default selection)
- `MPS_ORB_PROFILE` (`auto`/`tum_vi`/`euroc`; default `auto`)
- `MPS_SLAM_BIND_HOST` (default `0.0.0.0`)
- `MPS_SLAM_PORT` (default `19090`)
- `MPS_ORB_USE_VIEWER` (`1` to enable Pangolin)
- `MPS_ENABLE_TRAJECTORY_3D` (`1` to open Matplotlib 3D trajectory window from ORB camera pose)

## Run Client

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_DATASET_ROOT=/home/tranduong/dev/thesis_prototype/dataset/dataset-corridor1_512_16 \
MPS_YOLO_MODEL_PATH=/home/tranduong/dev/thesis_prototype/models/yolo26n.pt \
MPS_SLAM_HOST=127.0.0.1 \
MPS_SLAM_PORT=19090 \
MPS_SLAM_TIMEOUT_S=1.5 \
MPS_REPLAY_SPEED=0.0 \
python3 client/main.py
```

Optional:
- `MPS_MAX_FRAMES=20` for smoke tests.

## Models

- `models/yolo26n.pt` (YOLO model)
- `models/orbvoc.txt.bin` (ORB vocabulary used by native adapter)
