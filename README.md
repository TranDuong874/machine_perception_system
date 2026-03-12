# Machine Perception System (POC)

## Goal
End-to-end POC pipeline:

`Input -> LocalProcessingIngress -> YOLO + ORB -> ServerPacketBridge (gRPC) -> Server pipeline`

## Current Architecture

### Client
- `client/main.py` orchestrates only.
- `user_interface/UserViewRenderer` renders wearable-facing user view from local frames.
- `SlamService` talks to local native ORB adapter (TCP, local).
- `ServerPacketBridge` sends `ServerPerceptionPacket` to server via gRPC + Protobuf.
- `ServerPacketBridge` receives `TopDownMap` payloads in gRPC replies and updates client-side map cache.
- Local processing preview window remains available for debugging.

### Server
- `server/main.py` orchestrates only.
- gRPC ingress (`SubmitPacket`) receives packets.
- Server pipeline runtime:
  - ingress queue
  - fan-out router
  - telemetry worker
  - persistence worker (JSONL)
  - top-down mapper worker (fisheye undistort -> sparse depth backprojection -> 2D occupancy)

## Protobuf / gRPC Contract

- Proto source: `proto/perception.proto`
- Generated files:
  - `proto/perception_pb2.py`
  - `proto/perception_pb2_grpc.py`

Regenerate after proto changes:
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
/home/tranduong/dev/thesis_prototype/.venv/bin/python -m grpc_tools.protoc \
  -I. --python_out=. --grpc_python_out=. proto/perception.proto
```

## Prerequisites

- Python env with:
  - `grpcio`
  - `grpcio-tools`
  - `protobuf`
  - `opencv-python`
  - `ultralytics`
- ORB-SLAM3 fixed repo built and available at:
  - `dependency/ORB_SLAM3_FIXED`
  - with `lib/libORB_SLAM3.so`, `Thirdparty/DBoW2/lib/libDBoW2.so`, `Thirdparty/g2o/lib/libg2o.so`
- UniDepth GPU env (recommended for server):
  - `/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/depth_spike/.venv`

Prepare fixed ORB dependency in this repo (example symlink):
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
mkdir -p dependency
ln -s /home/tranduong/dev/thesis_prototype/spatial_mapping_demo/ORB-SLAM3-STEREO-FIXED dependency/ORB_SLAM3_FIXED
```

## 1) Build Native ORB Adapter

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
integration/orb_native/build_native_adapter.sh
```

## 2) Run ORB Adapter (Terminal A)

### Normal
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_DATASET_ROOT=/home/tranduong/dev/thesis_prototype/dataset/dataset-corridor1_512_16 \
MPS_ORB_ROOT=/home/tranduong/dev/thesis_prototype/machine_perception_system/dependency/ORB_SLAM3_FIXED \
MPS_ORB_PROFILE=tum_vi \
MPS_SLAM_PORT=19090 \
integration/orb_native/run_native_adapter.sh
```

### Debug (Pangolin)
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_DEBUG_MODE=1 \
MPS_DATASET_ROOT=/home/tranduong/dev/thesis_prototype/dataset/dataset-corridor1_512_16 \
MPS_ORB_ROOT=/home/tranduong/dev/thesis_prototype/machine_perception_system/dependency/ORB_SLAM3_FIXED \
MPS_ORB_PROFILE=tum_vi \
MPS_SLAM_PORT=19090 \
integration/orb_native/run_native_adapter.sh
```

## 3) Run Server (Terminal B)

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_SERVER_BIND_HOST=0.0.0.0 \
MPS_SERVER_PORT=19100 \
MPS_SERVER_TOPDOWN_ENABLE=1 \
MPS_SERVER_TOPDOWN_DEPTH_BACKEND=unidepth_v2 \
/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/depth_spike/.venv/bin/python server/main.py
```

Outputs:
- telemetry logs in stdout
- persistence jsonl at `/tmp/mps_server_packets.jsonl` (default)
- first accepted packet is logged as `[telemetry] processed=1 ...`
- `SubmitPacketReply` can include `topdown_map` metadata + occupancy PNG

Server debug visualization mode:
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_SERVER_BIND_HOST=0.0.0.0 \
MPS_SERVER_PORT=19100 \
MPS_SERVER_DEBUG_VIS=1 \
MPS_SERVER_DEBUG_EVERY_N=1 \
/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/depth_spike/.venv/bin/python server/main.py
```

In debug mode the server shows:
- received image panels (`input`, optional `YOLO`, optional `ORB tracking`)
- telemetry overlay text
- center pose axes (X/Y/Z) rotated from `orb_tracking.camera_quaternion_wxyz`

## 4) Run Client (Terminal C)

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_SERVER_TRANSPORT=grpc \
MPS_SERVER_HOST=127.0.0.1 \
MPS_SERVER_PORT=19100 \
MPS_SERVER_TIMEOUT_S=0.4 \
MPS_ENABLE_GUI=1 \
MPS_ENABLE_USER_GUI=1 \
MPS_DATASET_ROOT=/home/tranduong/dev/thesis_prototype/dataset/dataset-corridor1_512_16 \
MPS_YOLO_MODEL_PATH=/home/tranduong/dev/thesis_prototype/models/yolo26n.pt \
MPS_REPLAY_SPEED=1.0 \
MPS_MAX_PENDING_IMU=4000 \
MPS_SLAM_HOST=127.0.0.1 \
MPS_SLAM_PORT=19090 \
python3 client/main.py
```

Smoke test:
```bash
MPS_ENABLE_GUI=0 MPS_MAX_FRAMES=40 python3 client/main.py
```

## Key Environment Variables

### ORB adapter
- `MPS_ORB_ROOT`
- `MPS_ORB_PROFILE` (`auto` / `tum_vi` / `euroc`)
- `MPS_ORB_SETTINGS_FILE`
- `MPS_ORB_VOC_FILE`
- `MPS_DEBUG_MODE=1` (enable Pangolin)
- `MPS_SLAM_PORT`

### Client -> Server bridge
- `MPS_SERVER_TRANSPORT` (`grpc` default in bridge)
- `MPS_SERVER_HOST`
- `MPS_SERVER_PORT`
- `MPS_SERVER_TIMEOUT_S`
- `MPS_ENABLE_GUI` (`1` enables local processing preview window)
- `MPS_ENABLE_USER_GUI` (`1` enables wearable user-view window; defaults to `MPS_ENABLE_GUI`)

### Server
- `MPS_SERVER_BIND_HOST`
- `MPS_SERVER_PORT`
- `MPS_SERVER_GRPC_WORKERS`
- `MPS_SERVER_INGRESS_QUEUE_SIZE`
- `MPS_SERVER_WORKER_QUEUE_SIZE`
- `MPS_SERVER_ENQUEUE_TIMEOUT_S`
- `MPS_SERVER_PERSIST_JSONL`
- `MPS_SERVER_DEBUG_VIS` (`1` to enable server debug window)
- `MPS_SERVER_DEBUG_EVERY_N` (draw every N packets in debug mode)
- `MPS_SERVER_TOPDOWN_ENABLE`
- `MPS_SERVER_TOPDOWN_DEPTH_BACKEND` (`unidepth_v2` or `mock`, default `unidepth_v2`)
- `MPS_SERVER_TOPDOWN_ALLOW_DEPTH_FALLBACK` (`1` to allow fallback to mock when UniDepth is unavailable)
- `MPS_SERVER_TOPDOWN_FISHEYE_K` (csv: `fx,fy,cx,cy`)
- `MPS_SERVER_TOPDOWN_FISHEYE_D` (csv: `k1,k2,k3,k4`)
- `MPS_SERVER_TOPDOWN_UNIDEPTH_MODEL_ID`
- `MPS_SERVER_TOPDOWN_UNIDEPTH_RESOLUTION_LEVEL`
- `MPS_SERVER_TOPDOWN_UNIDEPTH_CONFIDENCE_QUANTILE`
- `MPS_SERVER_TOPDOWN_UNIDEPTH_CONFIDENCE_MAX_ERROR`
- `MPS_SERVER_TOPDOWN_RESOLUTION_M`
- `MPS_SERVER_TOPDOWN_WIDTH`
- `MPS_SERVER_TOPDOWN_HEIGHT`
- `MPS_SERVER_TOPDOWN_UPDATE_EVERY_N`

## Troubleshooting

- `bind() failed` on adapter:
  - port already in use, kill old process or change `MPS_SLAM_PORT`.
  - quick cleanup: `pkill -f orb_native_adapter`

- gRPC send warnings in client (`will retry`):
  - start server first when possible; client now auto-reconnects on transient failures.
  - check `MPS_SERVER_HOST` / `MPS_SERVER_PORT`.
  - check env has `grpcio` + protobuf installed.

- UniDepth backend init fails (for example missing `einops`):
  - install UniDepth dependencies in the active venv:
    - `pip install -r /home/tranduong/dev/thesis_prototype/spatial_mapping_demo/UniDepth/requirements.txt`
  - run server from GPU-enabled env:
    - `/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/depth_spike/.venv/bin/python server/main.py`
  - if you need temporary fallback while dependencies are not ready:
    - `MPS_SERVER_TOPDOWN_ALLOW_DEPTH_FALLBACK=1`

- GUI display issues:
  - run from desktop shell with valid `DISPLAY`.
  - check: `echo $DISPLAY && xdpyinfo >/dev/null`
  - if server debug window freezes, update to latest code where GUI is pumped on main thread.
  - client can show 2 windows: `User View` and `Local Processing Preview`.

## Commit Note

Commit source/config only. Do not commit generated runtime artifacts:
- `integration/orb_native/bin`
- `integration/orb_native/build`
- `dependency/ORB_SLAM3_FIXED` (compiled external dependency)
- dataset/model binaries unless intentionally versioned
