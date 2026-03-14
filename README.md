# Machine Perception System (POC)

## Goal
End-to-end POC pipeline:

`Input -> SensorPacketSynchronizer -> YOLO + ORB -> PerceptionPipeline -> server`

Runtime defaults are loaded automatically from:
- `config/client.env` if present, otherwise `config/client.env.example`
- `config/server.env` if present, otherwise `config/server.env.example`

Shell environment variables still override values from those files.

Local secret/config pattern:
- keep real local values in untracked `config/client.env` and `config/server.env`
- commit only `config/client.env.example` and `config/server.env.example`
- export secrets like `GROQ_API_KEY` from your shell instead of committing them

## Helper Scripts

- `start_stack.sh`: start ORB adapter, server, and client together
- `scripts/run_server.sh`: run only the server
- `scripts/query_scene.sh`: query the VLM HTTP API
- `scripts/query_memory.sh`: query the memory index directly
- `scripts/run_mcp_server.sh`: start the MCP stdio server

## Current Architecture

### Client
- `client/main.py` bootstraps only.
- `client/config.py` resolves dataset/model paths, GUI flags, and runtime env config.
- `client/runtime/client_runtime.py` coordinates the client pipelines and shared state.
- `client/pipeline/perception_pipeline.py` owns sensor ingestion, local perception, packet enrichment, and server send.
- `client/ui/user_view_pipeline.py` renders the local preview and user-facing view from shared client state.
- `client/state/shared_client_state.py` is the handoff point between perception and UI.
- `client/sync/sensor_packet_synchronizer.py` groups raw frame and IMU samples into synchronized packets.
- `client/transport/grpc_perception_client.py` sends `EnrichedPerceptionPacket` to the server over gRPC.
- `client/transport/grpc_assistant_client.py` is a placeholder for future prompt/assistant RPCs.
- `client/user_interface/user_view_renderer.py` renders wearable-facing user view from local frames.
- `SlamService` talks to local native ORB adapter (TCP, local).
- Local processing preview window remains available for debugging.

### Server
- `server/main.py` orchestrates only.
- gRPC ingress (`SubmitPacket`) receives packets.
- Minimal server modules:
  - `server/runtime/packet_models.py` defines the packet envelope and ingestion metrics
  - `server/runtime/packet_ingestion_service.py` owns the ingress queue and worker thread
  - `server/api/grpc_ingest_servicer.py` handles `SubmitPacket`
  - `server/rendering/preview_renderer.py` renders received frames with YOLO boxes and optional depth preview
  - `server/settings/server_settings.py` loads runtime config
  - `server/services/depth_estimation_service.py` wraps optional `Depth Anything 3` inference
  - `server/main.py` boots the gRPC server and wires everything together

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

./scripts/run_server.sh
```

Outputs:
- ingestion logs in stdout
- first accepted packet is logged as `[ingestion] processed=1 ...`
- `SubmitPacketReply` is currently used as an ingestion acknowledgement only
- preview window or headless preview image from `server/rendering/preview_renderer.py`
- `scripts/run_server.sh` reads `config/server.env` and launches the server from the main project `.venv`

Depth preview on the 6GB GPU:
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

MPS_SERVER_ENABLE_GUI=0 \
MPS_SERVER_HEADLESS_PREVIEW_PATH=/tmp/mps_server_depth_preview.jpg \
./scripts/run_server.sh
```

Notes:
- first depth run downloads `depth-anything/DA3-BASE` from Hugging Face
- the headless preview image becomes a side-by-side `RGB + depth` render
- `/home/tranduong/dev/thesis_prototype/.venv/bin/python` now reports `torch.cuda.is_available() == True`

## 4) Run Client (Terminal C)

```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system

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
- `MPS_SERVER_ENQUEUE_TIMEOUT_S`
- `MPS_SERVER_LOG_EVERY_N`
- `MPS_SERVER_ENABLE_GUI`
- `MPS_SERVER_HEADLESS_PREVIEW_PATH`
- `MPS_SERVER_ENABLE_DEPTH`
- `MPS_SERVER_DEPTH_MODEL_ID`
- `MPS_SERVER_DEPTH_DEVICE`
- `MPS_SERVER_DEPTH_PROCESS_RES`

Notes:
- future processing logic belongs in `server/runtime/packet_ingestion_service.py`, inside `PacketIngestionService.process_packet()`
- depth preview uses `depth-anything/DA3-BASE` by default and now runs from the main project `.venv`

## Troubleshooting

- `bind() failed` on adapter:
  - port already in use, kill old process or change `MPS_SLAM_PORT`.
  - quick cleanup: `pkill -f orb_native_adapter`

- gRPC send warnings in client (`will retry`):
  - start server first when possible; client now auto-reconnects on transient failures.
  - check `MPS_SERVER_HOST` / `MPS_SERVER_PORT`.
  - check env has `grpcio` + protobuf installed.

## Commit Note

Commit source/config only. Do not commit generated runtime artifacts:
- `integration/orb_native/bin`
- `integration/orb_native/build`
- `dependency/ORB_SLAM3_FIXED` (compiled external dependency)
- dataset/model binaries unless intentionally versioned
