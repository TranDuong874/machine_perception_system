# Machine Perception System Handoff

## Project
Machine perception system for AR/VR:
- Client runs lightweight perception on-device or via simulated dataset replay.
- Server receives enriched packets and handles heavier processing.
- Current focus is a thesis MVP, not the full paper stack.

## Current Understanding
Pipeline:
- Client:
  - dataset replay
  - sensor sync
  - YOLO
  - SLAM adapter
  - build `EnrichedPerceptionPacket`
  - send to server over gRPC
- Server:
  - gRPC ingest
  - queue packets
  - preview/render received frames
  - optional `Depth Anything 3` depth inference on server side
  - currently no full memory/mapping backend yet, just ingestion + preview

## Client Status
Refactored to cleaner architecture:
- `client/main.py`: bootstrap only
- `client/config.py`: dataset/model paths, GUI flags, runtime env config
- `client/runtime/client_runtime.py`: coordinates client pipelines and shared state
- `client/pipeline/perception_pipeline.py`: sensor ingestion, local perception, packet enrichment, server send
- `client/ui/user_view_pipeline.py`: local preview and user-facing view from shared state
- `client/state/shared_client_state.py`: handoff between perception and UI
- `client/sync/sensor_packet_synchronizer.py`: groups frame + IMU into synchronized packets
- `client/transport/grpc_perception_client.py`: sends `EnrichedPerceptionPacket` to server
- `client/transport/grpc_assistant_client.py`: placeholder for future prompt/assistant RPCs

Naming changes already made:
- `LocalSensorPacket` -> `SynchronizedSensorPacket`
- enriched server-facing packet -> `EnrichedPerceptionPacket`
- `LocalProcessingIngress` -> `SensorPacketSynchronizer`
- `GrpcServerBridge` -> `GrpcPerceptionClient`
- orchestration moved into `PerceptionPipeline`

## Server Status
Server is simplified and reorganized into shallow folders:
- `server/main.py`: bootstrap/app wiring
- `server/api/grpc_ingest_servicer.py`
- `server/runtime/packet_ingestion_service.py`
- `server/runtime/packet_models.py`
- `server/rendering/preview_renderer.py`
- `server/settings/server_settings.py`
- `server/services/depth_estimation_service.py`

Current server behavior:
- receives `SubmitPacket`
- queues packet
- decodes frame
- draws YOLO boxes
- optional DA3 depth inference
- renders side-by-side RGB + depth preview
- GUI if display works, otherwise headless preview image

## Depth Anything 3
Integrated:
- vendor clone exists locally under `dependency/Depth-Anything-3`
- server uses `Depth Anything 3` through `DepthEstimationService`
- default model is now `DA3-BASE`

Current default in `config/server.env`:
- `MPS_SERVER_ENABLE_DEPTH=1`
- `MPS_SERVER_DEPTH_MODEL_ID=depth-anything/DA3-BASE`
- `MPS_SERVER_DEPTH_DEVICE=cuda`

Important:
- Main env `/home/tranduong/dev/thesis_prototype/.venv` was converted to CUDA PyTorch
- verified:
  - `torch 2.10.0+cu128`
  - `torch.cuda.is_available() == True`
- DA3 runtime installed in main `.venv`:
  - `depth_anything_3`
  - `omegaconf`
  - `addict`
  - `evo`

## GPU Findings
- Server really is using CUDA for DA3.
- Verified with:
  - startup logs: `device=cuda`
  - `nvidia-smi` compute process for the server Python process
- Low VRAM does not mean "not using GPU".
- CPU is still high because a lot of the pipeline is CPU:
  - JPEG encode/decode
  - OpenCV preprocessing/postprocessing
  - drawing/rendering
  - gRPC
  - client YOLO currently on CPU
  - SLAM adapter CPU work

## Run Commands
Server:
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
./run_server.sh
```

Direct:
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
/home/tranduong/dev/thesis_prototype/.venv/bin/python server/main.py
```

Client:
```bash
cd /home/tranduong/dev/thesis_prototype/machine_perception_system
/home/tranduong/dev/thesis_prototype/.venv/bin/python client/main.py
```

## Config Files
- `config/client.env`
- `config/server.env`

Env loader behavior:
- shell env overrides file
- inside the file, later lines win

## OpenCLIP / FAISS Demo
- separate demo exists under `open_clip_faiss_demo/`
- it should not be part of the main commit
- `.gitignore` excludes it

## What Was Tested
- client/server end-to-end smoke tests passed multiple times
- server headless preview artifact generated successfully
- DA3 CUDA path in the main `.venv` was tested
- server CUDA usage was confirmed with `nvidia-smi`

## Commit Hygiene
- `.gitignore` excludes:
  - `open_clip_faiss_demo/`
  - `dependency/Depth-Anything-3/`
- accidental staged vendor submodule entry was removed
- `project_note.md` is still untracked and unrelated

## Likely Next Steps
1. Benchmark `DA3-BASE` on the real server path and decide whether to keep `BASE` or go back to `SMALL`
2. Reduce CPU pressure:
   - disable preview in throughput mode
   - run depth every `N` frames
   - move client YOLO off CPU if possible
3. Continue server work beyond preview:
   - memory ingestion
   - spatial metadata
   - retrieval/chat path

## Seed Prompt For Next Conversation
```text
Continue from this machine_perception_system handoff:

- Client is refactored into ClientRuntime + PerceptionPipeline + UserViewPipeline + SharedClientState + SensorPacketSynchronizer + GrpcPerceptionClient.
- Server is simplified and reorganized into server/api, server/runtime, server/rendering, server/settings, server/services.
- Server currently does gRPC ingest + queue + RGB/depth preview.
- Depth Anything 3 is integrated via server/services/depth_estimation_service.py.
- Main env /home/tranduong/dev/thesis_prototype/.venv now has CUDA torch (2.10.0+cu128) and DA3 deps installed.
- Default server depth model is depth-anything/DA3-BASE in config/server.env.
- Run server with ./run_server.sh.
- open_clip_faiss_demo exists but should not be part of the main commit.
- Next task: [put your next task here].
```
