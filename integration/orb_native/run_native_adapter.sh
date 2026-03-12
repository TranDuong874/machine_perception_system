#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ORB_ROOT="${MPS_ORB_ROOT:-}"
if [[ -z "${ORB_ROOT}" ]]; then
  if [[ -f "${REPO_ROOT}/dependency/ORB_SLAM3_FIXED/lib/libORB_SLAM3.so" ]]; then
    ORB_ROOT="${REPO_ROOT}/dependency/ORB_SLAM3_FIXED"
  elif [[ -f "/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/ORB-SLAM3-STEREO-FIXED/lib/libORB_SLAM3.so" ]]; then
    ORB_ROOT="/home/tranduong/dev/thesis_prototype/spatial_mapping_demo/ORB-SLAM3-STEREO-FIXED"
  else
    echo "Could not locate fixed ORB root with lib/libORB_SLAM3.so." >&2
    echo "Expected: ${REPO_ROOT}/dependency/ORB_SLAM3_FIXED" >&2
    echo "Set MPS_ORB_ROOT explicitly." >&2
    exit 1
  fi
fi

if [[ ! -f "${ORB_ROOT}/lib/libORB_SLAM3.so" ]]; then
  echo "Missing ${ORB_ROOT}/lib/libORB_SLAM3.so" >&2
  echo "Your ORB root is not built yet. Build ORB-SLAM3 first (e.g. ./build.sh in ORB root)." >&2
  exit 1
fi
if [[ ! -f "${ORB_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so" ]]; then
  echo "Missing ${ORB_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so" >&2
  echo "Your ORB root is not built yet. Build ORB-SLAM3 first (e.g. ./build.sh in ORB root)." >&2
  exit 1
fi
if [[ ! -f "${ORB_ROOT}/Thirdparty/g2o/lib/libg2o.so" ]]; then
  echo "Missing ${ORB_ROOT}/Thirdparty/g2o/lib/libg2o.so" >&2
  echo "Your ORB root is not built yet. Build ORB-SLAM3 first (e.g. ./build.sh in ORB root)." >&2
  exit 1
fi

ADAPTER_BIN="${SCRIPT_DIR}/bin/orb_native_adapter"
if [[ ! -x "${ADAPTER_BIN}" ]]; then
  "${SCRIPT_DIR}/build_native_adapter.sh"
fi

VOC_FILE="${MPS_ORB_VOC_FILE:-}"
if [[ -z "${VOC_FILE}" ]]; then
  if [[ -f "${ORB_ROOT}/Vocabulary/ORBvoc.txt.tar.gz" && ! -f "${ORB_ROOT}/Vocabulary/ORBvoc.txt" ]]; then
    tar -xzf "${ORB_ROOT}/Vocabulary/ORBvoc.txt.tar.gz" -C "${ORB_ROOT}/Vocabulary"
  fi

  if [[ -f "${ORB_ROOT}/Vocabulary/ORBvoc.txt" ]]; then
    VOC_FILE="${ORB_ROOT}/Vocabulary/ORBvoc.txt"
  elif [[ -f "${REPO_ROOT}/models/orbvoc.txt" ]]; then
    VOC_FILE="${REPO_ROOT}/models/orbvoc.txt"
  elif [[ -f "${REPO_ROOT}/models/orbvoc.txt.bin" ]]; then
    # Fallback for newer forks that support binary vocabulary loading.
    VOC_FILE="${REPO_ROOT}/models/orbvoc.txt.bin"
  fi
fi

SETTINGS_FILE="${MPS_ORB_SETTINGS_FILE:-}"
ORB_PROFILE="${MPS_ORB_PROFILE:-auto}"
if [[ -z "${SETTINGS_FILE}" ]]; then
  TUM_VI_SETTINGS="${ORB_ROOT}/Examples/Monocular-Inertial/TUM-VI.yaml"
  EUROC_SETTINGS="${ORB_ROOT}/Examples/Monocular-Inertial/EuRoC.yaml"
  DATASET_HINT="${MPS_DATASET_ROOT:-}"
  DATASET_HINT_LOWER="$(echo "${DATASET_HINT}" | tr '[:upper:]' '[:lower:]')"

  case "${ORB_PROFILE}" in
    tum_vi)
      SETTINGS_FILE="${TUM_VI_SETTINGS}"
      ;;
    euroc)
      SETTINGS_FILE="${EUROC_SETTINGS}"
      ;;
    auto)
      # Corridor / room / TUM-style datasets are usually TUM-VI in this project.
      if [[ "${DATASET_HINT_LOWER}" == *"corridor"* || "${DATASET_HINT_LOWER}" == *"room"* || "${DATASET_HINT_LOWER}" == *"tum"* || "${DATASET_HINT_LOWER}" == *"512_16"* ]]; then
        SETTINGS_FILE="${TUM_VI_SETTINGS}"
      elif [[ "${DATASET_HINT_LOWER}" == *"euroc"* || "${DATASET_HINT_LOWER}" == *"mh_"* || "${DATASET_HINT_LOWER}" == *"v1_"* || "${DATASET_HINT_LOWER}" == *"v2_"* ]]; then
        SETTINGS_FILE="${EUROC_SETTINGS}"
      elif [[ -f "${TUM_VI_SETTINGS}" ]]; then
        SETTINGS_FILE="${TUM_VI_SETTINGS}"
      else
        SETTINGS_FILE="${EUROC_SETTINGS}"
      fi
      ;;
    *)
      echo "Invalid MPS_ORB_PROFILE='${ORB_PROFILE}'. Use: auto | tum_vi | euroc" >&2
      exit 1
      ;;
  esac
fi
HOST="${MPS_SLAM_BIND_HOST:-0.0.0.0}"
PORT="${MPS_SLAM_PORT:-19090}"
USE_VIEWER="${MPS_ORB_USE_VIEWER:-0}"
if [[ "${MPS_DEBUG_MODE:-0}" == "1" ]]; then
  USE_VIEWER="1"
fi

echo "Using ORB settings: ${SETTINGS_FILE}"
echo "Using ORB root: ${ORB_ROOT}"

if [[ ! -f "${VOC_FILE}" ]]; then
  echo "Vocabulary file not found: ${VOC_FILE}" >&2
  exit 1
fi
if [[ ! -f "${SETTINGS_FILE}" ]]; then
  echo "Settings file not found: ${SETTINGS_FILE}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"

ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --voc-file "${VOC_FILE}"
  --settings-file "${SETTINGS_FILE}"
)
if [[ "${USE_VIEWER}" == "1" ]]; then
  ARGS+=(--use-viewer)
fi

exec "${ADAPTER_BIN}" "${ARGS[@]}"
