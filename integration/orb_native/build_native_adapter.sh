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
  echo "libORB_SLAM3.so not found under ${ORB_ROOT}/lib" >&2
  exit 1
fi
if [[ ! -f "${ORB_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so" ]]; then
  echo "libDBoW2.so not found under ${ORB_ROOT}/Thirdparty/DBoW2/lib" >&2
  exit 1
fi
if [[ ! -f "${ORB_ROOT}/Thirdparty/g2o/lib/libg2o.so" ]]; then
  echo "libg2o.so not found under ${ORB_ROOT}/Thirdparty/g2o/lib" >&2
  exit 1
fi

OUTPUT_BIN="${SCRIPT_DIR}/bin/orb_native_adapter"
SOURCE_FILE="${SCRIPT_DIR}/orb_native_adapter.cpp"

mkdir -p "${SCRIPT_DIR}/bin" "${SCRIPT_DIR}/build"

OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
OPENCV_LIBS="$(pkg-config --libs opencv4)"
PANGOLIN_LIBS=(
  -lpango_glgeometry
  -lpango_plot
  -lpango_scene
  -lpango_tools
  -lpango_video
  -lpango_geometry
  -ltinyobj
  -lpango_display
  -lpango_vars
  -lpango_windowing
  -lpango_opengl
  -lEGL
  -lOpenGL
  -lepoxy
  -lpango_image
  -lpango_packetstream
  -lpango_core
  -lrt
)

set -x
g++ \
  -std=c++14 -O2 -Wall -Wextra \
  -DCOMPILEDWITHC14 -DHAVE_EIGEN -DHAVE_EPOXY -D_LINUX_ \
  ${OPENCV_CFLAGS} \
  -I"${ORB_ROOT}" \
  -I"${ORB_ROOT}/include" \
  -I"${ORB_ROOT}/include/CameraModels" \
  -I"${ORB_ROOT}/Thirdparty/Sophus" \
  -I/usr/include/eigen3 \
  "${SOURCE_FILE}" \
  -L"${ORB_ROOT}/lib" \
  -L"${ORB_ROOT}/Thirdparty/DBoW2/lib" \
  -L"${ORB_ROOT}/Thirdparty/g2o/lib" \
  -Wl,-rpath,"${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib:/usr/local/lib" \
  ${OPENCV_LIBS} \
  -lORB_SLAM3 -lDBoW2 -lg2o -ljsoncpp -lboost_serialization -lcrypto \
  "${PANGOLIN_LIBS[@]}" \
  -lpthread \
  -o "${OUTPUT_BIN}"
set +x

echo "Built ${OUTPUT_BIN}"
