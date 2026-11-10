#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ ${1:-sm120} != sm120 || $# -gt 1 ]]; then
    echo 'Usage: build.sh [sm120] (RTX PRO 5500 target only)' >&2
    exit 2
fi
docker build -f docker/flexkv-minimal/Dockerfile \
  --build-arg CUDA_ARCH=12.0 --build-arg DEMB_ARCHS=120 \
  --build-arg BUILD_JOBS="${BUILD_JOBS:-4}" \
  -t "${IMAGE:-hstu-flexkv-minimal:sm120}" .
