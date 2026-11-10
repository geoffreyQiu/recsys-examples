#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
base=${BASE_IMAGE:-hstu-flexkv-minimal:sm120}
image=${IMAGE:-hstu-flexkv-minimal:sm120-allocfix}
if [[ $base == "$image" ]]; then
    echo 'Use a separate output image tag to preserve the original image.' >&2
    exit 2
fi
# Require the successfully built local image; do not fetch a replacement base.
docker image inspect "$base" >/dev/null
docker build --pull=false -f docker/flexkv-minimal/Dockerfile.cache-allocator \
    --build-arg BASE_IMAGE="$base" --build-arg BUILD_JOBS="${BUILD_JOBS:-4}" \
    -t "$image" .
