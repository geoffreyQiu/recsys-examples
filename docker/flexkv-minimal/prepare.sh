#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
git merge-base --is-ancestor d03198c35166e8dae907b6bc453f2fa1112700b0 HEAD || {
  echo 'Expected a descendant of upstream d03198c35166e8dae907b6bc453f2fa1112700b0.' >&2
  exit 1
}
git submodule update --init --depth 1 third_party/FBGEMM third_party/FlexKV
git -C third_party/FBGEMM submodule update --init --depth 1 external/cutlass
git -C third_party/FlexKV submodule update --init --depth 1 third_party/xxHash third_party/spdlog
