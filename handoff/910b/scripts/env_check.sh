#!/bin/bash
# 910B 环境快照：设备/UB/vector core/版本。用法：
#   bash handoff/910b/scripts/env_check.sh [--out results/env_910b.json]
set -euo pipefail

# 尽量自动加载 CANN 环境；若容器已有 set_env.sh 之外的配置，请先手动 source。
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit/*; do
    if [ -f "$d/set_env.sh" ]; then source "$d/set_env.sh"; break; fi
  done
fi
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  echo "[warn] ASCEND_HOME_PATH 未设置：请先 source CANN 的 set_env.sh，否则 torch_npu 可能加载失败" >&2
fi

PY="${PYTHON:-python}"
OUT="handoff/910b/results/env_910b.json"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$(dirname "$OUT")"

echo "== npu-smi =="
npu-smi info 2>/dev/null | head -40 || echo "(npu-smi unavailable)"

"$PY" - "$OUT" <<'PY'
import importlib.metadata as md
import json
import os
import platform
import sys

out_path = sys.argv[1]
info = {
    "machine": platform.machine(),
    "python": platform.python_version(),
    "cann": {
        k: os.environ.get(k)
        for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH")
    },
}

for pkg in ("torch", "torch-npu", "vllm", "triton-ascend", "triton", "vllm-ascend"):
    try:
        info[pkg] = md.version(pkg)
    except Exception:
        info[pkg] = None

import torch
import torch_npu
info["torch"] = torch.__version__
info["torch_npu"] = torch_npu.__version__
try:
    info["soc_version"] = int(torch_npu.npu.get_soc_version())
except Exception as exc:
    info["soc_version_error"] = str(exc)
try:
    info["device_name"] = torch_npu.npu.get_device_name(0)
except Exception as exc:
    info["device_name_error"] = str(exc)

import vllm
info["vllm_runtime"] = vllm.__version__
info["vllm_file"] = vllm.__file__

import vllm_ascend
info["vllm_ascend_file"] = vllm_ascend.__file__
info["vllm_ascend_version"] = getattr(vllm_ascend, "__version__", None)

try:
    from vllm_ascend.ops.triton.triton_utils import (
        get_ub_size_bytes,
        get_vectorcore_num,
        init_device_properties_triton,
    )
    init_device_properties_triton()
    info["vector_core"] = get_vectorcore_num()
    info["ub_size_bytes"] = get_ub_size_bytes()
except Exception as exc:
    info["device_props_error"] = str(exc)

with open(out_path, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
print(f"\nwritten: {out_path}")
PY
