#!/bin/bash
# 910B 单算子正确性：23 e2e + 6 UT = 29 项。用法：
#   bash handoff/910b/scripts/run_op_tests.sh [--out results/pytest_910b.log]
set -euo pipefail

OUT="handoff/910b/results/pytest_910b.log"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$(dirname "$OUT")"

ROOT="$(git rev-parse --show-toplevel)"
PY="${PYTHON:-python}"

E2E="$ROOT/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p.py"
UT="$ROOT/tests/ut/worker/v2/test_apply_top_k_top_p.py"

set +e
"$PY" -m pytest -q --no-header "$E2E" "$UT" >"$OUT" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "ModuleNotFoundError\|ImportError while loading conftest" "$OUT"; then
  echo "[warn] conftest/依赖不兼容，改用 --noconftest 重跑" | tee -a "$OUT"
  "$PY" -m pytest --noconftest -q --no-header "$E2E" "$UT" >"$OUT" 2>&1
  rc=$?
fi
set -e
tail -3 "$OUT"
exit $rc
