#!/bin/bash
# 有界 serving 压测：start → bench → stop（trap 保证退出时停服务）。
# 用法：
#   MODEL=/path/to/model bash handoff/910b/scripts/serve_bench.sh \
#     --out handoff/910b/results/serve_latest_910b --triton 1 --force-gate
#
# 可选环境变量：
#   PORT(8000) VLLM_BIN(vllm) NUM_PROMPTS(300) RATE(10) MAX_NUM_SEQS(16)
#   INPUT_LEN(1024) OUTPUT_LEN(128) MAX_MODEL_LEN(32768) MAX_BATCHED_TOKENS(8192)
#   COMPILATION_CONFIG('{"cudagraph_mode":"FULL_DECODE_ONLY"}')
set -euo pipefail

OUT="handoff/910b/results/serve_run"
TRITON=0
GATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --triton) TRITON="$2"; shift 2 ;;
    --force-gate) GATE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${MODEL:?MODEL is required, e.g. MODEL=/path/to/Qwen3.8-27B-w8a8}"
PORT="${PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
RATE="${RATE:-10}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"

mkdir -p "$OUT"
CFG=$(python3 - "$TRITON" "$GATE" <<'PY'
import json, sys
print(json.dumps({
    "enable_triton_apply_topk_topp": sys.argv[1] == "1",
    "triton_topk_topp_disable_gate": sys.argv[2] == "1",
}))
PY
)
echo "[config] $CFG" | tee "$OUT/MANIFEST.md"

PID=""
cleanup() {
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "[stop] SIGTERM $PID"
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 2
    done
    kill -KILL "$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[start] $VLLM_BIN serve $MODEL (port $PORT, triton=$TRITON gate=$GATE)"
"$VLLM_BIN" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 --quantization ascend \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" --trust-remote-code \
  --enable-prefix-caching --gpu-memory-utilization 0.90 \
  --compilation-config "$COMPILATION_CONFIG" \
  --additional-config "$CFG" \
  >"$OUT/server.log" 2>&1 &
PID=$!

ready=0
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[error] server exited"; tail -60 "$OUT/server.log"; exit 1
  fi
  sleep 5
done
[ "$ready" = "1" ] || { echo "[error] server not ready in 1800s"; tail -60 "$OUT/server.log"; exit 1; }
echo "[ready] server up"

echo "[bench] $NUM_PROMPTS prompts, rate $RATE"
"$VLLM_BIN" bench serve --backend openai --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" --trust-remote-code \
  --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --request-rate "$RATE" \
  --top-k 20 --top-p 0.95 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 99 \
  >"$OUT/bench.log" 2>&1

python3 - "$OUT" <<'PY'
import json, re, sys
out = sys.argv[1]
txt = open(f"{out}/bench.log", errors="ignore").read()
def num(pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None
res = {
    "success": int(num(r"Successful requests:\s+(\d+)")) if num(r"Successful requests:\s+(\d+)") else None,
    "failed": int(num(r"Failed requests:\s+(\d+)")) if num(r"Failed requests:\s+(\d+)") else None,
    "duration": num(r"Benchmark duration \(s\):\s+([\d.]+)"),
    "out_toks": num(r"Output token throughput \(tok/s\):\s+([\d.]+)"),
    "tpot_mean": num(r"Mean TPOT \(ms\):\s+([\d.]+)"),
    "tpot_p99": num(r"P99 TPOT \(ms\):\s+([\d.]+)"),
    "itl_mean": num(r"Mean ITL \(ms\):\s+([\d.]+)"),
    "itl_p99": num(r"P99 ITL \(ms\):\s+([\d.]+)"),
    "ttft_mean": num(r"Mean TTFT \(ms\):\s+([\d.]+)"),
    "ttft_p99": num(r"P99 TTFT \(ms\):\s+([\d.]+)"),
    "source": f"{out}/bench.log",
}
json.dump(res, open(f"{out}/result.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PY

echo "[done] $OUT"
