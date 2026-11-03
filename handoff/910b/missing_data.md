# 910B 待补数据清单

> 每项都给命令、输出路径、验收标准。执行顺序即编号顺序。

## 1. 环境快照（必做）

```bash
bash handoff/910b/scripts/env_check.sh \
  --out handoff/910b/results/env_910b.json
```

输出：soc、device name、UB、vector core、CANN/torch/torch_npu/triton/vllm/vllm_ascend 版本。
验收：JSON 可解析，`vllm_ascend` 指向本分支。

## 2. 单算子 pytest（必做）

```bash
bash handoff/910b/scripts/run_op_tests.sh \
  --out handoff/910b/results/pytest_910b.log
```

覆盖：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p.py`
（23 项）+ `tests/ut/worker/v2/test_apply_top_k_top_p.py`（6 项）。
验收：**29 passed**。若 conftest 因环境失败，脚本会提示用 `--noconftest` 重跑。

## 3. 算子级完整 API 扫描（必做）

```bash
python handoff/910b/scripts/bench_op.py \
  --out handoff/910b/results/op_sweep_910b.json \
  --V 32000 151936 248320 --B 1 8 16 32 64 \
  --dtype fp32 bf16 --iters 20
```

口径：完整 API（`torch.topk`/`sort` + k/p 归一化 + 输出分配 + kernel）；PyTorch baseline =
`_apply_top_k_top_p_pytorch`；warmup 5 / iters 20。
验收：每个 case 有 p50/p90/speedup/kernel_only；B=16 至少 ≥1.0x。

## 4. 门控交叉点报告（必做，但不改代码）

```bash
python handoff/910b/scripts/gate_report.py \
  --in handoff/910b/results/op_sweep_910b.json \
  --out handoff/910b/results/gate_910b.md
```

输出：每个 (V,dtype) 的实测交叉 B、A5 现阈值、910B 建议阈值、差值。
验收：报告明确“实测交叉点”与“建议值”；**不要在本分支改 `_GATE_*`**，另开 PR。

## 5. 精度（必做）

```bash
python handoff/910b/scripts/precision_check.py \
  --out handoff/910b/results/precision_910b
```

覆盖 4×1000、8×4096、4×4097、8×32000、2×151936 × fp32/bf16。
验收：10/10 PASS；fp32 masked logits `torch.equal`；bf16 kept 差 ≤3、无 token 互换。

## 6. serving 压测（必做，baseline 与 latest 各一次）

```bash
MODEL=/path/to/Qwen3.8-27B-w8a8 \
  bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_baseline_910b --triton 0

MODEL=/path/to/Qwen3.8-27B-w8a8 \
  bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_latest_910b --triton 1 --force-gate
```

固定口径：300 prompts、rate 10、max-num-seqs 16、input 1024 / output 128、top_k 20 / top_p 0.95。
输出：`server.log`、`bench.log`、`result.json`。
验收：0 failed；两次同配置；记录 TPOT/P99 TPOT/P99 ITL/output tok/s；
latest 的 server.log 必须出现 `[sample/v2] Using Triton apply_top_k_top_p op`。

## 7. trace 采样链（必做）

```bash
# 启动时加 --profiler-config 或使用脚本的 profile 选项（见 serve_bench.sh 注释）
# 压测后导出：
python -c "from torch_npu.profiler.profiler import analyse; \
analyse('<trace_dir>', export_type=['csv'])"

python handoff/910b/scripts/extract_chain.py \
  --csv <trace_dir>/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --outdir handoff/910b/results/trace_910b \
  --sort-key TopkV2 --apply-key split_pass2 \
  --title trace_910b --first-label topk
```

验收：`SortAiCoreForDavid` 计数为 0；给出链 p50/p99、topk/pass1/pass2 分段、
3 条最慢链的 timestamp。

## 8. 重编译黑盒计数（可选）

```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_910b
rm -rf "$TRITON_CACHE_DIR"
# 跑一轮 serving 压测（或数据集），然后：
find "$TRITON_CACHE_DIR" -name "*.json" | wc -l
find "$TRITON_CACHE_DIR" -name "*.so" | wc -l
```

记录：服务配置下的编译产物数量、是否随请求的 k 变化增长（混合 k 时最多 9 档 TOPK）。
验收：给出“稳态编译份数”与“混合 k 时的最大份数”。

## 9. GPQA（可选，需数据集）

本地 A5 无 GPQA 数据集（只有 aime24/25/26、gsm8k）。若 910B 环境有 GPQA：

```bash
lm_eval --model vllm --model_args pretrained=<model>,tensor_parallel_size=1 \
  --tasks gpqa_main --batch_size auto
# 关闭/开启 enable_triton_apply_topk_topp 各跑一次，记录分数差
```

验收：两次分数差在噪声内（|Δ| ≤ 1–2 题）或给出解释。

## 10. 回填位置

- 原始结果 → `handoff/910b/results/`
- 结论摘要 → 本文件的“910B 结果”小节（新增），或 `results/MANIFEST.md`
- 若需更新 PR 文档，回到 `1102/docs` 同步（不要在本分支改 `docs/` 之外的代码）
