# Triton 算子评审规范（910B 续测版）

> 对齐 PR head `24d76041c`（2026-11-03）。
> 当前代码为 **V2-only**：V1 sampler 未修改；算子位于 `vllm_ascend/ops/triton/sample/`；
> fused softmax 路径已删除，只返回 masked logits。
> 2026-11-02 的 A5 历史数据（对应旧 head `06fd67e70`、含 V1/fused）仅作参考，
> 见 `/workspace/user_data/1102/docs/`；910B 数据必须在本分支重测。

## 0. 代码范围与路径

| 项 | 路径 |
| --- | --- |
| 算子主体 | `vllm_ascend/ops/triton/sample/apply_top_k_top_p.py` |
| 算子文档 | `vllm_ascend/ops/triton/docs/apply_top_k_top_p.md` |
| V2 接线 | `vllm_ascend/worker/v2/sample/apply_top_k_top_p.py`（保留 upstream `_apply_top_k_top_p_pytorch`，条件满足时路由到算子） |
| V2 patch | `vllm_ascend/patch/worker/patch_triton.py`（`states.SamplingStates.get_top_k_top_p` + `topk_topp_sampler.apply_top_k_top_p_triton`） |
| V1 | **未修改**（仍是 stock 路径） |
| 单算子测试 | 23 e2e（`tests/e2e/.../triton/test_apply_top_k_top_p.py`）+ 6 UT（`tests/ut/worker/v2/test_apply_top_k_top_p.py`）= **29** |
| 配置开关 | `enable_triton_apply_topk_topp` / `triton_topk_topp_disable_gate` / `triton_topk_topp_disable_no_sort`（`additional_config`，均默认 false） |

## 1. 语义

- `top_p_mode="full_vocab"`（默认，V2 接线）：与 `_apply_top_k_top_p_pytorch` 一致，
  全词表 softmax 后算 cutoff；fp32 `torch.equal`，bf16/fp16 仅边界舍入差异。
- `top_p_mode="topk_renorm"`：CANN `torch_npu.npu_top_k_top_p` 语义，逐位对拍；
  与 V2 默认路径不等价，**仅用于验证**。
- `k/p` 保持 per-request `[B]` 张量，不做 `.item()`；`k>=V`、`p>=1.0` 按 vLLM 约定视为关闭；
  `k<=0` 收敛为“只保留 max”。

## 2. 调度

- `full_vocab` + `k<V`：`torch.topk(width)` + 两段分核：
  - pass1（`split_pass1`）：每个 `(row, chunk)` 求部分分母；
  - pass2（`split_pass2`）：按 chunk 顺序归约并写 masked logits；
  - `C = max(1, cores // B)`；scratch `[B,C,BLOCK]` fp32 + `[B,C]` int32。
- `k>=V` / `topk_renorm` / `no_sort=false`：升序 `torch.sort` + 统一 tiled kernel。
- `BLOCK` 由运行时 UB 推导（A2/A3 192 KB → 2048；A5 248 KB → 4096），不手工调。

## 3. V2 接线与去 host 同步

- `apply_top_k_top_p_npu` 先消费 `_K_MAX_HINT`，再判断开关 / `HAS_TRITON` / 310P / B-V 门控；
  不满足时调用 `_apply_top_k_top_p_pytorch`，并 `warning_once`。
- `SamplingStates.get_top_k_top_p` 被 patch 为 `get_top_k_top_p_with_hint`：
  从 `top_k.np`（UVA CPU 镜像）读 `max(k)` 并发 hint；兼容 vLLM main（3 参数/3 返回值）
  与 0.32.0（2 参数/2 返回值）。
- V2 使用非融合 masked logits；`gumbel_sample` 在外部完成。

## 4. 测试矩阵

### 4.1 单算子正确性（必做）

```bash
bash handoff/910b/scripts/run_op_tests.sh \
  --out handoff/910b/results/pytest_910b.log
```

- 23 e2e + 6 UT = **29 passed**。
- e2e 覆盖：`topk_renorm` 对 CANN；`full_vocab` 对 PyTorch；ties；bf16/fp16 低精度；
  退化行；标量/None；非法 `top_p_mode`。
- UT 覆盖：开关/门控/310P/HAS_TRITON/None/no-sort 回退与 `k_max` hint。

### 4.2 算子级性能（必做）

```bash
python handoff/910b/scripts/bench_op.py \
  --out handoff/910b/results/op_sweep_910b.json \
  --V 32000 151936 248320 --B 1 8 16 32 64 \
  --dtype fp32 bf16 --iters 20
```

- 完整 API：`torch.topk`/`sort` + k/p 归一化 + 输出分配 + kernel；
- baseline = `_apply_top_k_top_p_pytorch`（worker 内）；
- 输出 p50/p90/speedup/kernel_only；B=16 至少 ≥1.0x。

### 4.3 门控交叉点（必做，不改代码）

```bash
python handoff/910b/scripts/gate_report.py \
  --in handoff/910b/results/op_sweep_910b.json \
  --out handoff/910b/results/gate_910b.md
```

- A5 现阈值：`V>=65536 → B>=32`；`V<65536 → B>=8`；
- 910B 只记录实测交叉点与建议值，阈值调整另开 PR。

### 4.4 精度（必做）

```bash
python handoff/910b/scripts/precision_check.py \
  --out handoff/910b/results/precision_910b
```

- 4×1000、8×4096、4×4097、8×32000、2×151936 × fp32/bf16 = 10 组；
- fp32 `torch.equal`；bf16 交集值相等、kept 差 ≤3；验收 10/10 PASS。

### 4.5 serving（必做）

```bash
MODEL=/path/to/model bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_baseline_910b --triton 0
MODEL=/path/to/model bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_latest_910b --triton 1 --force-gate
```

- 300 prompts / rate 10 / max-num-seqs 16 / input 1024 / output 128 / top_k 20 / top_p 0.95；
- baseline 与 latest 同配置；0 failed；
- latest server log 必须出现 `[sample/v2] Using Triton apply_top_k_top_p op`；
- 记录 TPOT/P99 TPOT/P99 ITL/output tok/s。

### 4.6 trace（必做）

```bash
python handoff/910b/scripts/extract_chain.py \
  --csv <trace>/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --outdir handoff/910b/results/trace_910b \
  --sort-key TopkV2 --apply-key split_pass2 \
  --title trace_910b --first-label topk
```

验收：`SortAiCoreForDavid` 计数 0；链 p50/p99；topk/pass1/pass2 分段；3 条最慢链 timestamp。

## 5. 910B 与 A5 的差异

| 项 | A5（950PR） | 910B（A2/A3） |
| --- | --- | --- |
| UB | 248 KB | 192 KB |
| BLOCK | 4096 | 2048 |
| vector core | 56 | 待测 |
| 门控 | V≥65536→B≥32；V<65536→B≥8 | **必须重测** |
| 310P | 拒绝 | 不适用 |

## 6. 评审检查项

- [ ] 算子只在 `ops/triton/sample/`，worker 不承载 kernel；
- [ ] V1 未被修改；
- [ ] 无 fused 相关代码/测试/文档残留；
- [ ] 开关全部 `additional_config`，无新增环境变量；
- [ ] 29 项 pytest 通过；
- [ ] 910B 扫描、门控、精度、serving、trace 数据齐备；
- [ ] `results/*/MANIFEST.md` 记录 commit、vLLM pin、CANN/torch/triton 版本与命令。

## 7. 历史 A5 数据（只读参考）

- PR 回检：`/workspace/user_data/1102/docs/PR17211_AuroraEmiya回检.md`
- A5 规范：`/workspace/user_data/1102/docs/triton开发规范.md`
- A5 结果：`/workspace/user_data/1103/results/summary.md`
- 真实输入回放：`/workspace/user_data/1102/docs/logs/ae2_prefill_decode_replay.{log,json}`
