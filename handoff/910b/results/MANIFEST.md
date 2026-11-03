# 910B / A2-A3 续测回填结果（MANIFEST）

- 日期：2026-11-03
- 交接分支：`handoff/910b-a2a3` tip `d2d3650`
- 被测代码：PR head `24d76041c`（V2-only + additional_config）
- vLLM：main pin `544f182d9f0c9bfe8bd7645525951d5e3692a542`（本地 editable，`VLLM_TARGET_DEVICE=empty`）
- vllm-ascend：本分支源码（`PYTHONPATH` 指向工作树；`vllm_ascend_file` 见 `env_910b.json`）
- 设备：Ascend 910B3（A2），`soc_version=223`，vector core **40**，UB **196608 B（192 KB）**
- CANN：9.0.0（`/usr/local/Ascend/cann-9.0.0`）
- Python 3.12.13；torch 2.10.0+cpu
- torch_npu：算子级结论为 **2.10.0.post4**（`env_910b.json` 快照版本）；serving 尝试阶段升到 requirements 指定的 **2.10.0.post7**（见第 6 节）
- triton-ascend：**3.6.0.post2**（requirements 指定版本；3.2.2 实测存在重复异步下发 vector core 超时，见第 8 节）
- 模型：`Qwen3.8-27B-w8a8`，`text_config.vocab_size=248320`，TP=1（仅算子级/精度；serving 未跑通，见第 6 节）

## 1. 任务完成情况

| 项 | 状态 | 产物 |
| --- | --- | --- |
| 1 环境快照 | ✅ | `env_910b.json` |
| 2 单算子 pytest（29 项） | ✅ 29 passed | `pytest_910b.log` |
| 3 算子级完整 API 扫描 | ✅ 30 cases | `op_sweep_910b.json` |
| 4 门控交叉点报告（不改代码） | ✅ | `gate_910b.md` |
| 5 精度（10 组） | ✅ 10/10 PASS | `precision_910b.log` / `precision_910b.json` |
| 6 serving（baseline / latest） | ❌ 环境阻塞 | `serve_baseline_910b/`（0/300，含根因） |
| 7 trace 采样链 | ◐ 算子级 trace | `trace_910b/`（kernel 链 + sort=0；非 serving 链） |
| 8 重编译黑盒计数（可选） | ⬜ 未做 | — |
| 9 GPQA（可选） | ⬜ 无数据集 | — |

## 2. 单算子 pytest

```text
tests/e2e/.../triton/test_apply_top_k_top_p.py（23 项）
tests/ut/worker/v2/test_apply_top_k_top_p.py（6 项）
29 passed, 19 warnings in 91.73s
```

命令：

```bash
PYTHON=/workspace/1102/venv910b/bin/python \
PYTHONPATH=/workspace/1102/handoff_repo \
bash handoff/910b/scripts/run_op_tests.sh --out handoff/910b/results/pytest_910b.log
```

## 3. 算子级扫描（完整 API，warmup 5 / iters 20，p50 ms）

数据源：`op_sweep_910b.json`（含 p90、speedup、kernel_only；`kernel_only` 为 sort+tiled 路径）。

| V | dtype | B=1 | B=8 | B=16 | B=32 | B=64 |
| --- | --- | --- | --- | --- | --- | --- |
| 32000 | fp32 | PyTorch 0.722 / Triton 0.651 | 0.653 / 0.562 | 0.653 / 0.618 | 0.678 / 0.615 | 0.671 / 0.592 |
| 32000 | bf16 | 0.732 / 0.708 | 0.665 / 0.555 | 0.657 / 0.597 | 0.659 / 0.607 | 0.656 / 0.595 |
| 151936 | fp32 | 0.708 / 0.668 | 0.752 / 0.627 | 0.877 / 0.620 | 1.167 / 0.608 | 2.228 / 0.956 |
| 151936 | bf16 | 0.698 / 0.664 | 0.760 / 0.618 | 0.891 / 0.611 | 1.181 / 0.612 | 2.143 / 0.965 |
| 248320 | fp32 | 0.981 / 0.652 | 1.100 / 0.613 | 1.372 / 0.642 | 1.837 / 0.820 | 3.642 / 1.379 |
| 248320 | bf16 | 0.993 / 0.657 | 1.111 / 0.609 | 1.377 / 0.584 | 1.808 / 0.841 | 3.548 / 1.399 |

speedup（PyTorch/Triton，p50）：

| V | dtype | B=1 | B=8 | B=16 | B=32 | B=64 |
| --- | --- | --- | --- | --- | --- | --- |
| 32000 | fp32 | 1.11x | 1.16x | 1.06x | 1.10x | 1.13x |
| 32000 | bf16 | 1.03x | 1.20x | 1.10x | 1.08x | 1.10x |
| 151936 | fp32 | 1.06x | 1.20x | 1.42x | 1.92x | 2.33x |
| 151936 | bf16 | 1.05x | 1.23x | 1.46x | 1.93x | 2.22x |
| 248320 | fp32 | 1.50x | 1.79x | 2.14x | 2.24x | 2.64x |
| 248320 | bf16 | 1.51x | 1.82x | 2.36x | 2.15x | 2.54x |

验收：B=16 全部 ≥1.0x（32000: 1.06/1.10；151936: 1.42/1.46；248320: 2.14/2.36）。

## 4. 门控交叉点（`gate_910b.md`）

| V | dtype | A5 现阈值 | 910B 实测交叉点(≥1.0x) | 建议阈值(≥1.05x) |
| --- | --- | --- | --- | --- |
| 32000 | fp32 / bf16 | 8 / 8 | 1 / 1 | 1 / 8 |
| 151936 | fp32 / bf16 | 32 / 32 | 1 / 1 | 1 / 1 |
| 248320 | fp32 / bf16 | 32 / 32 | 1 / 1 | 1 / 1 |

结论：910B 上实测交叉点为 B=1；A5 阈值（大 V B≥32、小 V B≥8）在 910B 属于保守下界，仍可保留。**本分支未改 `_GATE_*`**。

## 5. 精度（`precision_910b.log` / `.json`）

```text
4x1000    fp32/bf16 PASS
8x4096    fp32/bf16 PASS
4x4097    fp32 逐位相等 / bf16 kept_diff=1 PASS
8x32000   fp32/bf16 PASS
2x151936  fp32/bf16 PASS
summary: 10/10 PASS
```

fp32 全部 `torch.equal`；bf16 交集位置值完全相等，仅 4×4097 有 1 个边界 token（≤3 容差）。

## 6. serving：CANN 9.0.0 阻塞（0/300）

baseline（`enable_triton_apply_topk_topp=false`，300 prompts / rate 10 / max-num-seqs 16 / input 1024 / output 128）：

```text
Successful requests: 0
Failed requests:     300
失败点：aclnnScatterPaKvCache error code 561000
"Op ScatterPaKvCache has 2 uncontiguous input: keyCacheRef, valueCacheRef,
 but set 0 auto conversion, may not set AutoContiguous() in OpDef."
```

- `ScatterPaKvCache` 是 KV cache 写入路径，与 top-k/top-p 开关无关；切到 `VLLM_KV_CACHE_LAYOUT=LBHNC` 后改为调用 `_C_ascend.npu_scatter_pa_kv_cache`，但当前可用的旧版扩展不含该算子（`AttributeError`）。
- 本机 CANN 9.0.0 无法编译本分支的 `vllm_ascend_C`：`torch_npu` 的 ACL 头（post4/post7）与 CANN 9.0 头冲突于 `aclmdlRITask`（该 API 属 CANN 9.1）；同时 `sparse_flash_mla` / `quant_lightning_indexer_v2` 的 host 代码引用 CANN 9.1 新增 `OP_LOGE_FOR_INVALID_*` 宏。
- 结论：**vllm-ascend 当前 main 代码需要 CANN 9.1 才能在本机跑 serving**；与 Triton top-k/top-p 算子本身无关。证据：`serve_baseline_910b/{server.log,bench.log,result.json}`。
- eager（`--enforce-eager`）与 `FULL_DECODE_ONLY` 两种模式都在同一 `ScatterPaKvCache` 处失败；graph capture 前的 `qknorm_rope_fusion` / `_triton_rope` 问题已记录在 server.log 中（见第 8 节）。

## 7. trace：算子级采样链（`trace_910b/`）

不做 serving（被第 6 节阻塞），改用 `torch_npu.profiler` 直接采集 `apply_top_k_top_p` 的
no-sort 路径（B=16、V=151936、k=20、p=0.95、fp32，4 个 active step）：

| 指标 | 值 |
| --- | --- |
| `aclnnTopk_TopkV2AiCore_TopKV2` | p50 133.1 µs |
| `_apply_topk_topp_split_pass1_kernel` | p50 34.3 µs |
| `_apply_topk_topp_split_pass2_kernel` | p50 27.0 µs |
| 链总（topk→pass2 结束）p50 / p99 | 314.2 / 347.0 µs |
| `SortAiCoreForDavid` 计数 | **0** |

最慢 4 条链的 timestamp 见 `trace_910b/kernel_timeline.md`。产物：`kernel_details.csv`、`kernel_timeline.md`、`sampling_chain.csv`。

## 8. 环境差异与实测问题

1. **triton-ascend 3.2.2（A5 同版本）在 910B3 上不可用于本算子基准**：同一 `apply_top_k_top_p_with_sorted` 连续两次异步下发时固定触发 vector core 超时（507034）；单独逐步同步调用正常。切到 requirements 指定的 **3.6.0.post2** 后消失（`async2/3/5` 全部通过）。算子级数据与 pytest 均在 3.6.0.post2 下复测。
2. **torch_npu 版本**：起初用 2.10.0.post4（与 A5 一致）；为尝试构建扩展升到 requirements 的 post7。post7 仍不能解决 CANN 9.0 的 `aclmdlRITask` 头冲突。
3. **自定义算子包**：本分支全量 AscendC 构建在 CANN 9.0 上失败于 `sparse_flash_mla` / `quant_lightning_indexer_v2`（CANN 9.1 宏缺失），其余算子可编译；serving 尝试时复用了 0819 环境中同 CANN/同设备的 `custom_transformer` vendor 包（仅 `add_rms_norm_bias` 等旧算子）。
4. **本地临时改动（未提交）**：为定位 serving 失败曾临时关闭 Triton RoPE（`rotary_embedding.py`）与跳过 ACLNN 构建（`setup.py`），提交前已全部还原；工作树只包含本 MANIFEST 及 `results/` 下产物。

## 9. 复现命令

```bash
export PYTHONPATH="/workspace/1102/handoff_repo:$PYTHONPATH"
export PYTHON=/workspace/1102/venv910b/bin/python

bash handoff/910b/scripts/env_check.sh --out handoff/910b/results/env_910b.json
bash handoff/910b/scripts/run_op_tests.sh --out handoff/910b/results/pytest_910b.log
$PYTHON handoff/910b/scripts/bench_op.py \
  --out handoff/910b/results/op_sweep_910b.json \
  --V 32000 151936 248320 --B 1 8 16 32 64 --dtype fp32 bf16 --iters 20
$PYTHON handoff/910b/scripts/gate_report.py \
  --in handoff/910b/results/op_sweep_910b.json --out handoff/910b/results/gate_910b.md
$PYTHON handoff/910b/scripts/precision_check.py --out handoff/910b/results/precision_910b
$PYTHON handoff/910b/scripts/extract_chain.py \
  --csv handoff/910b/results/trace_910b/kernel_details.csv \
  --outdir handoff/910b/results/trace_910b \
  --sort-key TopkV2 --apply-key split_pass2 --title trace_910b --first-label topk
```
