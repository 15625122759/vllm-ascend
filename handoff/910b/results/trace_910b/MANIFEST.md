# MANIFEST — trace_910b

- date: 2026-11-03
- git commit: `d2d3650`（handoff/910b-a2a3；被测代码 `24d76041c`）
- vllm pin: `544f182d9f0c9bfe8bd7645525951d5e3692a542`
- CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post7 / triton-ascend 3.6.0.post2
- soc 223 / vector_core 40 / ub 196608
- shape: B=16、V=151936、k=20、p=0.95、fp32
- 采集：`torch_npu.profiler`（Text，Level1），warmup 1 / active 4
- 说明：serving 被 CANN 9.0 `ScatterPaKvCache` 阻塞（见 `serve_baseline_910b/MANIFEST.md`），
  本 trace 为 **算子级 no-sort 路径** 的 kernel 链证据，不是 serving 采样链。

## 命令

```bash
PYTHONPATH=/workspace/1102/handoff_repo:$PYTHONPATH \
  /workspace/1102/venv910b/bin/python /tmp/trace_op.py

PYTHONPATH=/workspace/1102/handoff_repo:$PYTHONPATH \
  /workspace/1102/venv910b/bin/python handoff/910b/scripts/extract_chain.py \
  --csv handoff/910b/results/trace_910b/kernel_details.csv \
  --outdir handoff/910b/results/trace_910b \
  --sort-key TopkV2 --apply-key split_pass2 --title trace_910b --first-label topk
```

## 结果

| 段 | p50 | p90 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- |
| `aclnnTopk_TopkV2AiCore_TopKV2` | 133.1 | 133.7 | 133.7 | 133.7 | 133.2 |
| `_apply_topk_topp_split_pass1_kernel` | 34.3 | 34.5 | 34.5 | 34.5 | 34.3 |
| `_apply_topk_topp_split_pass2_kernel` | 27.0 | 27.4 | 27.4 | 27.4 | 27.1 |
| 链总（topk→pass2 结束） | 314.2 | 347.0 | 347.0 | 347.0 | 319.6 |

- `SortAiCoreForDavid` 计数：**0**（no-sort 路径生效）。
- 3 条最慢链的 timestamp 见 `kernel_timeline.md` 第 2 节。
- 原始 CSV：`kernel_details.csv`（24 行：4 TopK + 4 pass1 + 4 pass2）。
