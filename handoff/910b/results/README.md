# 910B 结果目录约定

每个 run 一个子目录或一组文件，文件名固定，便于后续汇总：

```text
results/
├── env_910b.json
├── pytest_910b.log
├── op_sweep_910b.json
├── gate_910b.md
├── precision_910b.{log,json}
├── serve_baseline_910b/{MANIFEST.md,server.log,bench.log,result.json}
├── serve_latest_910b/{MANIFEST.md,server.log,bench.log,result.json}
└── trace_910b/{kernel_details.csv,kernel_timeline.md,sampling_chain.csv}
```

每个 serve/trace 目录的 `MANIFEST.md` 至少包含：

```markdown
# MANIFEST
- date:
- git commit: 24d76041c（或 handoff/910b-a2a3 的 tip）
- vllm pin:
- vllm-ascend: local path / branch
- CANN / torch / torch_npu / triton-ascend:
- soc_version / vector_core / ub_size_bytes:
- model / V / dtype:
- max_num_seqs / max_model_len / max_num_batched_tokens:
- additional_config: {...}
- command: <原样粘贴，含所有参数>
- notes:
```

## 提交规则

- 只提交 JSON、日志（`.log`）、Markdown、CSV；**不要提交模型权重或原始 logits 张量**。
- 日志超过 20 MB 时，只保留首尾各 200 行与关键指标段。
- 结果与代码版本绑定：每个结果必须写 commit；若换了 vLLM pin，重跑并另建目录。

## 汇总口径

| 数据 | 必填字段 |
| --- | --- |
| op_sweep | V/B/dtype、pytorch_p50、triton_p50、speedup、kernel_only_ms |
| serve | success/failed、TPOT mean/P99、ITL mean/P99、TTFT mean/P99、out tok/s |
| trace | 链数、topk/pass1/pass2 p50、链 p50/p99、sort 计数、3 条最慢链 timestamp |
| precision | 10 组 pass、max_abs_err、kept_diff |
