# MANIFEST — serve_baseline_910b

- date: 2026-11-03
- git commit: `d2d3650`（handoff/910b-a2a3；被测代码 `24d76041c`）
- vllm pin: `544f182d9f0c9bfe8bd7645525951d5e3692a542`（editable，`VLLM_TARGET_DEVICE=empty`）
- vllm-ascend: `/workspace/1102/handoff_repo`（`PYTHONPATH` 优先）
- CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post7 / triton-ascend 3.6.0.post2
- soc_version 223 / vector_core 40 / ub_size_bytes 196608
- model: `Qwen3.8-27B-w8a8` / V=248320 / bf16 + w8a8（ascend 量化）
- max_num_seqs 16 / max_model_len 32768 / max_num_batched_tokens 8192
- mode: `--enforce-eager`（torch.compile/FULL_DECODE_ONLY 在 CANN 9.0 上先失败于 fusion pass，故改用 eager；baseline 与 latest 配置一致）
- additional_config: `{"enable_triton_apply_topk_topp": false, "triton_topk_topp_disable_gate": false}`
- command:

```bash
MODEL=/workspace/user_data/models/Qwen/Qwen3.8-27B-w8a8 \
PORT=8104 \
VLLM_BIN=/tmp/vllm_eager \
PYTHONPATH=/workspace/1102/handoff_repo:$PYTHONPATH \
bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_baseline_910b --triton 0
```

（`/tmp/vllm_eager` 仅在 `vllm serve` 子命令后追加 `--enforce-eager`；bench 命令原样透传。）

## 结果

```text
Successful requests: 0
Failed requests:     300
Benchmark duration:  76.42 s
```

**失败**：第 1 个 prefill step 即触发

```text
aclnnScatterPaKvCache failed, error code is 561000
Op ScatterPaKvCache has 2 uncontiguous input: keyCacheRef, valueCacheRef,
but set 0 auto conversion, may not set AutoContiguous() in OpDef.
```

这是 CANN 9.0.0 的 `ScatterPaKvCache` 不接受非连续 KV cache 视图，与
`enable_triton_apply_topk_topp` 开关无关（baseline 未开启 Triton 算子）。
vllm-ascend main 的目标环境为 CANN 9.1（`AutoContiguous()` 生效）。

## 备注

- 原生 `FULL_DECODE_ONLY` 模式更早失败：`qknorm_rope_fusion` 阶段异步报
  `_triton_rope` AICPU 507018；已记录在 `server.log`。
- `VLLM_KV_CACHE_LAYOUT=LBHNC` 会改走 `_C_ascend.npu_scatter_pa_kv_cache`，但当前可用的
  旧版 `vllm_ascend_C`（由 0819 环境提供，CANN 9.0 可加载）不含该新算子，报 `AttributeError`。
- 本分支无法在本机重新编译 `vllm_ascend_C`：`aclmdlRITask` 头冲突（CANN 9.0 vs torch_npu
  的 ACL 头），以及 `sparse_flash_mla` / `quant_lightning_indexer_v2` 依赖 CANN 9.1 的
  `OP_LOGE_FOR_INVALID_*` 宏。
