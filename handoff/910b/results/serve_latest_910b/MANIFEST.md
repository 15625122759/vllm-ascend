# MANIFEST — serve_latest_910b

- date: 2026-11-03
- git commit: `d2d3650`（handoff/910b-a2a3；被测代码 `24d76041c`）
- vllm pin: `544f182d9f0c9bfe8bd7645525951d5e3692a542`
- CANN 9.0.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post7 / triton-ascend 3.6.0.post2
- model: `Qwen3.8-27B-w8a8` / V=248320
- additional_config: `{"enable_triton_apply_topk_topp": true, "triton_topk_topp_disable_gate": true}`

## 状态：未运行（环境阻塞）

baseline（`enable_triton_apply_topk_topp=false`）在同一服务配置下 0/300 成功，
失败点与 top-k/top-p 开关无关：

```text
aclnnScatterPaKvCache error code 561000
Op ScatterPaKvCache has 2 uncontiguous input: keyCacheRef, valueCacheRef, ...
```

KV cache 写入在任何 sampling 之前发生，因此 latest 配置无法进入采样阶段；
在 CANN 9.0.0 上继续跑只会复现同一失败。详见 `serve_baseline_910b/MANIFEST.md`
与 `serve_baseline_910b/server.log`。

## 待环境满足后的复测命令

```bash
MODEL=/workspace/user_data/models/Qwen/Qwen3.8-27B-w8a8 \
PORT=8105 VLLM_BIN=/tmp/vllm_eager \
bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_latest_910b --triton 1 --force-gate
```

验收：0 failed；server.log 出现
`[sample/v2] Using Triton apply_top_k_top_p op (full_vocab semantics)`。
