# PR 描述草稿（对齐 `24d76041c`，可直接改写粘贴）

> 标题建议：`[Feature][Sample] Add Triton apply_top_k_top_p op for V2 sampling`
>
> 数据来源：A5（950PR）1103 验证包（`/workspace/user_data/1103`），910B/A2-A3 本轮未测。

## What this PR does / why we need it

1. 新增 `vllm_ascend/ops/triton/sample/apply_top_k_top_p.py`：用 Triton 实现 V2 LM-Head
   logits 的 top-k/top-p 过滤，支持两种 top-p 语义：
   - `full_vocab`（默认）：与 V2 默认路径 `_apply_top_k_top_p_pytorch` 一致（全词表概率算
     cutoff），fp32 逐位相等；
   - `topk_renorm`：CANN `torch.npu.npu_top_k_top_p` 语义，仅用于 bit-exact 对拍。
2. 性能路径（`full_vocab` + `k<V`）不再做全词表 `torch.sort`：
   `torch.topk` 取第 k 大值 + 两趟线性 kernel（pass1 求全词表分母，pass2 写 masked logits），
   并把一行按 `C = max(1, vector_cores // B)` 拆到多个 vector core。
3. 接线：仅 V2，`additional_config.enable_triton_apply_topk_topp`（默认 false）；
   V2 `apply_top_k_top_p_npu` 带硬件（310P 拒绝）与 B/V 交叉门控，低于门控时
   `warning_once` 回退 PyTorch。V1 sampler **未修改**；fused softmax 路径已删除，
   算子只返回 masked logits。
4. V2 去 host 同步：从 `SamplingStates.top_k` 的 UVA CPU 镜像读取 `max(k)`，
   避免每 step 的 `int(k.max())`（28–31 µs）阻塞流水。

## Does this PR introduce user-facing change

- 新增 `additional_config.enable_triton_apply_topk_topp`（默认 `false`），关闭时完全走原路径。
- 基准/回退开关同为 `additional_config`，**不使用环境变量**：
  `triton_topk_topp_disable_gate`（绕过 B/V 门控，仅基准）、
  `triton_topk_topp_disable_no_sort`（回退 sort 路径），均默认 `false`。
- 开启后 `full_vocab` 语义与当前默认路径等价（fp32 bit-exact）。

## How tested

### 1) 单算子正确性（1103）

- dtype×B×V：fp32/bf16/fp16 × B=1/8/16/32/64 × V=1000/32000/151936/248320，
  **60/60 PASS**；`topk_renorm` 对 CANN 逐位一致，`full_vocab` 对 PyTorch
  （fp32 逐位、bf16/fp16 边界容差）。
- 参数边界 **48/48 PASS**；非法参数 6 类按预期抛错。
- 回退矩阵 **8/8 + 5 项门控 PASS**（开关、B/V 门控、310P、HAS_TRITON、k/p=None、no-sort）。
- 仓库测试：23 e2e + 6 UT = **29 passed**。
- 原始数据：`1103/op-validation/logs/{1_correctness_sweep,2_fallback_matrix}.json`。

### 2) 真实 logits 回放（prefill / decode 分阶段）

8 条 1024-token prompt、rate 1、max_tokens 32；在 V2 sampler 入口抓取真实 logits/k/p，
离线完整 API 回放（p50/p90 ms）。数据引用 1102 原始回放（当前 no-sort 路径未变）：

| 阶段 | B×V | dtype | k/p | PyTorch | Triton | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| prefill 首 token | 1×248320 | fp32 | 20 / 0.95 | 0.1716 / 0.1767 | **0.1064 / 0.1099** | **1.61x** |
| 稳态 decode | 6×248320 | fp32 | 20 / 0.95 | 0.4762 / 0.4975 | **0.1096 / 0.1129** | **4.34x** |

原始日志：`1102/docs/logs/ae2_prefill_decode_replay.log` / `.json`。

### 3) 算子级扫描（1103，完整 API，p50 ms）

45 组，speedup **1.02x–18.90x**；V=248320：

| B | fp32 PyTorch | fp32 Triton | 加速比 | bf16 Triton | bf16 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1762 | 0.1181 | 1.49x | 0.1381 | 1.27x |
| 16 | 1.2595 | 0.2344 | 5.37x | 0.1728 | 7.65x |
| 64 | 11.0053 | 0.9364 | 11.75x | 0.5368 | 18.14x |

原始 JSON：`1103/op-validation/logs/3_perf_bench.json`。

### 4) 模型级（1103，300 prompts / rate 10 / max-num-seqs 16）

| 指标 | baseline（PyTorch） | 本 PR（Triton） |
| --- | --- | --- |
| Mean TPOT | 47.91 ms | **46.92 ms** |
| P99 TPOT | 52.67 ms | **51.60 ms** |
| P99 ITL | 353.00 ms | **350.33 ms** |
| Output tok/s | 304.35 | **310.30** |

日志：`1103/e2e/stage1_baseline/logs/bench/20261103_024259_vllm_bench_serve.log`、
`1103/e2e/stage2_triton/logs/bench/20261103_034722_vllm_bench_serve.log`。
服务日志分别命中 `[sample/v2] PyTorch fallback` 与
`[sample/v2] Using Triton apply_top_k_top_p op`。

### 5) 精度（1103，evalscope）

| 数据集 | baseline | Triton |
| --- | --- | --- |
| aime24（30） | 23.33% | 26.67% |
| aime25（30） | 20.00% | 16.67% |
| aime26（30） | 16.67% | 20.00% |
| gsm8k（1319） | 96.66% | **96.89%** |

AIME 单题 3.33pp，±1–2 题为采样波动；GSM8K +0.23pp，精度持平。
报告：`1103/e2e/stage{1,2}_*/logs/eval/evalscope_outputs_*/.../reports/`。

## 风险与需要评审知悉的点

1. **`TOPK` 是 constexpr**：`next_pow2(max(k))` 在混合 top_k 请求下最多 9 档编译；
   服务默认 `top_k=20` 固定 32（稳态 1 份）。建议接线层向上取整到 `{32,64,128,256}`。
2. **V2 新增 patch + 模块级 hint**：
   `patch/worker/patch_triton.py` 新增 `SamplingStates.get_top_k_top_p = get_top_k_top_p_with_hint`；
   `_K_MAX_HINT` 一次性消费、未经过该路径时回退 device 同步。
3. **分核 scratch**：每 step 分配 `[B,C,BLOCK]` fp32（B=16/C=3/4096 ≈ 0.8 MB）+ `[B,C]` int32；
   由缓存分配器复用。
4. **机型**：310P 运行时拒绝并回退；A2/A3 tile=2048，门控阈值需复测。
5. 低 batch 仍回退：`V≥65536 → B≥32`、`V<65536 → B≥8`；本 PR 定位是
   “高并发/大词表算子优化”，不宣称整体大幅提升。

## 尚未覆盖

- 910B3 / A3（本分支为 910B 续测交接包）。
- GPQA（本地无数据集）；GSM8K/AIME 见 1103 结果。
- 重编译黑盒产物计数（白盒分析已完成）。
