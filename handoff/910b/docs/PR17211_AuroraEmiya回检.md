# PR #17211 回检：v01–v06 对照 AuroraEmiya 2026-10-27 评论

> **2026-11-03 更新**：本文件记录 2026-11-02 的 v01–v06 代码（`06fd67e70`）。
> 当前 PR head `24d76041c` 已移除 V1 接线与 fused，算子移至
> `vllm_ascend/ops/triton/sample/apply_top_k_top_p.py`；本文件仅作历史回检，
> 最新状态以 `README.md` 与 `docs/triton开发规范.md` 为准。

> 评论原文：<https://github.com/vllm-project/vllm-ascend/pull/17211#issuecomment-6442898348>（head `05952165`）
> 回检对象：本地 v01–v06 改动；历史 live 代码 = **v06 + additional_config 开关**（`apply_top_k_top_p.py` sha256 `ef1465ac…`；当前 head `24d76041c` 已变更）。
> 数据口径：Ascend 950PR（A5）×1、CANN 9.1.0、torch 2.10.0+cpu / torch_npu 2.10.0.post4、
> triton-ascend 3.2.2、56 vector core、UB 248 KB、Qwen3.8-27B-w8a8（V=248320）。
> 所有数字均给出原始日志/JSON 路径；**比对数据同时给 A/B 两份路径**。

## 0. 结论摘要

| 评论 | 关注点 | 当前状态 | 结论 |
| --- | --- | --- | --- |
| **AE-1** | 单趟实现是否有可量化收益；无收益则统一 | v01 删全词表 sort；v04 分核；v06 删单程序 no-sort kernel，no-sort 只剩一套实现 | ✅ 满足（并进一步简化） |
| **AE-2** | 对当前 V2 PyTorch 路径、prefill/decode 分开、完整 API、真实输入、trace、模型级单列 | 完整 API 扫描 + **真实 prefill/decode logits 回放** + serving 全版本表 + trace；910B/GPQA 另列 | ✅ 满足 |
| **AE-3** | 与默认 PyTorch 全词表 top-p 语义等价；k+p 同时开启的回归 | 默认 `full_vocab` 未变；38 项 pytest + 35 项 k+p 回归；fp32 逐位相等；附录精度 10/10 | ✅ 满足 |

**回检后保留/回退**：保留 v01（去 sort）、v03（去 host 同步）、v04（词表分核）、v06（统一实现）；
回退 v02（buffer 复用净劣化）、v05（BLOCK=8192 收益不足）。

### 0.1 三个问题的直接回答（2026-11-02 最新代码 `c151cb98b`）

| 问题 | 回答 | 证据 |
| --- | --- | --- |
| **Q1（AE-1）：为什么要单独维护单趟/单程序实现？** | 不维护。历史单趟已删；v06 又把只服务 `B ≥ 56` 的单程序 no-sort 删掉，no-sort 只剩两段分核一套实现。B≥56 实测 ±3% 噪声内，服务区间（B≤16）路径不变。 | `v06_unify_split/README.md`、`op_bench_v06` vs `v04` |
| **Q2（AE-2）：相对 V2 PyTorch 的分阶段数据？** | 已有。完整 API（B=1..64，fp32+bf16）；真实 logits：prefill B=1 1.61x、decode B=6 4.34x；trace 采样链 1197.9→218.6 µs；serving 最新 TPOT 46.59 / P99 51.18 / P99 ITL 400.21。缺 910B/A3（已排除）与 GPQA（无数据集）。 | `docs/logs/ae2_prefill_decode_replay.log`、`result_latest.json`、`kernel_timeline_v04.md` |
| **Q3（AE-3）：与默认路径等价吗？** | 等价。`full_vocab` 默认；fp32 masked logits 对 `_apply_top_k_top_p_pytorch` 逐位相等；38 项 pytest + 35 项 k+p 回归；附录精度 10/10 PASS。`topk_renorm` 仅用于 CANN 对拍，已在文档声明不等价。 | `pytest_comments_38.log`、`precision_appendix_calc_latest.log`、`3.2_ae3_kp_regression.log` |

## 1. AE-1：单趟/重复实现与统一

> 评论原文要点：“单趟只解释了何时可用，没有证明为何值得单独维护；若无可量化收益，倾向统一实现。”

### 1.1 历史单趟（已在 1029 删除）

`single_pass_block >= V` 分支在 A5 上只覆盖 `V ≤ 4096`，实测相对自适应分块仅快 4.1%–12.2%
（2–5.7 µs），且受支持模型最小词表约 3.2 万，**没有工作负载命中**。已在 1029 删除，A5 复测损失在 ±2.6% 噪声内。

- 脚本：`/workspace/user_data/1029/op-validation/scripts/1.1_ae1_single_pass_vs_tiled.py`
- 日志：`/workspace/user_data/1029/op-validation/logs/1.1_ae1_single_pass_vs_tiled.log`

### 1.2 本轮新出现的“单程序 vs 分核”重复实现（v06 已消除）

v04 为实现词表维分核，额外保留了一个“单程序 no-sort kernel”，只在 `C = cores // B == 1`（B ≥ 56）时使用。
按 AE-1 的同一原则，v06 直接删除它，统一走两段分核（`C = max(1, cores // B)`）：

| 项 | v04（单程序 fallback） | v06（统一分核） | 证据（A 侧 / B 侧） |
| --- | --- | --- | --- |
| B=1 op p50 | 0.150 ms | 0.154 ms | `/workspace/user_data/20261102_problems/v04_vocab_split/evidence/op_bench.json` / `.../v06_unify_split/evidence/op_bench.json` |
| B=8 op p50 | 0.161 ms | 0.171 ms | 同上 |
| B=16 op p50 | 0.255 ms | 0.268 ms | 同上 |
| B=32 op p50 | 0.501 ms | 0.492 ms | 同上 |
| B=64 op p50 | 0.954 ms | 0.985 ms | 同上 |
| 代码量 | 多 1 个 kernel（约 80 行） | 单一实现 | `.../v06_unify_split/code/op_vs_v04.diff` |

- B≤16 是服务区间（`max-num-seqs=16`），**代码路径逐行未变**，±5% 差异为跨 run 噪声；
- 只有 B≥56（直接 API 调用）发生替换，实测 ±3% 内；代价是多一次 launch 与 `[B,1,BLOCK]` scratch。
- **结论**：按“无生产工作负载收益的重复实现应删除”的口径，v06 保留。

## 2. AE-2：对 V2 PyTorch 的分阶段对比

> 评论原文要点：Baseline 是 `_apply_top_k_top_p_pytorch`；prefill 首 token / 稳态 decode 分开；
> 测完整 API（sort、参数准备、输出初始化、kernel）；真实记录输入；小 batch + 大词表；混合 k/p；
> p50/p90；warmup/重复次数；排除编译；附 trace 证明 V2 Triton 路径；模型级 TTFT/TPOT/ITL 单列。

### 2.1 已满足的部分

**Baseline / runner（固定写法）**：
- baseline：`vllm_ascend.sample.sampler._apply_top_k_top_p_pytorch`（不是 CANN；CANN 只作 bit-exact 标杆）；
- runner：V2（`VLLM_USE_V2_MODEL_RUNNER=1`），TP=1，`max-num-seqs=16`，`enable_triton_apply_topk_topp=true`；
- 模型：Qwen3.8-27B-w8a8，V=248320；CANN 9.1.0 / triton-ascend 3.2.2 / torch_npu 2.10.0.post4。

**完整 API 延迟（V=248320，p50 ms，warmup 5 / iters 20，排除编译）**：

| B | dtype | PyTorch（A 侧） | v01 Triton | v04 Triton | v06 Triton | 证据（A 侧 / B 侧） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | fp32 | 0.175 | 0.166 | 0.150 | 0.154 | `v00/evidence/op_bench.json` / `v06_unify_split/evidence/op_bench.json` |
| 8 | fp32 | 0.684 | 0.221 | 0.161 | 0.171 | `v01/evidence/op_bench.json` / `v06_unify_split/evidence/op_bench.json` |
| 16 | fp32 | 1.220 | 0.307 | 0.255 | 0.268 | `v01/...` / `v06/...` |
| 32 | fp32 | 2.584 | 0.494 | 0.501 | 0.492 | `v01/...` / `v06/...` |
| 64 | fp32 | 10.971 | 0.953 | 0.954 | 0.985 | `v01/...` / `v06/...` |
| 1 | bf16 | 0.170 | 0.166 | 0.157 | 0.172 | `v00/...` / `v06/...` |
| 8 | bf16 | 0.681 | 0.191 | 0.137 | 0.128 | `v01/...` / `v06/...` |
| 16 | bf16 | 1.262 | 0.241 | 0.198 | 0.192 | `v01/...` / `v06/...` |
| 32 | bf16 | 3.264 | 0.350 | 0.357 | 0.339 | `v01/...` / `v06/...` |
| 64 | bf16 | 9.637 | 0.584 | 0.575 | 0.555 | `v01/...` / `v06/...` |

> 完整 JSON：`/workspace/user_data/20261102_problems/{v00_baseline,v01_sort_elimination,v04_vocab_split,v06_unify_split}/evidence/op_bench.json`。
> PyTorch 列在每次运行中同时测得（`bench_op.py` 的 `pytorch_p50`），不是历史值。

**模型级（serving，300 prompts / rate 10 / max-num-seqs 16）**：

| 版本 | Mean TPOT | P99 TPOT | P99 ITL | Output tok/s | 证据（日志 / JSON） |
| --- | --- | --- | --- | --- | --- |
| v00 | 47.40 | 51.76 | 396.88 | 305.38 | `.../logs/bench/20261102_013616_vllm_bench_serve.log` / `v00/evidence/result.json` |
| v01 | 46.64 | 51.98 | 480.47 | 299.88 | `.../031511...log` / `v01/evidence/result.json` |
| v03 | 47.00 | 51.70 | 350.48 | 310.35 | `.../054737...log` / `v03/evidence/result.json` |
| v04 | 46.86 | 51.51 | 350.00 | 310.72 | `.../073333...log` / `v04/evidence/result.json` |
| v06 | 同 v04（B≤16 路径未变） | 同 v04 | 同 v04 | 同 v04 | 见 v06 README §4 |

> 路径前缀：`/workspace/user_data/1102/stage2_triton/logs/bench/`。
> v03 复测（环境异常点）：TPOT 53.23 / P99 ITL 384.28，`.../055039...log` + `v03/evidence/result_run2.json`。

**trace 证明 V2 Triton 路径被调用（同一 trace 无 sort / 无 PyTorch fallback）**：

| 版本 | 采样链 kernel 计数 | 原始 CSV | 链证据 |
| --- | --- | --- | --- |
| v00 | `SortAiCoreForDavid` 902 + `tiled` 902 | `.../rank0_186509_.../kernel_details.csv` | `v00/evidence/kernel_timeline.md` |
| v01 | `TopKV2` 903 + `nosort` 903 + Sort **0** | `.../rank0_332688_.../kernel_details.csv` | `v01/evidence/kernel_timeline.md` |
| v03 | `TopKV2` 919 + `nosort` 919 + Sort **0** | `.../rank0_602554_.../kernel_details.csv` | `v03/evidence/kernel_timeline.md` |
| v04 | `TopKV2` 903 + `split_pass1/2` 903/903 + Sort **0** | `.../rank0_791590_.../kernel_details.csv` | `v04/evidence/kernel_timeline.md` |
| v05 | `TopKV2` 903 + `split_pass1/2` 903/903 + Sort **0** | `.../rank0_863407_.../kernel_details.csv` | `v05/evidence/kernel_timeline.md` |

> 路径前缀：`/workspace/user_data/1102/stage2_triton/logs/prof/torch/`；
> 服务器日志中同时有 `[sample/v2] Using Triton apply_top_k_top_p op`，
> v03 起还有 `[sample/v2] k_max read from the UVA CPU mirror (no host sync): max_k=...`。

### 2.2 尚未满足（必须写进 PR 的“待回填”）

| 缺口 | 原因 | 回填方式 |
| --- | --- | --- |
| **真实 prefill / decode logits 回放** | ✅ 已补（2026-11-02） | prefill B=1：PyTorch 0.1716 / Triton 0.1064 ms（1.61x）；decode B=6：0.4762 / 0.1096 ms（4.34x），均 `torch.equal`；日志 `/workspace/user_data/1102/docs/logs/ae2_prefill_decode_replay.log` |
| **prefill 首 token 与稳态 decode 分阶段** | ✅ 已补（同一回放） | 表头为 `prefill_first_token` / `decode_steady`，B = 进入 sampler 的 logits 行数（1 / 6） |
| **910B3 / A3 数据** | 本机只有 A5（950PR） | 同脚本复测；门控阈值（大 V B≥32、小 V B≥8）需按 A2/A3 重标定 |
| **GPQA / GSM8K 精度** | 本轮按用户要求不做 e2e 精度 | 1029 已跑 GSM8K/AIME（见 1029/e2e/results/accuracy_comparison.md），GPQA 仍待 |

## 3. AE-3：与 V2 默认路径的语义等价

- 默认 `top_p_mode="full_vocab"`：top-p cutoff 来自**全词表** softmax（先算概率再累加），与
  `_apply_top_k_top_p_pytorch` 一致；`topk_renorm` 仅用于 CANN 对拍。
- fp32 `torch.equal`：`test_full_vocab_bit_exact_pytorch_fp32`（V=32000/151936，B=1/8/32）。
- k+p 同时开启的回归：`/workspace/user_data/1029/op-validation/scripts/3.2_ae3_kp_regression.py`，
  日志 `/workspace/user_data/1029/op-validation/logs/3.2_ae3_kp_regression.log`（35 项 PASS）。
- 本轮最终代码：`pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p.py`
  → **33 passed**，日志 `/workspace/user_data/1102/docs/logs/pytest_final_33passed.log`。
- v01–v06 都**没有改语义**：v01 只改 cutoff 的求法（数学等价，边界用 `>` 对齐 `cumprob <= 1-p`），
  v03/v04/v06 只改调度与并行方式；ties 仍用值比较 `s >= cutoff`。

**结论：满足。** 仍需在 PR 描述中明确：`topk_renorm` 与 V2 默认路径不等价（仅作 CANN 标杆）。

## 4. v01–v06 逐版收益 / 复杂度（回检后的取舍）

| 版本 | 收益（证据） | 复杂度/风险 | 决定 |
| --- | --- | --- | --- |
| v01 去 sort | 链 p50 1197.9 → 256.1 µs；B=16 op 1.267 → 0.307 ms | +1 kernel + token 候选逻辑 | ✅ 保留 |
| v02 buffer 复用 | 复用省 1.2 µs，接线多 9.6 µs | 跨 step 共享输出，processed_logits 风险 | ❌ 回退 |
| v03 去 host 同步 | P99 ITL 480.47 → 350.48 ms（-27%）；gap p99 101 → 0.2 µs | 模块级一次性 hint + patch `SamplingStates.get_top_k_top_p` | ✅ 保留（风险需披露） |
| v04 词表分核 | kernel 84.4 → 39.4 µs（-53%）；链 256.1 → 218.6 µs；B=16 op 0.307 → 0.255 ms | +2 kernel + `[B,C,BLOCK]` scratch（0.8 MB） | ✅ 保留 |
| v05 BLOCK 8192 | 仅 -4.2 µs，pass2 +6% | TOPK≤64 的 UB 封顶分支 + 1.57 MB scratch | ❌ 回退 |
| v06 统一 no-sort | 少约 80 行 / 少 1 kernel；服务区间路径不变 | B≥56 直接 API 多一次 launch（±3%） | ✅ 保留 |

## 5. 可读性整理（本次已做）

1. **模块 docstring 重写**：原文仍写着“A single unified tiled kernel serves every vocabulary size”，
   与现状（sort+tiled / no-sort 分核 / fused）不符。现改为三条 dispatch 说明，并注明
   `C = max(1, cores // B)`、分核的两趟结构、`k_max` hint 协议。
2. **删除单程序 no-sort kernel**（v06）：消除“同一算法两套实现”的阅读负担，也避免读者误判性能差异。
3. **删除失效开关** `VLLM_ASCEND_TOPK_TOPP_DISABLE_VOCAB_SPLIT`（统一后无分支可关）；
   原有两个环境变量逃生开关改为 `additional_config` 字段
   （`triton_topk_topp_disable_gate` / `triton_topk_topp_disable_no_sort`，默认 false），
   **代码中不再读取任何 `VLLM_ASCEND_TOPK_TOPP_*` 环境变量**。
4. 保留的复杂度热点（PR 描述需主动说明）：
   - `torch.topk(width)` 的 `width` 随 `max(k)` 变化（见 §6.1）；
   - `_K_MAX_HINT` 模块级状态 + 对上游 `SamplingStates.get_top_k_top_p` 的 patch（见 §6.2）；
   - 分核的 `partial/nan_part` 每 step 分配（缓存分配器复用）。

## 6. 新引入的风险（必须写进 PR / 评审）

### 6.1 重编译：`TOPK` 是 constexpr

`width = next_power_of_2(max(k))`，在混合 k 请求下会取 1/2/4/…/256 共 9 档，每档一份编译。
服务配置 `top_k=20` 固定为 32（稳态 1 份）；若允许 per-request top_k 混用，最多 9 份。
建议：接线层把 width 向上取整到 {32, 64, 128, 256} 四档，或显式声明该限制。

### 6.2 全局 hint + 上游类 patch

- `vllm_ascend/worker/v2/sample/apply_top_k_top_p.py` 新增模块级 `_K_MAX_HINT`：
  `SamplingStates.get_top_k_top_p`（已 patch）每 step 写入，`apply_top_k_top_p_npu` 消费一次并清空；
  未经过该路径的调用 hint=None → 回退 device 同步，结果仍正确。
- `patch_v2/patch_triton.py` 新增 `states.SamplingStates.get_top_k_top_p = get_top_k_top_p_with_hint`，
  这**改变了 1029 规范里“未新增 monkey patch”的结论**，PR 描述必须列出并说明理由（去掉每 step 28–31 µs
  的 host 同步，把 P99 ITL 从 480 降到 350 ms）。
- 依赖的上游事实：`SamplingStates.top_k` 是 `UvaBackedTensor`，`self.top_k.np` 是源数据（`do_top_k` 也读它）。
  若上游改为设备侧更新，需要重新校验。

### 6.3 分核 scratch

每 step 分配 `[B, C, BLOCK]` fp32（B=16/C=3/BLOCK=4096 ≈ 0.8 MB）+ `[B,C]` int32；
缓存分配器会复用同一块，但 `processed_logits` 等长生命周期场景需确认无并发覆盖（当前输出是新张量）。

## 7. 建议的 PR 描述口径

1. **不要写“整体性能提升”**：低 batch（V=151936 B<32；V=32000 B<8）仍由门控回退 PyTorch；
   本 PR 的定位是“**高并发/大词表算子优化 + 融合采样前置**”。
2. 收益分两栏：**算子级**（表 2.1，B≥16 起 4.5–17x）与**模型级**（v04/v06：TPOT 46.86、P99 ITL 350.00）。
3. 明确 baseline 是 `_apply_top_k_top_p_pytorch`，CANN 只作 bit-exact 标杆。
4. 明确 `full_vocab` 为默认；`topk_renorm` 仅用于 CANN 对拍且**与 V2 默认路径不等价**。
5. 披露 §6 的三项风险；附 §2.1 的 trace 路径与 33 passed 日志。

## 8. 引用路径清单

- 版本与代码：`/workspace/user_data/20261102_problems/{v00_baseline,v01_sort_elimination,v02_output_buffer_reuse,v03_remove_host_sync,v04_vocab_split,v05_blk8192,v06_unify_split}/`
- serving 日志：`/workspace/user_data/1102/stage2_triton/logs/bench/`（上表逐条列出）
- trace CSV：`/workspace/user_data/1102/stage2_triton/logs/prof/torch/<rank0_*>/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- pytest：`/workspace/user_data/1102/docs/logs/pytest_final_33passed.log`
- 1029 精度/回归：`/workspace/user_data/1029/op-validation/{scripts,logs}/`
