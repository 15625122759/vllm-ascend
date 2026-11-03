# 910B / A2-A3 续测交接包

> 分支：`handoff/910b-a2a3`（从最新 PR head `24d76041c` 切出）
> 本分支**只新增 `handoff/910b/` 下的资料与脚本，不修改算子/接线代码**。
> 目的：把 A5（950PR）已完成的测试框架搬到 910B（A2/A3）环境，补齐缺失数据。
> 当前 PR 代码为 **V2-only**：V1 sampler 未修改；算子位于
> `vllm_ascend/ops/triton/sample/apply_top_k_top_p.py`；fused 路径已删除。

## 1. 背景

PR `vllm-project/vllm-ascend#17211` 的最新代码 = `24d76041c`：

- 算子位于 `vllm_ascend/ops/triton/sample/apply_top_k_top_p.py`；
- V2 接线在 `vllm_ascend/worker/v2/sample/apply_top_k_top_p.py`（保留 upstream 的
  `_apply_top_k_top_p_pytorch`，满足条件时路由到算子）；V1 未修改；
- no-sort 路径：`torch.topk` + 两段分核 kernel（`split_pass1` / `split_pass2`）；
- 默认 `full_vocab` 语义，fp32 对 `_apply_top_k_top_p_pytorch` 逐位相等；
- 只保留 masked logits 输出，fused softmax 路径已删除；
- 开关全部走 `--additional-config`，**不使用环境变量**：
  - `enable_triton_apply_topk_topp`
  - `triton_topk_topp_disable_gate`
  - `triton_topk_topp_disable_no_sort`

A5 上已完成：29 项 pytest（23 e2e + 6 UT）、算子级 B=1..64 扫描、真实 prefill/decode logits 回放、
300 prompts 压测、trace 链证据、10 组精度计算。**这些结论不能直接套到 910B**：
910B 的 UB 是 192 KB（tile 会降到 2048）、vector core 数与交叉点不同，门控阈值必须重测。

## 2. 已完成 vs 待补

| 项 | A5（已完成） | 910B（回填状态） | 脚本 |
| --- | --- | --- | --- |
| 环境快照 | ✅ | ✅ `results/env_910b.json` | `scripts/env_check.sh` |
| 单算子 pytest（29 项） | ✅ | ✅ 29 passed | `scripts/run_op_tests.sh` |
| 算子级完整 API 扫描 | ✅ B=1..64，V=248320 | ✅ 3 V × B=1..64 × fp32/bf16 | `scripts/bench_op.py` |
| 门控交叉点 | ✅ A5：V≥65536→B≥32；V<65536→B≥8 | ✅ 实测交叉 B=1（`results/gate_910b.md`） | `scripts/gate_report.py` |
| 精度（10 组 shape/dtype） | ✅ 10/10 PASS | ✅ 10/10 PASS | `scripts/precision_check.py` |
| serving 压测（300 prompts / rate 10） | ✅ TPOT 46.59 / P99 51.18 / P99 ITL 400.21 | ❌ CANN 9.0 阻塞（0/300，`ScatterPaKvCache` 非连续 KV cache） | `scripts/serve_bench.sh` |
| trace 采样链 | ✅ 链 218.6 µs，sort 计数 0 | ◐ 算子级链 p50 314.2 µs，sort 计数 0（非 serving 链） | profiler + `scripts/extract_chain.py` |
| 真实 logits 回放 | ✅ prefill 1.61x / decode 4.34x | ⬜ 需 serving 先可用 | `docs/` 说明 + 复用 A5 脚本 |
| 重编译黑盒计数 | ⬜ 白盒已完成 | ⬜（可选） | 见 `missing_data.md` §8 |
| GPQA 精度 | ⬜ 无数据集 | ⬜（有数据集再做） | 见 `missing_data.md` §9 |


## 3. 快速开始

```bash
# 0) 准备：安装 PR 分支（建议 editable），并确认 vLLM pin 与 910B 匹配
cd /path/to/vllm-ascend
git checkout handoff/910b-a2a3   # 当前 head 为 24d76041c
pip install -e .                 # 具体以项目安装文档为准

# 1) 环境快照（先跑，结果进 results/env_910b.json）
bash handoff/910b/scripts/env_check.sh

# 2) 正确性（29 项）
bash handoff/910b/scripts/run_op_tests.sh

# 3) 算子级扫描（3 个 V × B=1..64 × fp32/bf16，约 10–20 分钟）
python handoff/910b/scripts/bench_op.py \
  --out handoff/910b/results/op_sweep_910b.json \
  --V 32000 151936 248320 --B 1 8 16 32 64 \
  --dtype fp32 bf16 --iters 20

# 4) 门控报告（只输出建议，不改代码）
python handoff/910b/scripts/gate_report.py \
  --in handoff/910b/results/op_sweep_910b.json \
  --out handoff/910b/results/gate_910b.md

# 5) 精度
python handoff/910b/scripts/precision_check.py \
  --out handoff/910b/results/precision_910b

# 6) serving（baseline / latest 各一次；脚本内部 start→bench→stop）
MODEL=/path/to/model bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_baseline_910b --triton 0
MODEL=/path/to/model bash handoff/910b/scripts/serve_bench.sh \
  --out handoff/910b/results/serve_latest_910b --triton 1 --force-gate
```

## 4. 结果目录与命名

```text
handoff/910b/results/
├── env_910b.json                 # 机器/版本/UB/vector core
├── pytest_910b.log               # 29 项结果
├── op_sweep_910b.json            # bench_op 输出
├── gate_910b.md                  # 交叉点与建议阈值
├── precision_910b.{log,json}     # 10 组精度
├── serve_baseline_910b/          # {server.log, bench.log, result.json}
├── serve_latest_910b/
└── trace_910b/                   # kernel_details.csv + kernel_timeline.md
```

每个结果目录附 `MANIFEST.md`：日期、代码 commit、vLLM pin、CANN/torch/triton 版本、
模型与 V、命令（原样）、备注。模板见 `results/README.md`。

## 5. 验收口径

| 项 | 通过标准 |
| --- | --- |
| pytest | 29 passed（23 e2e + 6 UT） |
| 算子级 | B=16 对 PyTorch ≥1.0x；记录每个 V/dtype 的交叉点 |
| 精度 | 10/10 PASS（fp32 逐位相等；bf16 无 token 互换、kept 差 ≤3） |
| serving | 0 failed；baseline 与 latest 同配置；记录 TPOT/P99 TPOT/P99 ITL/tok/s |
| trace | 链内 sort 计数为 0（latest）；给出 p50/p99 与 3 条最慢链时间戳 |
| 门控 | 给出每个 V/dtype 的交叉 B；建议值只是下界，**需叠加噪声余量**；**不直接改代码阈值**，另开 PR |

## 6. 注意事项

1. **不要在本分支改算子代码**：阈值调整请另开 PR，附 910B 数据。
2. A5 的门控（V≥65536→B≥32；V<65536→B≥8）是 950PR 标定的；910B 需重测，
   报告里要明确“建议值”与“实测交叉点”。
3. 910B 的 UB 192 KB，tile 由代码自动推导为 2048；如果实测 UB 报错，
   记录 `torch_npu.npu.get_soc_version()`、`get_ub_size_bytes()` 与 kernel 名。
4. serving 的 `triton_topk_topp_disable_gate` 仅用于基准（B=16 低于交叉点时强制走 Triton），
   生产不要开；baseline 用 `--triton 0`。
5. 结果只提交 JSON/日志/Markdown，不要提交模型权重或原始 logits。

## 7. 脚本清单

| 脚本 | 作用 |
| --- | --- |
| `scripts/env_check.sh` | 环境/版本/UB/vector core 快照 |
| `scripts/run_op_tests.sh` | e2e + UT pytest |
| `scripts/bench_op.py` | 完整 API 的 PyTorch vs Triton 扫描 |
| `scripts/gate_report.py` | 从扫描结果算交叉点与建议阈值 |
| `scripts/precision_check.py` | 附录精度标准的 10 组计算 |
| `scripts/serve_bench.sh` | 有界 start→bench→stop 压测 |
| `scripts/extract_chain.py` | kernel_details.csv → 采样链时间戳证据 |

参考文档：`docs/triton开发规范.md`、`docs/PR17211_AuroraEmiya回检.md`、`docs/PR描述草稿.md`。

## 8. 910B 回填结果（2026-11-03）

完整数据与命令见 **`results/MANIFEST.md`**。要点：

- 单算子：**29 passed**；精度 **10/10 PASS**（fp32 `torch.equal`，bf16 kept 差 ≤1）。
- 算子级（完整 API，p50）：V=248320 时 B=1/8/16/32/64 对 PyTorch
  **1.50x/1.79x/2.14x/2.24x/2.64x（fp32）**、**1.51x/1.82x/2.36x/2.15x/2.54x（bf16）**；
  V=151936 B=64 为 2.33x/2.22x；V=32000 各 B 约 1.03–1.20x。
- 门控：910B 实测交叉点全部为 **B=1**；A5 阈值（32/8）作为保守下界保留，未改代码。
- trace（算子级 no-sort）：TopK p50 133.1 µs + pass1 34.3 µs + pass2 27.0 µs，
  链 p50 314.2 µs、p99 347.0 µs，`SortAiCoreForDavid` 计数 **0**。
- serving **未跑通**：CANN 9.0.0 的 `aclnnScatterPaKvCache` 拒绝非连续 KV cache（错误码 561000），
  baseline 0/300；这是 CANN 版本约束，与 `enable_triton_apply_topk_topp` 开关无关。
  需要 CANN 9.1 环境后按 `serve_baseline_910b/MANIFEST.md` 的同一命令复测。

