# 采样链时间戳证据（trace_910b）

- 数据源：`handoff/910b/results/trace_910b/prof/4e64f72c07f1_138067_20261103151753745_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
- 采样次数（配对成功）：**4**；配对规则：每个 `split_pass2` 取之前最近的 `TopkV2`，间隙 ≤ 2000 µs
- 时间基准 t0 = 第一个被筛中 kernel 的 start（绝对时间 1793719073830907 µs），表中为相对秒

## 1. 链各段统计（µs）

| 段 | p50 | p90 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- |
| topk (device) | 133.1 | 133.7 | 133.7 | 133.7 | 133.2 |
| 下发间隙 gap | 154.0 | 186.5 | 186.5 | 186.5 | 159.3 |
| apply_topk kernel | 27.0 | 27.4 | 27.4 | 27.4 | 27.1 |
| **链总时长 (topk→kernel 结束)** | 314.2 | 347.0 | 347.0 | 347.0 | 319.6 |

## 2. 最慢的 10 条采样链（可直接对照 trace 时间轴）

| # | first_start(s) | topk(µs) | gap(µs) | apply_start(s) | apply(µs) | blocks | 链总(µs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.000000 | 133.1 | 186.5 | 0.000320 | 27.4 | 32 | 347.0 |
| 2 | 0.000819 | 133.1 | 154.0 | 0.001106 | 27.0 | 32 | 314.2 |
| 3 | 0.001575 | 133.7 | 150.0 | 0.001858 | 26.8 | 32 | 310.5 |
| 4 | 0.002327 | 133.0 | 146.8 | 0.002606 | 26.9 | 32 | 306.7 |

## 3. 最快与最慢 kernel 对比（用于判断是否存在 kernel 内长尾）

- apply_topk 最慢 27.4 µs / p50 27.0 µs = **1.01×**（>3× 的实例：0 个）

| 全局最慢 5 个 kernel | Start(s) | Duration(µs) | Name |
| --- | --- | --- | --- |
| 1 | 0.001575 | 133.7 | aclnnTopk_TopkV2AiCore_TopKV2 |
| 2 | 0.000000 | 133.1 | aclnnTopk_TopkV2AiCore_TopKV2 |
| 3 | 0.000819 | 133.1 | aclnnTopk_TopkV2AiCore_TopKV2 |
| 4 | 0.002327 | 133.0 | aclnnTopk_TopkV2AiCore_TopKV2 |
| 5 | 0.000320 | 27.4 | _apply_topk_topp_split_pass2_kernel |


## 4. 分核 kernel 分段（pass1 / pass2，µs）

| 段 | p50 | p90 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- |
| split_pass1 | 34.3 | 34.5 | 34.5 | 34.5 | 34.3 |
| split_pass2 | 27.0 | 27.4 | 27.4 | 27.4 | 27.1 |

- `SortAiCoreForDavid` 计数：**0**（no-sort 路径生效）。
- 采样次数（配对）：4；shape B=16、V=151936、k=20、p=0.95、fp32。
