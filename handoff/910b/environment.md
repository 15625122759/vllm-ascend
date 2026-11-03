# 环境与版本基线

## 1. A5（950PR）已完成测试的实际环境

| 项 | 值 |
| --- | --- |
| 设备 | Ascend 950PR ×1（`soc_version=260`） |
| vector core | 56 |
| UB | 248 KB |
| CANN | 9.1.0 |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| triton-ascend | 3.2.2 |
| vLLM（实验 pin） | 0.29.0（commit `84030bbe3d74d99bad477a3d2e37a973ccd8865c`） |
| vLLM（main pin） | `544f182d9f0c9bfe8bd7645525951d5e3692a542` |
| vllm-ascend | PR head `24d76041c`（V2-only + additional_config） |
| 模型 | Qwen3.8-27B-w8a8，**V=248320**，TP=1 |
| serving | `max-num-seqs=16`、`max-model-len=32768`、`max-num-batched-tokens=8192`、`gpu-memory-utilization=0.90`、`cudagraph_mode=FULL_DECODE_ONLY` |

## 2. 910B（A2/A3）预期差异（必须实测记录）

| 项 | 预期 | 怎么确认 |
| --- | --- | --- |
| soc | A2=910B3 / A3=910C（具体以 `get_soc_version()` 为准） | `env_check.sh` |
| UB | 192 KB（代码按 52 B/elem 推导 tile=2048） | `get_ub_size_bytes()` |
| vector core | 与 A5 不同（以 `get_vectorcore_num()` 为准） | `env_check.sh` |
| CANN/torch/triton | 可能不同，逐项记录 | `env_check.sh` |
| vLLM pin | 建议用 main 分支对应 pin `544f182d`；若用发布版需记录 | `pip show vllm` / `vllm.__version__` |
| 模型/V | 可用同模型（V=248320）或 151936/32000 的模型；记录 tokenizer 的 V | `python -c "from transformers import AutoConfig..."` |

## 3. 安装与校验

```bash
# 1) 选分支
cd /path/to/vllm-ascend
git fetch fork handoff/910b-a2a3
git checkout handoff/910b-a2a3
pip install -e .            # 或按项目文档安装；确保 vllm_ascend 指向本目录

# 2) 校验 vLLM 版本（与 PR 分支的 vllm_version_is 分支匹配）
python - <<'PY'
import vllm, vllm_ascend
print("vllm", vllm.__version__, vllm.__file__)
print("vllm_ascend", vllm_ascend.__file__)
PY

# 3) 环境快照
bash handoff/910b/scripts/env_check.sh
```

## 4. 关键开关（全部 additional_config）

```json
{
  "enable_triton_apply_topk_topp": true,
  "triton_topk_topp_disable_gate": true,
  "triton_topk_topp_disable_no_sort": false
}
```

- `enable_triton_apply_topk_topp`：总开关。
- `triton_topk_topp_disable_gate`：仅基准使用（B 低于交叉点时强制走 Triton）。
- `triton_topk_topp_disable_no_sort`：回退 exact sort 路径（对照用）。

> 代码中不再读取任何 `VLLM_ASCEND_TOPK_TOPP_*` 环境变量；若在 910B 上看到相关环境变量，
> 说明用的是旧分支，请切到 `24d76041c` / `handoff/910b-a2a3`。
