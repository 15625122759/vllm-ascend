#!/usr/bin/env python3
"""910B 精度计算：10 组 shape/dtype，masked logits vs PyTorch baseline。

用法：
  python handoff/910b/scripts/precision_check.py \
      --out handoff/910b/results/precision_910b

口径（对应当前 V2-only 代码，fused 已删除）：
- fp32：`torch.equal`；
- bf16：交集位置值完全相等，允许边界 kept 差 <= 3。
输出：<out>.log 与 <out>.json
"""
import argparse
import json

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sample.apply_top_k_top_p import apply_top_k_top_p
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import _apply_top_k_top_p_pytorch

SHAPES = [(4, 1000), (8, 4096), (4, 4097), (8, 32000), (2, 151936)]
DTYPES = [torch.float32, torch.bfloat16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="handoff/910b/results/precision_910b")
    args = ap.parse_args()

    init_device_properties_triton()
    torch.manual_seed(42)
    lines = []
    rows = []
    for dt in DTYPES:
        for B, V in SHAPES:
            g = torch.Generator().manual_seed(0)
            logits = (torch.randn(B, V, generator=g) * 4.0).to(dt).npu()
            k = torch.randint(1, 200, (B,), generator=g).int().npu()
            p = (torch.rand(B, generator=g) * 0.09 + 0.90).float().npu()

            ref = _apply_top_k_top_p_pytorch(logits.clone(), k, p)
            out = apply_top_k_top_p(logits.clone(), k, p)

            if dt == torch.float32:
                equal = torch.equal(out, ref)
                kept_diff = 0
                passed = equal
            else:
                kept_out = torch.isfinite(out)
                kept_ref = torch.isfinite(ref)
                kept_diff = int((kept_out.sum(-1) - kept_ref.sum(-1)).abs().max().item())
                both = kept_out & kept_ref
                values_equal = torch.equal(out[both], ref[both])
                moves = int((kept_out != kept_ref).sum().item())
                equal = values_equal
                passed = values_equal and kept_diff <= 3 and moves <= 3

            row = {
                "B": B,
                "V": V,
                "dtype": str(dt).split(".")[-1],
                "masked_equal_vs_pytorch": bool(equal),
                "kept_diff_max": kept_diff,
                "pass": bool(passed),
            }
            rows.append(row)
            lines.append(
                f"{B}x{V:<7} {row['dtype']:<7} masked_equal={equal} kept_diff={kept_diff} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    n_pass = sum(1 for r in rows if r["pass"])
    summary = {
        "baseline": "vllm_ascend.worker.v2.sample.apply_top_k_top_p._apply_top_k_top_p_pytorch",
        "rows": rows,
        "passed": n_pass,
        "total": len(rows),
    }
    print("\n".join(lines))
    print(f"summary: {n_pass}/{len(rows)} PASS")
    with open(f"{args.out}.log", "w") as f:
        f.write("\n".join(lines) + f"\nsummary: {n_pass}/{len(rows)} PASS\n")
    with open(f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"written: {args.out}.log / {args.out}.json")
    if n_pass != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
