#!/usr/bin/env python3
"""910B 算子级 A/B：Triton(full_vocab) vs _apply_top_k_top_p_pytorch，完整 API 计时。

用法：
  python handoff/910b/scripts/bench_op.py \
      --out handoff/910b/results/op_sweep_910b.json \
      --V 32000 151936 248320 --B 1 8 16 32 64 \
      --dtype fp32 bf16 --iters 20

注意：这里测的是直接 API（未传 k_max hint，因此包含一次 host sync），
与 1102 的 bench_op 口径一致；V2 接线有 hint，实际服务更快。
"""
import argparse
import json
import platform
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sample.apply_top_k_top_p import (
    apply_top_k_top_p,
    apply_top_k_top_p_with_sorted,
)
from vllm_ascend.ops.triton.triton_utils import (
    get_ub_size_bytes,
    get_vectorcore_num,
    init_device_properties_triton,
)

try:  # main 的规范位置
    from vllm_ascend.worker.v2.sample.apply_top_k_top_p import (
        _apply_top_k_top_p_pytorch as baseline_topk_topp,
    )
except ImportError:  # 旧树
    from vllm_ascend.sample.sampler import (
        _apply_top_k_top_p_pytorch as baseline_topk_topp,
    )

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def bench(fn, prepare, warmup, iters):
    for _ in range(warmup):
        fn(prepare())
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        x = prepare()
        t0 = time.perf_counter()
        fn(x)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    n = len(ts)
    return ts[n // 2], ts[int(n * 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="910b_handoff")
    ap.add_argument("--V", type=int, nargs="+", default=[32000, 151936, 248320])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ap.add_argument("--dtype", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--top-k-range", type=int, nargs=2, default=[20, 50])
    args = ap.parse_args()

    init_device_properties_triton()
    out = {
        "version": args.version,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {
            "soc": int(torch_npu.npu.get_soc_version()),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "machine": platform.machine(),
            "vector_core": get_vectorcore_num(),
            "ub_size_bytes": get_ub_size_bytes(),
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "cases": [],
    }
    for V in args.V:
        for dtype_name in args.dtype:
            dtype = DTYPES[dtype_name]
            for B in args.B:
                g = torch.Generator().manual_seed(0)
                logits = (torch.randn(B, V, generator=g) * 4).to(dtype).npu()
                k = torch.randint(args.top_k_range[0], args.top_k_range[1] + 1,
                                  (B,), generator=g).int().npu()
                p = (torch.rand(B, generator=g) * 0.09 + 0.90).float().npu()
                sv, si = torch.sort(logits, dim=-1, descending=False, stable=True)

                b50, b90 = bench(lambda x: baseline_topk_topp(x, k, p),
                                 lambda: logits.clone(), args.warmup, args.iters)
                t50, t90 = bench(lambda x: apply_top_k_top_p(x, k, p),
                                 lambda: logits.clone(), args.warmup, args.iters)
                kc50, _ = bench(lambda x: apply_top_k_top_p_with_sorted(sv, si, k, p),
                                lambda: None, args.warmup, args.iters)
                case = {"V": V, "B": B, "dtype": dtype_name,
                        "pytorch_p50": round(b50, 4), "pytorch_p90": round(b90, 4),
                        "triton_p50": round(t50, 4), "triton_p90": round(t90, 4),
                        "speedup": round(b50 / t50, 3),
                        "kernel_only_ms": round(kc50, 4)}
                out["cases"].append(case)
                print(f"V={V} B={B:<4} {dtype_name:<5} pytorch={b50:7.3f} triton={t50:7.3f} "
                      f"speedup={b50 / t50:5.2f}x kernel_only={kc50:7.3f}", flush=True)

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
