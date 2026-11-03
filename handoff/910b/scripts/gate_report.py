#!/usr/bin/env python3
"""从 bench_op 的扫描结果算 910B 的 B/V 交叉点，并给出门控建议值（不改代码）。

用法：
  python handoff/910b/scripts/gate_report.py \
      --in handoff/910b/results/op_sweep_910b.json \
      --out handoff/910b/results/gate_910b.md
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=1.05,
                    help="建议阈值取 speedup >= margin 的最小 B")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    groups = {}
    for c in data["cases"]:
        groups.setdefault((c["V"], c["dtype"]), []).append((c["B"], c["speedup"]))

    lines = [
        "# 910B 门控交叉点报告\n",
        f"- 数据源：`{args.inp}`",
        f"- 环境：{json.dumps(data.get('env', {}), ensure_ascii=False)}",
        f"- 建议阈值口径：speedup ≥ {args.margin:.2f} 的最小 B\n",
        "| V | dtype | A5 现阈值 | 实测交叉点(≥1.0x) | 建议阈值(≥%.2fx) | 各 B 的 speedup |" % args.margin,
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for (V, dtype), points in sorted(groups.items()):
        points.sort()
        cross = next((b for b, s in points if s >= 1.0), None)
        rec = next((b for b, s in points if s >= args.margin), None)
        if rec is None:
            rec = points[-1][0]
        a5 = 32 if V >= 65536 else 8
        pts = ", ".join(f"B{b}:{s:.2f}x" for b, s in points)
        lines.append(
            f"| {V} | {dtype} | {a5} | {cross if cross is not None else '未达到'} "
            f"| **{rec}** | {pts} |"
        )

    lines += [
        "",
        "> 结论写法：先写实测交叉点，再写建议阈值；建议值只是**下界**，",
        "> 最终阈值还要叠加噪声余量（A5 实测 B=1 已 >1.0x，但最终取 32/8）。",
        "> 若建议值与 A5 不同，请在 PR 里单独开阈值调整 commit，并附本报告，",
        "> 不要在本分支直接改 `_GATE_*`。",
        "",
        "## 复测命令",
        "",
        "```bash",
        "python handoff/910b/scripts/bench_op.py \\",
        "  --out handoff/910b/results/op_sweep_910b.json \\",
        "  --V 32000 151936 248320 --B 1 8 16 32 64 --dtype fp32 bf16 --iters 20",
        "```",
        "",
    ]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
