#!/usr/bin/env python3
"""从 Ascend profiler 的 kernel_details.csv 抽取"采样链"的时间戳级证据。

配对逻辑（每个 decode step 一次采样）：
    torch.sort(device kernel) -> gap -> apply_topk_topp_tiled_kernel
统计：每次采样的 sort 时长、下发间隙、kernel 时长、链总时长；按链时长排序给出最慢 N 条，
并列出全局最长单 kernel（用于排除"长尾在采样 kernel"的假设）。

用法：
  python extract_evidence.py --csv <kernel_details.csv> --outdir <v00_baseline/evidence> \
      --top 10
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--sort-key", default="SortAiCoreForDavid",
                    help="链首段 kernel 名字匹配（v00=SortAiCoreForDavid，v01=TopK）")
    ap.add_argument("--apply-key", default="apply_topk")
    ap.add_argument("--max-gap-us", type=float, default=2000.0,
                    help="链首段与 apply 之间允许的最大配对间隙")
    ap.add_argument("--title", default="v00_baseline")
    ap.add_argument("--first-label", default="torch.sort",
                    help="链首段在报告中的名字（v01 用 topk）")
    args = ap.parse_args()

    rows = []       # (start, dur, name, blocks, wait)
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {n.strip(): i for i, n in enumerate(header)}
        ni, si, di = cols["Name"], cols["Start Time(us)"], cols["Duration(us)"]
        wi, bi = cols.get("Wait Time(us)"), cols.get("Block Num")
        for row in reader:
            if len(row) <= di:
                continue
            name = row[ni]
            if args.apply_key not in name and args.sort_key not in name \
                    and "MatMulV3" not in name:
                continue
            try:
                start = float(row[si].strip().rstrip("\t"))
                dur = float(row[di])
            except ValueError:
                continue
            wait = None
            if wi is not None and len(row) > wi and row[wi].strip():
                try:
                    wait = float(row[wi])
                except ValueError:
                    pass
            blocks = None
            if bi is not None and len(row) > bi and row[bi].strip():
                try:
                    blocks = int(float(row[bi]))
                except ValueError:
                    pass
            rows.append((start, dur, name, blocks, wait))

    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]

    applies = [r for r in rows if args.apply_key in r[2]]
    sorts = [r for r in rows if args.sort_key in r[2]]

    # 配对：每个 apply 取它之前最近的 sort（间隙 <= max_gap_us）
    chains = []
    si = 0
    for a in applies:
        best = None
        while si < len(sorts) and sorts[si][0] <= a[0]:
            best = sorts[si]
            si += 1
        if best is None:
            continue
        sort_end = best[0] + best[1]
        gap = a[0] - sort_end
        if gap > args.max_gap_us or gap < -1:
            continue
        chains.append({
            "first_start_s": (best[0] - t0) / 1e6,
            "first_dur_us": best[1],
            "gap_us": gap,
            "apply_start_s": (a[0] - t0) / 1e6,
            "apply_dur_us": a[1],
            "blocks": a[3],
            "apply_wait_us": a[4],
            "chain_us": best[1] + gap + a[1],
        })

    os.makedirs(args.outdir, exist_ok=True)

    # 明细 csv（按 apply 时间顺序）
    csv_path = os.path.join(args.outdir, "sampling_chain.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chains[0].keys()) if chains else [])
        w.writeheader()
        for c in chains:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})

    def stat(vals, q):
        if not vals:
            return 0.0
        return vals[min(len(vals) - 1, int(len(vals) * q))]

    def us(v):
        return f"{v:.1f}"

    sort_durs = sorted(c["first_dur_us"] for c in chains)
    apply_durs = sorted(c["apply_dur_us"] for c in chains)
    gaps = sorted(c["gap_us"] for c in chains)
    chain_durs = sorted(c["chain_us"] for c in chains)

    # 全局最慢的单 kernel（任意类型）
    slowest = sorted(rows, key=lambda r: r[1], reverse=True)[:5]

    md = []
    md.append(f"# 采样链时间戳证据（{args.title}）\n")
    md.append(f"- 数据源：`{args.csv}`")
    md.append(f"- 采样次数（配对成功）：**{len(chains)}**；配对规则：每个 `{args.apply_key}` "
              f"取之前最近的 `{args.sort_key}`，间隙 ≤ {args.max_gap_us:.0f} µs")
    md.append(f"- 时间基准 t0 = 第一个被筛中 kernel 的 start（绝对时间 {t0:.0f} µs），"
              f"表中为相对秒\n")
    md.append("## 1. 链各段统计（µs）\n")
    md.append("| 段 | p50 | p90 | p99 | max | mean |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    md.append(f"| {args.first_label} (device) | {us(stat(sort_durs,0.5))} | {us(stat(sort_durs,0.9))} | "
              f"{us(stat(sort_durs,0.99))} | {us(sort_durs[-1])} | "
              f"{us(sum(sort_durs)/len(sort_durs))} |")
    md.append(f"| 下发间隙 gap | {us(stat(gaps,0.5))} | {us(stat(gaps,0.9))} | "
              f"{us(stat(gaps,0.99))} | {us(gaps[-1])} | {us(sum(gaps)/len(gaps))} |")
    md.append(f"| apply_topk kernel | {us(stat(apply_durs,0.5))} | {us(stat(apply_durs,0.9))} | "
              f"{us(stat(apply_durs,0.99))} | {us(apply_durs[-1])} | "
              f"{us(sum(apply_durs)/len(apply_durs))} |")
    md.append(f"| **链总时长 ({args.first_label}→kernel 结束)** | {us(stat(chain_durs,0.5))} | "
              f"{us(stat(chain_durs,0.9))} | {us(stat(chain_durs,0.99))} | "
              f"{us(chain_durs[-1])} | {us(sum(chain_durs)/len(chain_durs))} |")
    md.append("")

    md.append("## 2. 最慢的 %d 条采样链（可直接对照 trace 时间轴）\n" % args.top)
    md.append(f"| # | first_start(s) | {args.first_label}(µs) | gap(µs) | apply_start(s) | apply(µs) | blocks | 链总(µs) |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    worst = sorted(chains, key=lambda c: c["chain_us"], reverse=True)[: args.top]
    for i, c in enumerate(worst, 1):
        md.append(f"| {i} | {c['first_start_s']:.6f} | {c['first_dur_us']:.1f} | {c['gap_us']:.1f} | "
                  f"{c['apply_start_s']:.6f} | {c['apply_dur_us']:.1f} | {c['blocks']} | "
                  f"{c['chain_us']:.1f} |")
    md.append("")

    md.append("## 3. 最快与最慢 kernel 对比（用于判断是否存在 kernel 内长尾）\n")
    md.append(f"- apply_topk 最慢 {us(apply_durs[-1])} µs / p50 {us(stat(apply_durs,0.5))} µs "
              f"= **{apply_durs[-1] / stat(apply_durs,0.5):.2f}×**（>3× 的实例："
              f"{sum(1 for v in apply_durs if v > 3 * sum(apply_durs)/len(apply_durs))} 个）")
    md.append("")
    md.append("| 全局最慢 5 个 kernel | Start(s) | Duration(µs) | Name |")
    md.append("| --- | --- | --- | --- |")
    for i, (s, d, name, blocks, wait) in enumerate(slowest, 1):
        md.append(f"| {i} | {(s - t0)/1e6:.6f} | {d:.1f} | {name[:60]} |")
    md.append("")

    out_md = os.path.join(args.outdir, "kernel_timeline.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    print(f"\nwritten: {out_md}\nwritten: {csv_path}")


if __name__ == "__main__":
    main()
