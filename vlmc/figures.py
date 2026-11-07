"""Report figures, drawn only from results/summary.csv and results/latency.csv.

pareto.png      accuracy vs size for every full-eval variant (y zoomed to where the variants are;
                collapsed 2-bit GPTQ/AWQ are marked at the bottom edge with their true values)
latency_mac.png end-to-end seconds per image question on the laptop, per variant
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax, fig):
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def _read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def pareto(R, b_acc, b_size, thr, rec):
    rows = [r for r in _read(R / "summary.csv")
            if r["comparable_n"] == "True" and not r["variant"].startswith("mac-")]
    for r in rows:
        r["size_gb"], r["vqa_acc"] = float(r["size_gb"]), float(r["vqa_acc"])
    lo = min(0.74, b_acc * thr - 0.02)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    _style(ax, fig)
    series = [("GPTQ (HF)", lambda r: r["family"] == "GPTQ", BLUE, "o", None),
              ("AWQ (HF)", lambda r: r["family"] == "AWQ", ORANGE, "s", None),
              ("GGUF, fine-tuned 3B", lambda r: r["family"] == "GGUF" and r["variant"].startswith("ft-"), AQUA, "D", None),
              ("GGUF, distilled 2B student", lambda r: "distilled" in r["family"], INK, "^", "none"),
              ("Other (base, QLoRA, runtime LoRA)", lambda r: r["family"] in ("QLORA", "GGUF + runtime LoRA")
               or r["variant"] in ("base-bf16", "base-Q4_K_M"), INK2, "o", "none")]
    off = []
    for label, pick, color, marker, face in series:
        pts = [r for r in rows if pick(r)]
        on = [r for r in pts if r["vqa_acc"] >= lo]
        off += [(r, color, marker) for r in pts if r["vqa_acc"] < lo]
        if not pts:
            continue
        kw = dict(facecolors=face, edgecolors=color, linewidths=1.5) if face else \
            dict(c=color, edgecolors=SURF, linewidths=1.5)
        ax.scatter([r["size_gb"] for r in on], [r["vqa_acc"] for r in on], s=46, marker=marker, label=label,
                   zorder=3, **kw)
    ax.scatter([b_size], [b_acc], s=90, c=INK, marker="*", zorder=4, label="Baseline (fine-tuned bf16)")
    ax.axhline(b_acc * thr, color=INK2, lw=1, ls="--", zorder=1)
    ax.set_ylim(lo - 0.012, max(r["vqa_acc"] for r in rows) + 0.008)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter(""))
    x0, x1 = ax.get_xlim()
    ax.text(x1, b_acc * thr, f"{100 * thr:.0f}% of baseline ", va="bottom", ha="right", fontsize=8, color=INK2)
    y_off = lo - 0.008
    for r, color, marker in off:  # collapsed variants: pin to the bottom edge
        ax.scatter([r["size_gb"]], [y_off], s=46, marker="v", c=color, zorder=3)
    if off:  # one combined label so near-identical sizes don't overprint
        txt = ", ".join(f"{r['variant']} {r['vqa_acc']:.3f}" for r, _, _ in sorted(off, key=lambda t: t[0]["variant"]))
        ax.annotate(f"collapsed: {txt}", (max(r["size_gb"] for r, _, _ in off), y_off), xytext=(8, 0),
                    textcoords="offset points", fontsize=7, color=INK2, va="center")
    if off:
        ax.text(x0 * 1.01, lo - 0.002, "below this line: off scale (true value printed)", fontsize=7, color=INK2,
                va="bottom")
        ax.axhline(lo - 0.003, color=GRID, lw=1.2, zorder=1)
    for r in rows:
        if r["variant"] == rec:
            ax.annotate(f"recommended: {r['size_gb']:.2f} GB, {100 * r['vqa_acc'] / b_acc:.1f}%",
                        (r["size_gb"], r["vqa_acc"]), xytext=(8, 8), textcoords="offset points", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Size on disk, LLM + vision encoder (GB, log scale)", color=INK2)
    ax.set_ylabel("VQA accuracy (3,000 held-out questions)", color=INK2)
    ax.set_title("Accuracy vs. size, every variant", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "pareto.png")
    plt.close(fig)


def _pretty(v):
    v = v.replace("mac-", "")
    if v.startswith("student-kd-"):
        return f"2B student {v.split('-')[2]} + Q8_0 vision"
    if v.startswith("ft-") and "-mmproj" in v:
        q, mp = v[len("ft-"):].rsplit("-mmproj", 1)
        return f"3B {q} + {mp} vision"
    return v


def latency(R):
    rows = _read(R / "latency.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["median_s"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.2), dpi=200)
    _style(ax, fig)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.grid(False, axis="y")
    ys = range(len(rows))
    cols = [AQUA if r["variant"].startswith("mac-student") else BLUE for r in rows]
    ax.barh(list(ys), [float(r["median_s"]) for r in rows], color=cols, height=0.62, zorder=2)
    for y, r in zip(ys, rows):
        ax.text(float(r["median_s"]) + 0.04, y, f"{float(r['median_s']):.2f} s  ({float(r['size_gb']):.2f} GB)",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(ys), [_pretty(r["variant"]) for r in rows], fontsize=7.5, color=INK)
    ax.set_xlim(0, max(float(r["median_s"]) for r in rows) * 1.3)
    ax.set_xlabel("Median seconds per image question (single stream)", color=INK2)
    dev = rows[0]["device"]
    ax.set_title(f"End-to-end latency on {dev}: model size barely moves it", loc="left", color=INK, fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Fine-tuned 3B"), Patch(color=AQUA, label="Distilled 2B student")],
              frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(R / "latency_mac.png")
    plt.close(fig)


def make_figures(R, b_acc, b_size_gb, thr, rec):
    R = Path(R)
    pareto(R, b_acc, b_size_gb, thr, rec)
    latency(R)


if __name__ == "__main__":  # regenerate figures from the CSVs alone
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--baseline", default="ft-bf16")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--recommended")
    a = ap.parse_args()
    rows = {r["variant"]: r for r in _read(Path(a.results_dir) / "summary.csv")}
    b = rows[a.baseline]
    make_figures(a.results_dir, float(b["vqa_acc"]), float(b["size_gb"]), a.threshold, a.recommended)
