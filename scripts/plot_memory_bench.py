#!/usr/bin/env python3
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
"""Draw docs/images/memory_bench.png and memory_bench_fr.png, the README's
process-memory figure of the reference benchmark (shrink_bench).

The figures are the measured ones (Grace, 64 threads, 10^9 elements, 2,000
iterations, static chunks of 256): edit MEASURES when the benchmark is run
again. Needs matplotlib; uses the Inter font when installed.

    python3 scripts/plot_memory_bench.py   # from the repository root
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# (container, GB), top to bottom; the last one is highlighted.
MEASURES = [("std::vector", 125.9), ("zero", 40.2), ("zero_one_value", 17.3), ("zero_any", 7.26)]
DYN_GB = 6.0  # zero_any with dynamic chunks (_dyn)

TEXT = {
    "en": {"title": "Process memory, 10⁹ elements, 2,000 iterations (Grace, 64 threads)",
           "unit": "GB", "dec": ".", "dyn": "{} GB with dynamic chunks (_dyn)", "out": "memory_bench.png"},
    "fr": {"title": "Mémoire du process, 10⁹ éléments, 2 000 itérations (Grace, 64 threads)",
           "unit": "Go", "dec": ",", "dyn": "{} Go avec les chunks dynamiques (_dyn)", "out": "memory_bench_fr.png"},
}

GREY, GREEN, AXIS, INK = "#8e95a1", "#1e8a3c", "#6b7280", "#222222"


def _value(gb, dec):
    text = f"{gb:.2f}" if gb < 10 else f"{gb:.1f}"
    return text.replace(".", dec)


def draw(lang, out_dir):
    t = TEXT[lang]
    if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Inter"
    fig, ax = plt.subplots(figsize=(11.7, 4.42), dpi=100)
    fig.subplots_adjust(left=0.145, right=0.97, top=0.86, bottom=0.04)
    names = [n for n, _ in MEASURES][::-1]
    values = [v for _, v in MEASURES][::-1]
    colors = [GREEN] + [GREY] * (len(values) - 1)
    ax.barh(names, values, height=0.62, color=colors)
    top = max(values)
    for i, v in enumerate(values):
        best = i == 0
        ax.text(v + top * 0.012, i, f"{_value(v, t['dec'])} {t['unit']}", va="center", fontsize=13,
                color=INK, fontweight="bold" if best else "normal")
    ax.text(top * 0.24, 0, t["dyn"].format(f"{DYN_GB:.1f}".replace(".", t["dec"])), va="center", fontsize=11, color=GREEN)
    ax.set_xlim(0, top * 1.12)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=INK, pad=4)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    fig.text(0.145, 0.93, t["title"], fontsize=13.5, color=INK, va="center")
    fig.savefig(out_dir / t["out"], facecolor="white")
    plt.close(fig)


def main():
    out_dir = Path("docs/images")
    for lang in TEXT:
        draw(lang, out_dir)
        print(f"wrote {out_dir / TEXT[lang]['out']}")


if __name__ == "__main__":
    main()
