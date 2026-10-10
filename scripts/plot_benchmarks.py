"""Plot the botorch-rust vs BoTorch benchmark as a dot plot (light and dark).

Reads the JSON written by ``scripts/benchmark.py --json`` and writes
``<out>-light.png`` and ``<out>-dark.png``, which the README shows with a
``<picture>`` element so that GitHub picks the one matching the reader's theme.

    python scripts/benchmark.py --json docs/assets/benchmark.json
    python scripts/plot_benchmarks.py docs/assets/benchmark.json --out docs/assets/benchmark
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

# Validated two-slot categorical palette (blue / orange) and neutral tokens.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e0",
        "link": "#d9d8d2",
        "rust": "#2a78d6",
        "botorch": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#33332f",
        "link": "#45443f",
        "rust": "#3987e5",
        "botorch": "#d95926",
    },
}


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        value = seconds * 1e3
        return f"{value:.1f} ms" if value < 10 else f"{value:.0f} ms"
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.1f} s"


def fmt_tick(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:g} {unit}"
    return f"{seconds:g} s"


def fmt_speedup(ratio: float) -> str:
    return f"{ratio:.1f}×" if ratio < 10 else f"{ratio:.0f}×"


def plot(rows: list[dict], meta: dict, theme: str, out: Path) -> None:
    t = THEMES[theme]
    n = len(rows)
    fig_h = 1.15 + 0.62 * n
    fig, ax = plt.subplots(figsize=(9.0, fig_h), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    times = [r["botorch_rs_s"] for r in rows] + [r["botorch_s"] for r in rows]
    lo, hi = min(times) / 2.2, max(times) * 2.6
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(n - 0.45, -0.75)

    # Recessive hairline gridlines at the decades, no frame.
    decades = [10.0**k for k in range(math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1)]
    ax.xaxis.set_major_locator(FixedLocator(decades))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([fmt_tick(d) for d in decades])
    ax.grid(axis="x", color=t["grid"], linewidth=0.75, zorder=0)
    ax.tick_params(axis="x", colors=t["muted"], length=0, labelsize=9, pad=6)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Wall-clock time (log scale)", color=t["muted"], fontsize=9, labelpad=8)

    for i, r in enumerate(rows):
        rs, bt = r["botorch_rs_s"], r["botorch_s"]
        ax.plot([rs, bt], [i, i], color=t["link"], linewidth=2.2, solid_capstyle="round", zorder=1)
        close = math.log10(bt / rs) < 0.45
        for k, (x, color) in enumerate(((rs, t["rust"]), (bt, t["botorch"]))):
            ax.scatter(
                [x], [i], s=62, color=color, edgecolors=t["surface"], linewidths=1.6, zorder=3
            )
            if close:  # beside the dots: left of the faster one, right of the slower one
                ax.annotate(
                    fmt_time(x),
                    (x, i),
                    xytext=(-9 if k == 0 else 9, 0),
                    textcoords="offset points",
                    ha="right" if k == 0 else "left",
                    va="center",
                    fontsize=8.5,
                    color=t["muted"],
                )
            else:
                ax.annotate(
                    fmt_time(x),
                    (x, i),
                    xytext=(0, 8.5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=t["muted"],
                )
        mid = math.sqrt(rs * bt)
        ax.annotate(
            fmt_speedup(bt / rs),
            (mid, i),
            xytext=(0, 9 if close else 0),
            textcoords="offset points",
            ha="center",
            va="bottom" if close else "center",
            fontsize=10.5,
            fontweight="bold",
            color=t["text"],
            zorder=4,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": t["surface"], "edgecolor": "none"},
        )
        # Row labels in the left margin: task name and its setting.
        ax.annotate(
            r["label"],
            (0, i),
            xycoords=("axes fraction", "data"),
            xytext=(-10, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=t["text"],
        )
        ax.annotate(
            r["detail"],
            (0, i),
            xycoords=("axes fraction", "data"),
            xytext=(-10, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color=t["muted"],
        )

    fig.text(
        0.015,
        1 - 0.36 / fig_h,
        "botorch-rust vs BoTorch",
        fontsize=15,
        fontweight="bold",
        color=t["text"],
        ha="left",
        va="center",
    )
    fig.text(
        0.015,
        1 - 0.68 / fig_h,
        meta["subtitle"],
        fontsize=9,
        color=t["muted"],
        ha="left",
        va="center",
    )
    # Legend: identity never by color alone (dots + names), top right.
    for k, (name, color) in enumerate((("botorch-rust", t["rust"]), ("BoTorch", t["botorch"]))):
        x_in, y_in = 9.0 * (0.725 + 0.14 * k), fig_h - 0.38
        fig.text(
            (x_in + 0.14) / 9.0,
            y_in / fig_h,
            name,
            fontsize=10,
            color=t["text"],
            ha="left",
            va="center",
        )
        fig.patches.append(
            matplotlib.patches.Circle(
                (x_in, y_in),
                0.055,
                transform=fig.dpi_scale_trans,
                facecolor=color,
                edgecolor="none",
                figure=fig,
            )
        )

    fig.subplots_adjust(left=0.215, right=0.975, top=1 - 0.95 / fig_h, bottom=0.62 / fig_h)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="JSON written by scripts/benchmark.py --json")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/benchmark"),
        help="output prefix; writes <out>-light.png and <out>-dark.png",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r.get("plot", True)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        plot(rows, data["meta"], theme, args.out.with_name(f"{args.out.name}-{theme}.png"))


if __name__ == "__main__":
    main()
