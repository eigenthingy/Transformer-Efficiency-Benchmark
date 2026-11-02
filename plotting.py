"""Figures for the benchmark report.

Every figure is rendered twice — once for a light surface, once for a dark one —
so the README can serve whichever matches the reader's theme. The two are not a
colour inversion of each other: each mode has its own steps of the same three
hues, chosen for its own surface.

Design constraints enforced here:

* Three categorical hues, in fixed slot order, validated for colour-vision
  deficiency at all pairs in both modes (worst-pair CVD ΔE 9.2 light / 9.4
  dark; worst normal-vision ΔE 24.0 / 20.9). Past three models the palette
  folds to a single hue rather than inventing a fourth — identity then comes
  from the direct labels alone.
* Bars are thin and capped in width, with a rounded data-end and a square
  baseline; gridlines and axes are solid hairlines one step off the surface.
* Labels, values and legend text wear ink tokens, never a series colour.
* Bars always start at zero. The accuracy panel needs a zoomed scale to show
  differences of a fraction of a point, so it uses **dots with intervals**
  rather than bars — a truncated bar axis would make length stop meaning
  magnitude.
* Never two scales on one plot: measures in different units get their own panel.
* Values are direct-labelled, so nothing is reachable by colour-matching alone
  (`results/benchmark.md` is the table-view twin).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
THEMES: dict[str, dict] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

FONT_STACK = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]

#: Max bar thickness: the spec is 24 CSS px, i.e. 18 typographic points.
#: Bars never fill their slot — the leftover band is air.
BAR_MAX_PT = 18.0
#: Radius of the rounded data-end: 4 CSS px = 3 pt.
BAR_ROUND_PT = 3.0
#: Marker diameter for dot panels and scatter minimums (spec floor is 8 px).
MARKER_MIN_PT = 10.0


def series_colors(n: int, theme: dict) -> list[str]:
    """Identity colours for `n` models.

    The validated all-pairs palette holds for three slots. Beyond that a fourth
    generated hue would be indistinguishable under CVD, so we drop to a single
    hue and let direct labels carry identity instead.
    """
    slots = theme["series"]
    return list(slots[:n]) if n <= len(slots) else [slots[0]] * n


def color_map(rows: list[dict], theme: dict) -> dict[str, str]:
    """Assign one colour per model, keyed by slug.

    Built once from the **full** model list and shared by every figure, so a
    model keeps its colour in a panel that omits some models. Deriving colours
    from a filtered list instead would repaint the survivors — a reader who
    learned "DistilBERT is orange" would be misled by the next panel.
    """
    return {row["slug"]: c for row, c in zip(rows, series_colors(len(rows), theme))}


def _colors_for(rows: list[dict], theme: dict, cmap: dict[str, str] | None) -> list[str]:
    if cmap is None:
        cmap = color_map(rows, theme)
    fallback = theme["series"][0]
    return [cmap.get(row["slug"], fallback) for row in rows]


def apply_theme(theme: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["ink"],
        "axes.labelcolor": theme["ink2"],
        "axes.edgecolor": theme["axis"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "xtick.labelcolor": theme["ink2"],
        "ytick.labelcolor": theme["ink2"],
        "axes.titlecolor": theme["ink"],
        "grid.color": theme["grid"],
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlepad": 12,
    })


def _style_axes(ax, theme: dict, value_axis: str = "y") -> None:
    """Hairline grid on the value axis only, recessive spines, no tick marks."""
    ax.set_axisbelow(True)
    ax.grid(axis=value_axis, color=theme["grid"], linewidth=1.0, linestyle="-")
    ax.grid(axis="x" if value_axis == "y" else "y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(length=0, pad=6)


def _titles(ax, theme: dict, title: str, subtitle: str = "") -> None:
    """Left-aligned title with an optional subtitle beneath it, both above the axes.

    Drawn as text rather than `set_title` so the subtitle can sit between the
    title and the plot without either colliding with the other.
    """
    if subtitle:
        ax.text(0.0, 1.13, title, transform=ax.transAxes, fontsize=11,
                fontweight="600", color=theme["ink"], va="bottom", ha="left")
        ax.text(0.0, 1.035, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=theme["muted"], va="bottom", ha="left")
    else:
        ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=11,
                fontweight="600", color=theme["ink"], va="bottom", ha="left")


# --------------------------------------------------------------------------- #
# Rounded bars
# --------------------------------------------------------------------------- #
def _pt_to_data(ax, points: float, axis: str) -> float:
    """Convert a length in typographic points to data units on `axis`."""
    px = points * ax.figure.dpi / 72.0
    inv = ax.transData.inverted()
    origin = inv.transform((0.0, 0.0))
    if axis == "y":
        return abs(inv.transform((0.0, px))[1] - origin[1])
    return abs(inv.transform((px, 0.0))[0] - origin[0])


def _data_to_pt(ax, length: float, axis: str) -> float:
    unit = _pt_to_data(ax, 1.0, axis)
    return length / unit if unit else 0.0


def _rounded_bar_path(lo: float, hi: float, base: float, tip: float,
                      r_cross: float, r_value: float, horizontal: bool) -> MPath:
    """A bar with its two data-end corners rounded and its baseline square.

    `r_cross` and `r_value` are the same visual radius expressed in each axis's
    own data units — the axes have different scales, so one radius cannot serve
    both without distorting the corner (or, if applied to the wrong axis,
    drawing a spike clear across the panel).
    """
    sign = 1.0 if tip >= base else -1.0
    r_value = min(r_value, abs(tip - base))
    r_cross = min(r_cross, (hi - lo) / 2.0)
    inner = tip - sign * r_value

    codes = [MPath.MOVETO, MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]

    if not horizontal:
        verts = [
            (lo, base), (lo, inner),
            (lo, tip), (lo + r_cross, tip),      # corner: left wall -> top edge
            (hi - r_cross, tip),
            (hi, tip), (hi, inner),              # corner: top edge -> right wall
            (hi, base), (lo, base),
        ]
    else:
        verts = [
            (base, lo), (inner, lo),
            (tip, lo), (tip, lo + r_cross),
            (tip, hi - r_cross),
            (tip, hi), (inner, hi),
            (base, hi), (base, lo),
        ]
    return MPath(verts, codes)


def rounded_bars(ax, positions: Sequence[float], values: Sequence[float],
                 colors: Sequence[str], theme: dict, *, horizontal: bool = False,
                 slot: float = 1.0, base: float = 0.0) -> None:
    """Draw thin bars with a rounded data-end anchored to a square baseline.

    The caller must have set the limits and drawn the figure once
    (`fig.canvas.draw()`) so the transforms used for the point-to-data
    conversions are current.
    """
    cross_axis = "x" if not horizontal else "y"
    value_axis = "y" if not horizontal else "x"

    width = min(slot * 0.5, _pt_to_data(ax, BAR_MAX_PT, cross_axis))
    width_pt = _data_to_pt(ax, width, cross_axis)

    for pos, val, color in zip(positions, values, colors):
        if val == base:
            continue
        height_pt = _data_to_pt(ax, abs(val - base), value_axis)
        r_pt = max(0.0, min(BAR_ROUND_PT, width_pt / 2.0, height_pt))
        path = _rounded_bar_path(
            pos - width / 2, pos + width / 2, base, val,
            _pt_to_data(ax, r_pt, cross_axis),
            _pt_to_data(ax, r_pt, value_axis),
            horizontal,
        )
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none",
                               linewidth=0, zorder=3))


def _fmt(value: float, digits: int = 1, suffix: str = "") -> str:
    if abs(value) >= 100 and digits <= 1:
        return f"{value:,.0f}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def _legend(ax, labels: Sequence[str], colors: Sequence[str], theme: dict, **kwargs):
    """Legend with fixed small round keys, independent of the marks' own sizes."""
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=c, markeredgecolor=theme["surface"], markeredgewidth=1.5)
        for c in colors
    ]
    return ax.legend(handles, list(labels), labelcolor=theme["ink2"], fontsize=9,
                     handletextpad=0.5, borderpad=0.2, labelspacing=0.5, **kwargs)


# --------------------------------------------------------------------------- #
# Figure 1 — the trade-off frontier
# --------------------------------------------------------------------------- #
def fig_pareto(rows: list[dict], frontier: set[str], theme: dict, *, latency_key: str,
               latency_label: str, cmap: dict[str, str] | None = None):
    """Accuracy against per-request latency; marker area carries parameter count.

    The one figure that answers the question directly: down-and-right is worse
    on both axes, up-and-left is better on both. The connector traces the
    non-dominated set — models off it are beaten on accuracy *and* speed by
    something else in the sweep.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    colors = _colors_for(rows, theme, cmap)

    params = [r["params_total"] for r in rows]
    p_min, p_max = min(params), max(params)

    def marker_size(p: float) -> float:  # area in pt^2; diameter 12..26 pt
        t = 0.0 if p_max == p_min else (p - p_min) / (p_max - p_min)
        return (12.0 + 14.0 * t) ** 2

    xs = [r[latency_key] for r in rows]
    ys = [100 * r["accuracy"] for r in rows]
    x_pad = (max(xs) - min(xs)) * 0.20 or max(xs) * 0.2
    y_span = (max(ys) - min(ys)) or 2.0
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    # Asymmetric: frontier labels sit above their point, so the headroom is the
    # side that gets used. Below-axis room is kept for off-frontier labels.
    ax.set_ylim(min(ys) - y_span * 0.38, max(ys) + y_span * 0.60)

    front = sorted((r for r in rows if r["slug"] in frontier), key=lambda r: r[latency_key])
    if len(front) > 1:
        ax.plot([r[latency_key] for r in front], [100 * r["accuracy"] for r in front],
                color=theme["axis"], linewidth=2.0, solid_capstyle="round", zorder=2)

    for row, color in zip(rows, colors):
        ax.scatter(row[latency_key], 100 * row["accuracy"],
                   s=marker_size(row["params_total"]), color=color,
                   edgecolor=theme["surface"], linewidth=2.0, zorder=4)

    for row in rows:
        on_front = row["slug"] in frontier
        ax.annotate(
            f"{row['label']}\n{100 * row['accuracy']:.1f}%  ·  {row[latency_key]:.0f} ms"
            f"  ·  {row['params_total'] / 1e6:.0f}M",
            xy=(row[latency_key], 100 * row["accuracy"]),
            xytext=(0, 30 if on_front else -36), textcoords="offset points",
            ha="center", va="center", fontsize=8.5, linespacing=1.5,
            color=theme["ink"] if on_front else theme["ink2"],
        )

    ax.set_xlabel(latency_label)
    ax.set_ylabel("Accuracy  (%)")
    _titles(ax, theme, "Accuracy against inference latency",
            "up and to the left is better  ·  marker area scales with parameter count"
            "  ·  the line traces the non-dominated set")
    if len(rows) <= len(theme["series"]):
        _legend(ax, [r["label"] for r in rows], colors, theme, loc="upper left")
    _style_axes(ax, theme, value_axis="y")
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 — what you keep vs what you gain
# --------------------------------------------------------------------------- #
def fig_tradeoff(rows: list[dict], theme: dict, *, baseline_label: str,
                 cmap: dict[str, str] | None = None):
    """Two panels, two units — never one plot with two scales.

    Left: the share of baseline accuracy that survives, on a zero-based axis so
    "almost all of it" reads as almost-full bars. Right: how much faster the
    model answers. A model is worth it when the left bar is near 100 and the
    right bar is well past 1.

    The baseline is normally excluded here (its bars would be 100% and 1× by
    construction), so `cmap` matters: without it the survivors would be
    recoloured from slot 1 and stop matching the other figures.
    """
    height = 1.4 + 0.75 * len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, height), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.08, w_pad=0.06)
    colors = _colors_for(rows, theme, cmap)
    positions = list(range(len(rows)))[::-1]  # first model at the top
    labels = [r["label"] for r in rows]

    panels = [
        (axes[0], [100 * r["accuracy_retention"] for r in rows],
         f"Accuracy retained  (% of {baseline_label})", 100.0, "100%", "%", 1),
        (axes[1], [r["latency_speedup"] for r in rows],
         "Inference speedup  (× vs baseline, p50)", 1.0, "1×", "×", 2),
    ]

    for ax, values, title, ref, ref_label, suffix, digits in panels:
        ax.set_yticks(positions, labels)
        ax.set_ylim(-0.55, len(rows) - 0.45)
        ax.set_xlim(0, max(values + [ref]) * 1.18)
        _titles(ax, theme, title)
        _style_axes(ax, theme, value_axis="x")

    fig.canvas.draw()  # transforms must be current before we build bar paths

    for ax, values, title, ref, ref_label, suffix, digits in panels:
        rounded_bars(ax, positions, values, colors, theme, horizontal=True)
        ax.axvline(ref, color=theme["axis"], linewidth=1.0, zorder=2)
        ax.annotate(ref_label, xy=(ref, len(rows) - 0.5), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=theme["muted"],
                    va="top", ha="left")
        for pos, val in zip(positions, values):
            ax.annotate(_fmt(val, digits, suffix), xy=(val, pos), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=9, color=theme["ink"])
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 — the columns of the table, as small multiples
# --------------------------------------------------------------------------- #
def fig_cost_panels(rows: list[dict], theme: dict, *, latency_key: str, latency_label: str,
                    cmap: dict[str, str] | None = None):
    """One panel per measure.

    Accuracy gets dots with 95% Wilson intervals rather than bars: the
    differences are fractions of a point, so the axis has to be zoomed, and a
    bar on a zoomed axis no longer encodes magnitude by length. Reading the
    overlap of those intervals is the point — where they overlap, the ranking
    on that column is not evidence of a difference.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8), constrained_layout=True)
    # Titles are drawn as text above each axes, which the layout engine does not
    # measure — so reserve the row gap explicitly.
    fig.get_layout_engine().set(hspace=0.13, wspace=0.05, h_pad=0.06, w_pad=0.06)
    colors = _colors_for(rows, theme, cmap)
    positions = list(range(len(rows)))
    labels = [r["label"] + ("\n(baseline)" if r.get("is_baseline") else "") for r in rows]

    accuracy = (axes[0][0], [100 * r["accuracy"] for r in rows])
    bar_panels = [
        (axes[0][1], [r[latency_key] for r in rows], latency_label, " ms", 0),
        (axes[1][0], [r["train_time_s"] for r in rows], "Training time  (s)", " s", 0),
        (axes[1][1], [r["params_total"] / 1e6 for r in rows], "Parameters  (millions)", "M", 0),
    ]

    # --- accuracy: dots + intervals on a zoomed axis ---------------------- #
    ax, values = accuracy
    lows = [100 * r["accuracy_ci95_low"] for r in rows]
    highs = [100 * r["accuracy_ci95_high"] for r in rows]
    pad = (max(highs) - min(lows)) * 0.30 or 1.0
    ax.set_xticks(positions, labels, fontsize=9)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    # Headroom above for the value labels; less below, where nothing is drawn.
    ax.set_ylim(max(0.0, min(lows) - pad * 0.45), min(100.0, max(highs) + pad * 1.5))
    _titles(ax, theme, "Accuracy  (%)")
    ax.annotate("whiskers: 95% Wilson interval", xy=(1.0, 1.035), xycoords="axes fraction",
                fontsize=8.5, color=theme["muted"], va="bottom", ha="right")
    ax.errorbar(positions, values,
                yerr=[[v - lo for v, lo in zip(values, lows)],
                      [hi - v for v, hi in zip(values, highs)]],
                fmt="none", ecolor=theme["ink2"], elinewidth=1.0,
                capsize=4, capthick=1.0, zorder=3)
    for pos, val, hi, color in zip(positions, values, highs, colors):
        ax.plot([pos], [val], marker="o", markersize=MARKER_MIN_PT,
                markerfacecolor=color, markeredgecolor=theme["surface"],
                markeredgewidth=2.0, linestyle="none", zorder=4)
        ax.annotate(_fmt(val, 2), xy=(pos, hi), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=theme["ink"])
    _style_axes(ax, theme, value_axis="y")

    # --- the three cost panels: zero-based bars --------------------------- #
    for ax, values, title, suffix, digits in bar_panels:
        ax.set_xticks(positions, labels, fontsize=9)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(0, max(values) * 1.20)
        _titles(ax, theme, title)
        _style_axes(ax, theme, value_axis="y")

    fig.canvas.draw()

    for ax, values, title, suffix, digits in bar_panels:
        rounded_bars(ax, positions, values, colors, theme)
        for pos, val in zip(positions, values):
            ax.annotate(_fmt(val, digits, suffix), xy=(pos, val), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=theme["ink"])
    return fig


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render(builder: Callable[[dict], plt.Figure], out_dir: Path, stem: str) -> list[Path]:
    """Build `builder` once per mode and write `<stem>.png` / `<stem>-dark.png`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, suffix in (("light", ""), ("dark", "-dark")):
        theme = THEMES[mode]
        apply_theme(theme)
        fig = builder(theme)
        path = out_dir / f"{stem}{suffix}.png"
        fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight", pad_inches=0.24)
        plt.close(fig)
        written.append(path)
    return written
