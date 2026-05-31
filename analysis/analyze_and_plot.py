"""Aggregate the eight per-market ``results_sweep.csv`` files into
pooled, family-level, threshold-level, horizon-level and market-level
summaries, and produce all main-text and appendix figures.

Two batches of CSV outputs are written:

* ``analysis/tables/`` — raw aggregations (numeric, one row per indicator
  or family or market).
* ``tables_tex/`` (repo root) — display-ready versions formatted as
  percentages and consumed directly by the LaTeX source via
  ``\\csvreader``.

Primary metric: the **median** p-value of the slope coefficient. Median
is preferred over the mean because the p-value distribution is heavily
right-skewed; the mean column is retained as a secondary statistic.

Usage::

    python3 analyze_and_plot.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_MKT = FIG_DIR / "markets"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_MKT.mkdir(parents=True, exist_ok=True)

MARKETS = ["SP500", "DAX", "Nikkei", "EuroStoxx", "FTSE100",
           "CAC40", "ASX200", "China"]

INDICATOR_FAMILY = {
    "HY_Spread":        "Credit & Term-Structure",
    "Yield_Curve":      "Credit & Term-Structure",
    "YC_10Y3M":         "Credit & Term-Structure",
    "SahmIndicator":    "Macro & Real-Activity",
    "M2_Growth":        "Macro & Real-Activity",
    "CFNAI":            "Macro & Real-Activity",
    "EPU":              "Macro & Real-Activity",
    "Shiller_CAPE":     "Valuation & Trend",
    "BuffettDeviation": "Valuation & Trend",
    "Neely":            "Valuation & Trend",
    "VIX":              "Volatility & Tail-Risk",
    "RSkew":            "Volatility & Tail-Risk",
    "RSV_Ratio":        "Volatility & Tail-Risk",
    "HillTail":         "Volatility & Tail-Risk",
    "CISS":             "Systemic-Risk & Contagion",
    "NFCI":             "Systemic-Risk & Contagion",
    "PatroCorr":        "Systemic-Risk & Contagion",
    "Billio_Corr":      "Systemic-Risk & Contagion",  # renamed from Systemic_Corr
    "Systemic_Corr":    "Systemic-Risk & Contagion",
    "Downside_Beta":    "Systemic-Risk & Contagion",
}

# Display labels: indicator name → label used in figures/tables
DISPLAY_NAME = {
    "HY_Spread":        "HY_Spread",
    "Yield_Curve":      "Yield_Curve",
    "YC_10Y3M":         "YC_10Y3M",
    "SahmIndicator":    "SahmIndicator",
    "M2_Growth":        "M2_Growth",
    "CFNAI":            "CFNAI",
    "EPU":              "EPU",
    "Shiller_CAPE":     "Shiller_CAPE",
    "BuffettDeviation": "BuffettDeviation",
    "Neely":            "Neely",
    "VIX":              "VIX",
    "RSkew":            "RSkew",
    "RSV_Ratio":        "RSV_Ratio",
    "HillTail":         "HillTail",
    "CISS":             "CISS",
    "NFCI":             "NFCI",
    "PatroCorr":        "PatroCorr",
    "Systemic_Corr":    "Billio_Corr",   # renamed
    "Downside_Beta":    "Downside_Beta",
}

INDICATOR_ORDER = [
    "HY_Spread", "Yield_Curve", "YC_10Y3M",
    "SahmIndicator", "M2_Growth", "CFNAI", "EPU",
    "Shiller_CAPE", "BuffettDeviation", "Neely",
    "VIX", "RSkew", "RSV_Ratio", "HillTail",
    "CISS", "NFCI", "PatroCorr", "Systemic_Corr", "Downside_Beta",
]

# Consistent indicator-family colour palette used across all figures.
# A muted, accessible palette (Color Brewer-like) — applied to every chart
# that splits values by family so Figures 6 and 7 share the same scheme.
FAMILY_COLOURS = {
    "Credit & Term-Structure":   "#3B6FB6",
    "Macro & Real-Activity":     "#6C5B9E",
    "Valuation & Trend":         "#4C9A6F",
    "Volatility & Tail-Risk":    "#C53A3A",
    "Systemic-Risk & Contagion": "#D69A2D",
}

# A single sequential colormap re-used by every heatmap so they look uniform.
HEATMAP_CMAP = "RdYlGn_r"
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 0.5

CONFIG_ORDER = ["D-5", "D-10", "D-15",
                "W-4", "W-8", "W-12", "W-16",
                "M-6", "M-12", "M-18", "M-24"]


# ───────────────────────────────────────────────────────────────────────────
# Load
# ───────────────────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    """Pool every results_sweep.csv into one DataFrame."""
    parts = []
    for m in MARKETS:
        d = pd.read_csv(DATA / m / "results" / "results_sweep.csv")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["p_value"].notna()].copy()
    df["family"] = df["indicator"].map(INDICATOR_FAMILY)
    df["config"] = df["freq"].astype(str) + "-" + df["horizon"].astype(str)
    df["sig5"] = (df["p_value"] < 0.05).astype(int)
    df["sig1"] = (df["p_value"] < 0.01).astype(int)
    return df


# ───────────────────────────────────────────────────────────────────────────
# Plot setup — uniform style across the whole thesis
# ───────────────────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
})


# ───────────────────────────────────────────────────────────────────────────
# Figures
# ───────────────────────────────────────────────────────────────────────────

def fig_sig5_ranking(df: pd.DataFrame, out: Path):
    """Headline figure — share of configurations significant at the 5%
    level, one bar per indicator, coloured by family and sorted from
    most to least frequently significant."""
    g = (df.groupby("indicator")
           .agg(sig5=("sig5", "mean"))
           .reset_index())
    g["family"] = g["indicator"].map(INDICATOR_FAMILY)
    g["display"] = g["indicator"].map(DISPLAY_NAME)
    g = g.sort_values("sig5", ascending=False)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    colours = [FAMILY_COLOURS[f] for f in g["family"]]
    bars = ax.bar(g["display"], g["sig5"] * 100, color=colours,
                  edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, g["sig5"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.6,
                f"{val*100:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticklabels(g["display"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Share of configurations significant at 5% (%)")
    ax.set_title("Share of configurations significant at the 5% level, "
                 "by indicator (higher = stronger)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.4)
               for c in FAMILY_COLOURS.values()]
    ax.legend(handles, list(FAMILY_COLOURS.keys()),
              loc="upper right", fontsize=8, framealpha=0.95, ncol=1)
    ax.set_ylim(0, max(g["sig5"] * 100) * 1.18)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_global_ranking(df: pd.DataFrame, out: Path):
    """Figure 1 — Global ranking by *median* p-value.
       Vertical bar chart (one bar per indicator) coloured by family."""
    g = (df.groupby("indicator")
           .agg(median_p=("p_value", "median"),
                sig5=("sig5", "mean"))
           .reset_index())
    g["family"] = g["indicator"].map(INDICATOR_FAMILY)
    g["display"] = g["indicator"].map(DISPLAY_NAME)
    g = g.sort_values("median_p", ascending=True)  # smallest p first (left)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    colours = [FAMILY_COLOURS[f] for f in g["family"]]
    bars = ax.bar(g["display"], g["median_p"], color=colours,
                  edgecolor="black", linewidth=0.4)
    ax.axhline(0.05, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.text(len(g) - 0.4, 0.06, "$p = 0.05$", fontsize=8.5)
    for bar, val, sig in zip(bars, g["median_p"], g["sig5"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}\n({sig*100:.0f}%)",
                ha="center", fontsize=7.5)
    ax.set_xticklabels(g["display"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Median $p$-value")
    ax.set_title("Figure 1 — Global ranking of crisis-prediction indicators "
                 "(median $p$-value; lower $=$ stronger)")
    handles = [plt.Rectangle((0,0),1,1, color=c, ec="black", lw=0.4)
               for c in FAMILY_COLOURS.values()]
    ax.legend(handles, list(FAMILY_COLOURS.keys()),
              loc="upper left", fontsize=8, framealpha=0.95, ncol=1)
    ax.set_ylim(0, max(0.55, g["median_p"].max() * 1.25))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_family_pooled(df: pd.DataFrame, out: Path):
    """Figure 2 — Mean vs Median p-value per family, vertical bars.
       This is the figure cited to justify the mean-to-median switch:
       the gap between mean and median exposes the right-skew."""
    fam_order = ["Volatility & Tail-Risk", "Systemic-Risk & Contagion",
                 "Valuation & Trend", "Credit & Term-Structure",
                 "Macro & Real-Activity"]
    g = (df.groupby("family")
           .agg(mean_p=("p_value", "mean"),
                median_p=("p_value", "median"))
           .reindex(fam_order))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(g))
    w = 0.36
    ax.bar(x - w/2, g["mean_p"],   width=w, color="#3B6FB6",
           edgecolor="black", linewidth=0.4, label="Mean $p$-value")
    ax.bar(x + w/2, g["median_p"], width=w, color="#C53A3A",
           edgecolor="black", linewidth=0.4, label="Median $p$-value")
    ax.axhline(0.05, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" & ", "\n& ") for s in g.index], fontsize=9)
    ax.set_ylabel("$p$-value")
    ax.set_title("Figure 2 — Mean vs Median $p$-value by indicator family")
    ax.legend(fontsize=9)
    for i, v in enumerate(g["mean_p"]):
        ax.text(i - w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(g["median_p"]):
        ax.text(i + w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_threshold_pattern(df: pd.DataFrame, out: Path):
    """Figure 3 — Median p-value by family across drawdown thresholds.
       Axis is *not* inverted; lower y = stronger evidence."""
    g = (df.groupby(["family", "drawdown_threshold"])["p_value"].median()
           .unstack("drawdown_threshold"))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for fam in ["Volatility & Tail-Risk", "Systemic-Risk & Contagion",
                "Valuation & Trend", "Credit & Term-Structure",
                "Macro & Real-Activity"]:
        ax.plot(g.columns * 100, g.loc[fam], marker="o", lw=2.0,
                color=FAMILY_COLOURS[fam], label=fam)
    ax.axhline(0.05, color="black", lw=1.0, ls="--", alpha=0.7,
               label="$p = 0.05$")
    ax.set_xlabel("Drawdown threshold $\\delta$ (%)")
    ax.set_ylabel("Median $p$-value")
    ax.set_title("Figure 3 — Median $p$-value by family and drawdown threshold")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_horizon_heatmap(df: pd.DataFrame, out: Path):
    """Horizon x indicator heatmap over ALL indicators.

    Cells show the share of configurations significant at the 5%
    level. To keep colour semantics homogeneous across every heatmap
    in the paper (green = strong predictor, red = weak predictor) we
    plot ``1 - sig5`` against the shared ``HEATMAP_CMAP``. Indicators
    are ordered by their pooled median p-value (strongest at top).
    """
    rank = df.groupby("indicator")["p_value"].median().sort_values()
    order = rank.index.tolist()
    g = (df.groupby(["indicator", "config"])["sig5"]
           .mean()
           .unstack("config")
           .reindex(index=order, columns=CONFIG_ORDER))
    plotted = 1 - g.values   # high sig5 → low plotted value → green under RdYlGn_r
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    im = ax.imshow(plotted, aspect="auto", cmap=HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(g.columns)))
    ax.set_xticklabels(g.columns, rotation=0, fontsize=9)
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels([DISPLAY_NAME[i] for i in g.index], fontsize=9)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j] * 100
            if not np.isnan(v):
                col = "white" if plotted[i, j] > 0.55 else "black"
                ax.text(j, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8, color=col)
    cb = plt.colorbar(im, ax=ax,
                      label="Share of configurations with $p<0.05$ "
                            "(green = high share, red = low share)",
                      shrink=0.9)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["100%", "75%", "50%", "25%", "0%"])
    ax.set_title("Share of configurations significant at 5% "
                 "by indicator and forecasting horizon (all indicators)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_market_comparison(df: pd.DataFrame, out: Path):
    """Figure 5 — Cross-market comparison by *median* p-value, vertical bars."""
    g = df.groupby("market")["p_value"].median().sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(g.index, g.values, color="#3B6FB6",
                  edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, g.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8.5)
    ax.axhline(0.05, color="black", lw=1.0, ls="--", alpha=0.7, label="$p = 0.05$")
    ax.set_ylabel("Median $p$-value")
    ax.set_title("Figure 5 — Cross-market comparison by median $p$-value "
                 "(lower $=$ more predictable)")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_market_heatmap(df: pd.DataFrame, market: str, out: Path):
    """Per-market heatmap of median p-value, with shared colormap."""
    sub = df[df["market"] == market]
    g = (sub.groupby(["indicator", "drawdown_threshold"])["p_value"].median()
            .unstack("drawdown_threshold")
            .reindex(INDICATOR_ORDER))
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(g.values, aspect="auto", cmap=HEATMAP_CMAP,
                   vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
                   interpolation="nearest")
    ax.set_xticks(range(g.shape[1]))
    ax.set_xticklabels([f"{int(c*100)}%" for c in g.columns])
    ax.set_yticks(range(g.shape[0]))
    ax.set_yticklabels([DISPLAY_NAME[i] for i in g.index], fontsize=9)
    ax.set_xlabel("Drawdown threshold $\\delta$")
    ax.set_title(f"{market}: median $p$-value (lower $=$ stronger evidence) "
                 f"by indicator and drawdown")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            if not np.isnan(v):
                col = "white" if (v < 0.10 or v > 0.40) else "black"
                ax.text(j, i, f"{v:.3f}",
                        ha="center", va="center", fontsize=7.5, color=col)
    plt.colorbar(im, ax=ax, label="Median $p$-value", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_indicator_market_heatmap(df: pd.DataFrame, out: Path):
    """Indicator x market heatmap of median p-value.
       Uses the same heatmap colormap as the per-market versions and the
       Figure 7 significance map so the section reads as one visual story."""
    g = (df.groupby(["market", "indicator"])["p_value"].median()
           .unstack("indicator")
           .reindex(columns=INDICATOR_ORDER))
    market_order = (df.groupby("market")["p_value"].median()
                      .sort_values().index)
    g = g.reindex(market_order)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    im = ax.imshow(g.values, aspect="auto", cmap=HEATMAP_CMAP,
                   vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
                   interpolation="nearest")
    ax.set_xticks(range(len(g.columns)))
    ax.set_xticklabels([DISPLAY_NAME[c] for c in g.columns],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels(g.index, fontsize=9)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            if not np.isnan(v):
                col = "white" if (v < 0.10 or v > 0.40) else "black"
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=7.0, color=col)
    plt.colorbar(im, ax=ax, label="Median $p$-value", shrink=0.9)
    ax.set_title("Indicator $\\times$ market grid of median $p$-value "
                 "(lower $=$ stronger)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_significance_heatmap(df: pd.DataFrame, out: Path):
    """Figure 7 — Indicator × market significance share at the 5 % level.
       Uses the same colormap as the median-p heatmap so Figures 6 and 7
       form a visually-consistent pair, as requested."""
    g = (df.groupby(["market", "indicator"])["sig5"].mean()
           .unstack("indicator")
           .reindex(columns=INDICATOR_ORDER))
    market_order = (df.groupby("market")["sig5"].mean()
                      .sort_values(ascending=False).index)
    g = g.reindex(market_order)
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    # Same RdYlGn_r colormap as the median-p heatmap, but inverted in meaning:
    # high sig-share = strong → green; so we plot (1 − share) so the visual
    # encoding matches the other heatmap (low value = strong).
    plotted = 1 - g.values
    im = ax.imshow(plotted, aspect="auto", cmap=HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(g.columns)))
    ax.set_xticklabels([DISPLAY_NAME[c] for c in g.columns],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(g.index)))
    ax.set_yticklabels(g.index, fontsize=9)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            if np.isnan(v):
                continue
            col = "white" if (v > 0.6 or v < 0.2) else "black"
            ax.text(j, i, f"{int(v*100)}%", ha="center", va="center",
                    fontsize=7.0, color=col)
    cb = plt.colorbar(im, ax=ax, label="Share significant at $p<0.05$ "
                      "(green = high share)", shrink=0.9)
    cb.set_ticks([0, 0.2, 0.5, 0.8, 1.0])
    cb.set_ticklabels(["100%", "80%", "50%", "20%", "0%"])
    ax.set_title("Figure 7 — Share of configurations significant at 5% "
                 "by indicator and market")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# Tables (raw CSVs in analysis/tables) — kept as a record of the aggregations
# ───────────────────────────────────────────────────────────────────────────

def export_tables(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    g = (df.groupby("indicator")
            .agg(median_p=("p_value", "median"),
                 mean_p=("p_value", "mean"),
                 std_p=("p_value", "std"),
                 min_p=("p_value", "min"),
                 sig5=("sig5", "mean"),
                 sig1=("sig1", "mean"),
                 n=("p_value", "size"))
            .sort_values("median_p"))
    g["family"] = g.index.map(INDICATOR_FAMILY)
    g.round(4).to_csv(out_dir / "global_ranking.csv")

    f = (df.groupby("family")
            .agg(median_p=("p_value", "median"),
                 mean_p=("p_value", "mean"),
                 sig5=("sig5", "mean"),
                 sig1=("sig1", "mean"),
                 n=("p_value", "size")))
    f.round(4).to_csv(out_dir / "family_pooled.csv")

    t = (df.groupby(["family", "drawdown_threshold"])["p_value"].median()
            .unstack("drawdown_threshold"))
    t.round(4).to_csv(out_dir / "threshold_family.csv")

    h = (df.groupby(["indicator", "config"])["p_value"].median()
            .unstack("config").reindex(columns=CONFIG_ORDER))
    h.round(4).to_csv(out_dir / "horizon_indicator.csv")

    mi = (df.groupby(["market", "indicator"])["p_value"].median()
            .unstack("indicator").reindex(columns=INDICATOR_ORDER))
    mi.round(4).to_csv(out_dir / "market_indicator.csv")

    for mkt in MARKETS:
        sub = df[df["market"] == mkt]
        d = (sub.groupby(["drawdown_threshold", "indicator"])["p_value"].median()
                .unstack("indicator").reindex(columns=INDICATOR_ORDER))
        d.round(4).to_csv(out_dir / f"detail_{mkt}.csv")

    rows = []
    for (mkt, dd), grp in df.groupby(["market", "drawdown_threshold"]):
        top = grp.groupby("indicator")["p_value"].median().sort_values().head(3)
        for rk, (ind, val) in enumerate(top.items(), 1):
            rows.append({"market": mkt, "drawdown": dd,
                         "rank": rk, "indicator": ind,
                         "median_p": round(val, 4)})
    pd.DataFrame(rows).to_csv(out_dir / "top3_per_market.csv", index=False)


# ───────────────────────────────────────────────────────────────────────────
# LaTeX-ready CSVs (consumed by csvsimple's \csvreader)
# ───────────────────────────────────────────────────────────────────────────

LATEX_INDICATOR = {
    "HY_Spread":        "HY\\_Spread",
    "Yield_Curve":      "Yield\\_Curve",
    "YC_10Y3M":         "YC\\_10Y3M",
    "SahmIndicator":    "SahmIndicator",
    "M2_Growth":        "M2\\_Growth",
    "CFNAI":            "CFNAI",
    "EPU":              "EPU",
    "Shiller_CAPE":     "Shiller\\_CAPE",
    "BuffettDeviation": "BuffettDeviation",
    "Neely":            "Neely",
    "VIX":              "VIX",
    "RSkew":            "RSkew",
    "RSV_Ratio":        "RSV\\_Ratio",
    "HillTail":         "HillTail",
    "CISS":             "CISS",
    "NFCI":             "NFCI",
    "PatroCorr":        "PatroCorr",
    "Systemic_Corr":    "Billio\\_Corr",
    "Downside_Beta":    "Downside\\_Beta",
}

LATEX_FAMILY = {
    "Volatility & Tail-Risk":    "Volatility \\& Tail-Risk",
    "Systemic-Risk & Contagion": "Systemic-Risk \\& Contagion",
    "Valuation & Trend":         "Valuation \\& Trend",
    "Credit & Term-Structure":   "Credit \\& Term-Structure",
    "Macro & Real-Activity":     "Macro \\& Real-Activity",
}


def _fmt_p_pct(v):
    """Format a p-value as a percentage string (e.g. 0.0164 -> '1.6\\%')."""
    if pd.isna(v): return "---"
    pct = v * 100
    if pct < 0.1: return "$<$0.1\\%"
    return f"{pct:.1f}\\%"


def _fmt_pct(v, decimals=1):
    if pd.isna(v): return "---"
    return f"{v*100:.{decimals}f}\\%"


def export_tex_csvs(df: pd.DataFrame, out_dir: Path):
    """Emit display-ready CSVs for direct insertion via csvsimple.

    Cell values are formatted as percentages where appropriate (p-values
    shown as a percentage, e.g.\\ 0.0164 -> ``1.6\\%``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global ranking
    g = (df.groupby("indicator")
            .agg(median_p=("p_value", "median"),
                 mean_p=("p_value", "mean"),
                 std_p=("p_value", "std"),
                 min_p=("p_value", "min"),
                 sig5=("sig5", "mean"),
                 sig1=("sig1", "mean"),
                 n=("p_value", "size"))
            .sort_values("median_p")
            .reset_index())
    g["Rank"] = range(1, len(g)+1)
    g["Indicator"] = g["indicator"].map(LATEX_INDICATOR)
    g["MedianP"]   = g["median_p"].apply(_fmt_p_pct)
    g["MeanP"]     = g["mean_p"].apply(_fmt_p_pct)
    g["MinP"]      = g["min_p"].apply(_fmt_p_pct)
    g["SigFive"]   = g["sig5"].apply(_fmt_pct)
    g["SigOne"]    = g["sig1"].apply(_fmt_pct)
    g["N"]         = g["n"].astype(int)
    g["Family"]    = g["indicator"].map(INDICATOR_FAMILY).map(LATEX_FAMILY)
    g[["Rank","Indicator","MedianP","MeanP","MinP",
       "SigFive","SigOne","N","Family"]].to_csv(
        out_dir / "global_ranking.csv", index=False)

    # Family pooled
    fam_order = ["Volatility & Tail-Risk", "Systemic-Risk & Contagion",
                 "Valuation & Trend", "Credit & Term-Structure",
                 "Macro & Real-Activity"]
    f = (df.groupby("family")
            .agg(median_p=("p_value", "median"),
                 mean_p=("p_value", "mean"),
                 sig5=("sig5", "mean"),
                 sig1=("sig1", "mean"),
                 n=("p_value", "size"))
            .reindex(fam_order).reset_index())
    f["Family"]   = f["family"].map(LATEX_FAMILY)
    f["MedianP"]  = f["median_p"].apply(_fmt_p_pct)
    f["MeanP"]    = f["mean_p"].apply(_fmt_p_pct)
    f["SigFive"]  = f["sig5"].apply(_fmt_pct)
    f["SigOne"]   = f["sig1"].apply(_fmt_pct)
    # No thousand-separator: csvsimple uses ',' as field delimiter, so
    # adding a thousand-separator would break the CSV parser.
    f["N"]        = f["n"].apply(lambda x: f"{int(x)}")
    f[["Family","MedianP","MeanP","SigFive","SigOne","N"]].to_csv(
        out_dir / "family_pooled.csv", index=False)

    # Threshold × family
    t = (df.groupby(["family", "drawdown_threshold"])["p_value"].median()
            .unstack("drawdown_threshold")
            .reindex(fam_order))
    t.columns = ["dthree","dfive","deight","dten","dtwelve","dfifteen"]
    t = t.reset_index()
    t["Family"] = t["family"].map(LATEX_FAMILY)
    for c in ["dthree","dfive","deight","dten","dtwelve","dfifteen"]:
        t[c] = t[c].apply(_fmt_p_pct)
    overall = df.groupby("drawdown_threshold")["p_value"].median()
    overall_row = {
        "Family":  "\\textbf{All families}",
        "dthree":  f"\\textbf{{{_fmt_p_pct(overall[0.03])}}}",
        "dfive":   f"\\textbf{{{_fmt_p_pct(overall[0.05])}}}",
        "deight":  f"\\textbf{{{_fmt_p_pct(overall[0.08])}}}",
        "dten":    f"\\textbf{{{_fmt_p_pct(overall[0.10])}}}",
        "dtwelve": f"\\textbf{{{_fmt_p_pct(overall[0.12])}}}",
        "dfifteen":f"\\textbf{{{_fmt_p_pct(overall[0.15])}}}"}
    t = pd.concat([t, pd.DataFrame([overall_row])], ignore_index=True)
    t[["Family","dthree","dfive","deight","dten","dtwelve","dfifteen"]].to_csv(
        out_dir / "threshold_family.csv", index=False)

    # Horizon × indicator, ALL indicators (rows = indicator, cols = the 11
    # frequency/horizon configs). Indicators ordered by pooled median p.
    # csvsimple uses letters-only column names, so the 11 configs are mapped
    # to safe macro names.
    config_safe = {
        "D-5": "Dfive", "D-10": "Dten", "D-15": "Dfifteen",
        "W-4": "Wfour", "W-8": "Weight", "W-12": "Wtwelve", "W-16": "Wsixteen",
        "M-6": "Msix", "M-12": "Mtwelve", "M-18": "Meighteen", "M-24": "Mtwentyfour",
    }
    rank_order = df.groupby("indicator")["p_value"].median().sort_values()
    order = rank_order.index.tolist()
    h = (df.groupby(["indicator", "config"])["p_value"].median()
            .unstack("config")
            .reindex(index=order, columns=CONFIG_ORDER))
    h.columns = [config_safe[c] for c in h.columns]
    h = h.reset_index()
    h["Indicator"] = h["indicator"].map(LATEX_INDICATOR)
    for c in config_safe.values():
        h[c] = h[c].apply(_fmt_p_pct)
    h[["Indicator"] + list(config_safe.values())].to_csv(
        out_dir / "horizon_all.csv", index=False)

    # Market × indicator
    market_order = ["SP500","DAX","EuroStoxx","CAC40","ASX200","FTSE100",
                    "Nikkei","China"]
    cols = INDICATOR_ORDER
    mi = (df.groupby(["market", "indicator"])["p_value"].median()
            .unstack("indicator")
            .reindex(index=market_order, columns=cols))
    mi["Avg"] = mi.median(axis=1)
    mi = mi.reset_index().rename(columns={"market": "Market"})
    abbr = {"HY_Spread":"HY","Yield_Curve":"YCtwo","YC_10Y3M":"YCthree",
            "SahmIndicator":"Sahm","M2_Growth":"Mtwo","CFNAI":"CFN","EPU":"EPU",
            "Shiller_CAPE":"CAPE","BuffettDeviation":"Buff","Neely":"Neel",
            "VIX":"VIX","RSkew":"RSkw","RSV_Ratio":"RSV",
            "HillTail":"Hill","CISS":"CIS","NFCI":"NFI","PatroCorr":"Patr",
            "Systemic_Corr":"Billio","Downside_Beta":"DBeta"}
    mi.columns = ["Market"] + [abbr[c] for c in cols] + ["Avg"]
    for c in mi.columns[1:]:
        mi[c] = mi[c].apply(_fmt_p_pct)
    mi.to_csv(out_dir / "market_indicator.csv", index=False)

    # Per-market detail
    for mkt in MARKETS:
        sub = df[df["market"] == mkt]
        d = (sub.groupby(["drawdown_threshold", "indicator"])["p_value"].median()
                .unstack("indicator")
                .reindex(columns=cols))
        d = d.reset_index().rename(columns={"drawdown_threshold": "Delta"})
        if mkt == "China":
            d["Yield_Curve"] = np.nan
            d["YC_10Y3M"]   = np.nan
        d["Delta"] = d["Delta"].apply(lambda x: f"{int(x*100)}\\%")
        d.columns = ["Delta"] + [abbr[c] for c in cols]
        for c in d.columns[1:]:
            d[c] = d[c].apply(_fmt_p_pct)
        d.to_csv(out_dir / f"detail_{mkt}.csv", index=False)

    # Top-3 indicators per (market, drawdown) in wide format for csvsimple.
    # Each row: Market | delta | Rank1 | Rank2 | Rank3 where each Rank cell
    # is formatted "Indicator (p)".
    pretty_market = {
        "SP500": "S\\&P~500", "DAX": "DAX", "Nikkei": "Nikkei",
        "EuroStoxx": "Euro~Stoxx", "FTSE100": "FTSE~100", "CAC40": "CAC~40",
        "ASX200": "ASX~200", "China": "China",
    }
    market_order = ["SP500","DAX","EuroStoxx","CAC40","ASX200","FTSE100",
                    "Nikkei","China"]
    top3_rows = []
    for mkt in market_order:
        if mkt not in df["market"].unique():
            continue
        for dd in sorted(df["drawdown_threshold"].unique()):
            sub = df[(df["market"] == mkt) & (df["drawdown_threshold"] == dd)]
            ranked = sub.groupby("indicator")["p_value"].median().sort_values().head(3)
            cells = []
            for ind, val in ranked.items():
                cells.append(f"{LATEX_INDICATOR[ind]} ({_fmt_p_pct(val)})")
            while len(cells) < 3:
                cells.append("---")
            top3_rows.append({
                "Market":  pretty_market.get(mkt, mkt),
                "Delta":   f"{int(dd*100)}\\%",
                "RankOne": cells[0],
                "RankTwo": cells[1],
                "RankThree": cells[2],
            })
    pd.DataFrame(top3_rows).to_csv(out_dir / "top3_per_market.csv", index=False)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    df = load_all()
    print(f"Loaded {len(df):,} regression rows from {df['market'].nunique()} markets.")
    export_tables(df, ROOT / "analysis" / "tables")
    print("Wrote summary tables -> analysis/tables/")
    export_tex_csvs(df, ROOT / "tables_tex")
    print("Wrote LaTeX-ready CSVs -> tables_tex/")
    fig_sig5_ranking(df, FIG_DIR / "fig_sig5_ranking.png")
    fig_global_ranking(df, FIG_DIR / "fig_global_ranking.png")
    fig_family_pooled(df, FIG_DIR / "fig_family_pooled.png")
    fig_threshold_pattern(df, FIG_DIR / "fig_threshold_pattern.png")
    fig_horizon_heatmap(df, FIG_DIR / "fig_horizon_top6.png")
    fig_market_comparison(df, FIG_DIR / "fig_market_comparison.png")
    fig_indicator_market_heatmap(df, FIG_DIR / "fig_market_indicator_heatmap.png")
    fig_significance_heatmap(df, FIG_DIR / "fig_significance_heatmap.png")
    for mkt in MARKETS:
        fig_market_heatmap(df, mkt, FIG_MKT / f"fig_heatmap_{mkt}.png")
    print(f"Wrote figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
