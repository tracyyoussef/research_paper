"""Hill tail-index sensitivity to the order-statistic count k.

The Hill (1975) tail estimator depends on a tuning parameter ``k`` -
the number of order statistics from the left tail used to compute the
estimate. Small ``k`` produces high-variance estimates; large ``k``
pulls in observations that no longer belong to the tail and biases
the estimator downwards. This script computes the rolling Hill index
on the S&P 500 sample for a range of ``k`` values and plots both the
full time series and the long-run mean as a function of ``k``, so the
production choice ``k = 12`` can be defended.

Outputs::

    figures/diagnostics/fig_hill_tail_k_sweep.png
    figures/diagnostics/hill_tail_k_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "SP500" / "index.csv"
OUT_DIR = ROOT / "figures" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rolling_hill(returns: pd.Series, window: int = 252, k: int = 25) -> pd.Series:
    """Rolling Hill tail-index estimator using the ``k`` largest negative
    returns within each ``window``-day window.

    Returns a Series aligned to the input index.
    """
    vals = returns.values
    out = np.full(len(vals), np.nan)
    for i in range(window, len(vals)):
        chunk = vals[i - window:i]
        chunk = chunk[~np.isnan(chunk)]
        losses = np.sort(-chunk[chunk < 0])[::-1]
        if k + 1 >= len(losses) or losses[k] <= 0:
            continue
        out[i] = np.mean(np.log(losses[:k] / losses[k]))
    return pd.Series(out, index=returns.index, name=f"hill_k{k}")


def main():
    idx = pd.read_csv(DATA, index_col=0, parse_dates=True).sort_index()
    ret = idx["Close"].pct_change().dropna()

    # Sweep a wide range around the production value k = 12 (Hall rule
    # for n = 252: k ~ n^{2/5}).
    ks = [5, 8, 12, 16, 20, 25, 30]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5))

    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for j, k in enumerate(ks):
        s = rolling_hill(ret, window=252, k=k).dropna()
        ax.plot(s.index, s.values, lw=1.0,
                color=cmap(j / max(len(ks) - 1, 1)),
                label=f"$k = {k}$")
    ax.set_ylabel("Hill tail index $\\hat\\xi^{\\mathrm{Hill}}$")
    ax.set_title("Hill-tail estimator on the S\\&P 500 "
                 "for $k \\in \\{5, 8, 12, 16, 20, 25, 30\\}$ "
                 "(production: $k = 12$)")
    ax.legend(ncol=4, fontsize=8.5)
    ax.grid(alpha=0.25)

    ks_full = list(range(3, 41))
    avg, std = [], []
    for k in ks_full:
        s = rolling_hill(ret, window=252, k=k)
        avg.append(s.mean())
        std.append(s.std())
    ax2 = axes[1]
    ax2.plot(ks_full, avg, color="#3B6FB6", lw=1.8, label="mean Hill index")
    ax2.fill_between(ks_full,
                     np.array(avg) - np.array(std),
                     np.array(avg) + np.array(std),
                     color="#3B6FB6", alpha=0.15, label="$\\pm 1\\sigma$")
    ax2.axvspan(8, 20, color="green", alpha=0.10, label="recommended range (Hall rule)")
    ax2.axvline(12, color="red", lw=1.2, ls="--", label="production: $k = 12$")
    ax2.set_xlabel("Order-statistic count $k$")
    ax2.set_ylabel("Hill tail index $\\hat\\xi^{\\mathrm{Hill}}$")
    ax2.set_title("Average Hill tail index as a function of $k$")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.25)

    out = OUT_DIR / "fig_hill_tail_k_sweep.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")

    df = pd.DataFrame({"k": ks_full, "mean_hill": avg, "std_hill": std})
    df.round(4).to_csv(OUT_DIR / "hill_tail_k_summary.csv", index=False)
    print(f"Wrote: {OUT_DIR / 'hill_tail_k_summary.csv'}")


if __name__ == "__main__":
    main()
