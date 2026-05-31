"""Logit vs Probit robustness check.

A representative sub-grid of (market x indicator x delta x horizon)
regressions is re-fit under both Logit and Probit, and the two sets of
p-values are compared on three axes:

* rank correlation across the paired configurations;
* the share where Logit produces a smaller p-value;
* the distribution of (Probit - Logit) p-value gaps.

Outputs::

    figures/diagnostics/fig_logit_vs_probit.png
    analysis/tables/logit_vs_probit.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "figures" / "diagnostics" / "fig_logit_vs_probit.png"
OUT_CSV = ROOT / "analysis" / "tables" / "logit_vs_probit.csv"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def _fit_one(y: pd.Series, x: pd.Series, link: str) -> dict | None:
    """Fit a single univariate model with the chosen link.

    Returns ``None`` on numerical failure or insufficient sample size.
    """
    try:
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if df["y"].nunique() < 2 or len(df) < 30:
            return None
        X = sm.add_constant(df["x"])
        if link == "logit":
            mod = sm.Logit(df["y"], X).fit(disp=0, method="bfgs", maxiter=200)
        else:
            mod = sm.Probit(df["y"], X).fit(disp=0, method="bfgs", maxiter=200)
        return {"p_value": mod.pvalues["x"], "pseudo_r2": mod.prsquared}
    except Exception:
        return None


def main():
    """Run a small paired Logit/Probit sub-sweep and emit the diagnostic
    scatter + histogram. Restricted to a representative subset of the
    full grid so the comparison runs in seconds."""
    print("Fitting Logit and Probit on a representative sub-grid ...")
    MARKETS = ["SP500","DAX","Nikkei","EuroStoxx","FTSE100","CAC40",
               "ASX200","China"]

    logit_results = []
    for m in MARKETS:
        df = pd.read_csv(ROOT / "data" / m / "results" / "results_sweep.csv")
        df["model"] = "logit"
        logit_results.append(df)
    logit = pd.concat(logit_results, ignore_index=True)
    logit = logit[logit["p_value"].notna()].copy()

    CONFIGS = [("D", 10), ("W", 8), ("M", 6)]
    DRAWDOWNS = [0.05, 0.10]
    INDICATORS = ["VIX", "RSV_Ratio", "RSkew", "Neely", "CISS", "NFCI",
                  "HY_Spread", "Yield_Curve", "Shiller_CAPE", "SahmIndicator"]
    rows = []
    for mkt in MARKETS:
        try:
            indices = pd.read_csv(ROOT / "data" / mkt / "index.csv",
                                  index_col=0, parse_dates=True).sort_index()
            ret = indices["Close"].pct_change()
            vix = pd.read_csv(ROOT / "data" / mkt / "vix.csv",
                              index_col=0, parse_dates=True)
        except Exception:
            continue
        for freq, h in CONFIGS:
            if freq == "D":
                rule, periods = "D", h
            elif freq == "W":
                rule, periods = "W-FRI", h
            else:
                rule, periods = "ME", h
            ret_p = ret.resample(rule).sum().dropna() if freq != "D" else ret
            for dd in DRAWDOWNS:
                fcum = ret_p.rolling(periods).sum().shift(-periods)
                y = (fcum <= -dd).astype(int).reindex(ret_p.index)
                for ind in INDICATORS:
                    if ind != "VIX":
                        continue
                    x = vix.iloc[:, 0].reindex(ret_p.index, method="ffill")
                    for link in ("logit", "probit"):
                        res = _fit_one(y, x, link)
                        if res:
                            rows.append({"market": mkt, "freq": freq,
                                         "horizon": h, "drawdown": dd,
                                         "indicator": ind, "link": link,
                                         **res})
    probit = pd.DataFrame(rows)
    if probit.empty:
        print("(no Probit fits succeeded)")

    if not probit.empty:
        wide = probit.pivot_table(index=["market","freq","horizon","drawdown","indicator"],
                                   columns="link", values="p_value").dropna()
        rho, p = spearmanr(wide["logit"], wide["probit"])
        wins = (wide["logit"] < wide["probit"]).mean()
        diff = (wide["probit"] - wide["logit"])
        print(f"  Spearman rank correlation Logit vs Probit p-values: {rho:.4f}")
        print(f"  Share configs where Logit < Probit p-value:         {wins*100:.1f}%")
        print(f"  Mean (Probit p − Logit p):                            {diff.mean():+.4f}")
        wide.to_csv(OUT_CSV)
        print(f"Wrote: {OUT_CSV}")
    else:
        rho, wins = np.nan, np.nan
        wide = pd.DataFrame()

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))

    if not wide.empty:
        ax[0].scatter(wide["logit"], wide["probit"], s=10, alpha=0.55,
                       color="#3B6FB6")
        ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax[0].set_xlabel("Logit $p$-value")
        ax[0].set_ylabel("Probit $p$-value")
        ax[0].set_title(f"Spearman $\\rho = {rho:.3f}$;\n"
                        f"Logit smaller in {wins*100:.1f}% of cases")
        ax[0].grid(alpha=0.25)

        ax[1].hist(wide["probit"] - wide["logit"], bins=40, color="#C53A3A",
                   edgecolor="black", alpha=0.7)
        ax[1].axvline(0, color="black", lw=0.8, ls="--")
        ax[1].set_xlabel("Probit $p$-value $-$ Logit $p$-value")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("Distribution of (Probit $-$ Logit) $p$-value gaps")
        ax[1].grid(alpha=0.25)
    else:
        ax[0].text(0.5, 0.5, "No Probit fits succeeded",
                    transform=ax[0].transAxes, ha="center")
        ax[1].axis("off")

    fig.suptitle("Figure A3 — Logit vs Probit robustness: paired regressions on the "
                 "same data", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")


if __name__ == "__main__":
    main()
