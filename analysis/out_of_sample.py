"""Out-of-sample robustness check: drop the most recent five years.

We re-fit the full univariate-Logit grid on the pre-2021 history only
(every observation up to 1 January 2021) and compare the resulting
per-indicator statistics to the full-sample statistics produced by the
main sweep in ``crisis_indicator_sweep.ipynb``. If the ranking survives the removal of
the most recent five years (which include the COVID-19 crash and the
2022 drawdown), it is not an artefact of recent data.

The full-sample p-values are read from each market's
``results_sweep.csv`` (so the full-sample side reproduces the main
ranking table by construction); the pre-2021 p-values are produced by
re-running the same regression grid here, on the truncated price
series. The sweep machinery is reproduced in this file so the script
is self-contained.

Two panels are produced (mirroring the headline figures of the paper):

  * share of configurations significant at the 5% level, full vs pre-2021;
  * median p-value, full vs pre-2021.

Outputs::

    figures/diagnostics/fig_oos_pvalue_comparison.png
    analysis/tables/oos_comparison.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_FIG = ROOT / "figures" / "diagnostics" / "fig_oos_pvalue_comparison.png"
OUT_CSV = ROOT / "analysis" / "tables" / "oos_comparison.csv"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Cut-off: keep every observation up to 1 January 2021 (i.e. drop the
# most recent five years and re-fit on the pre-2021 history).
CUTOFF = pd.Timestamp("2021-01-01")

MARKETS = ["SP500", "DAX", "Nikkei", "EuroStoxx", "FTSE100",
           "CAC40", "ASX200", "China"]

MARKET_CFG = {
    "SP500":     {"mkt": "SPY",        "fin": "XLF",     "ff_finance_col": 10, "ff_market_col": 0},
    "DAX":       {"mkt": "EXSA.DE",    "fin": "EXV1.DE", "ff_finance_col": 10, "ff_market_col": 0},
    "Nikkei":    {"mkt": "1615.T",     "fin": "1617.T",  "ff_finance_col": 10, "ff_market_col": 0},
    "EuroStoxx": {"mkt": "^STOXX50E",  "fin": "EXX1.DE", "ff_finance_col": 10, "ff_market_col": 0},
    "FTSE100":   {"mkt": "ISF.L",      "fin": "IUKF.L",  "ff_finance_col": 10, "ff_market_col": 0},
    "CAC40":     {"mkt": "EXSA.DE",    "fin": "EXV1.DE", "ff_finance_col": 10, "ff_market_col": 0},
    "ASX200":    {"mkt": "STW.AX",     "fin": "OZF.AX",  "ff_finance_col": 10, "ff_market_col": 0},
    "China":     {"mkt": "MCHI",       "fin": "CHIE",    "ff_finance_col": 10, "ff_market_col": 0},
}

DRAWDOWN_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
EXPERIMENT = [(5, "D"), (10, "D"), (15, "D"),
              (4, "W"), (8, "W"), (12, "W"), (16, "W"),
              (6, "M"), (12, "M"), (18, "M"), (24, "M")]
_ME = "ME" if tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2) else "M"
FREQ_MAP = {"D": "D", "W": "W-FRI", "M": _ME}

INDICATOR_COLS = [
    "HY_Spread", "Yield_Curve", "Neely", "PatroCorr",
    "SahmIndicator", "VIX", "M2_Growth", "BuffettDeviation",
    "Systemic_Corr", "Shiller_CAPE",
    "YC_10Y3M", "CFNAI", "CISS", "Downside_Beta", "NFCI", "EPU",
    "RSkew", "RSV_Ratio", "HillTail",
]

PLAIN = {
    "HY_Spread": "HY_Spread", "Yield_Curve": "Yield_Curve",
    "YC_10Y3M": "YC_10Y3M", "SahmIndicator": "SahmIndicator",
    "M2_Growth": "M2_Growth", "CFNAI": "CFNAI", "EPU": "EPU",
    "Shiller_CAPE": "Shiller_CAPE", "BuffettDeviation": "BuffettDeviation",
    "Neely": "Neely", "VIX": "VIX", "RSkew": "RSkew",
    "RSV_Ratio": "RSV_Ratio", "HillTail": "HillTail", "CISS": "CISS",
    "NFCI": "NFCI", "PatroCorr": "PatroCorr",
    "Systemic_Corr": "Billio_Corr", "Downside_Beta": "Downside_Beta",
}


# ───────────────────────────────────────────────────────────────────────────
#  Sweep machinery (kept identical to the main analysis so the full-sample
#  side reproduces the main ranking table)
# ───────────────────────────────────────────────────────────────────────────

def compute_crisis_target(returns: pd.Series, horizon: int, drawdown: float) -> pd.Series:
    fut_cum = returns.rolling(window=horizon).sum().shift(-horizon)
    out = (fut_cum <= -drawdown).astype("Int64")
    out[fut_cum.isna()] = pd.NA
    return out


def fit_logit_one(df: pd.DataFrame, indicator: str, target: str) -> float | None:
    sub = df[[indicator, target]].dropna()
    if sub.empty or sub[target].nunique() < 2 or len(sub) < 30:
        return None
    X = sm.add_constant(sub[[indicator]])
    y = sub[target].astype(int)
    try:
        mod = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
        return float(mod.pvalues[indicator])
    except Exception:
        return None


def rolling_hill(returns: pd.Series, window: int = 252, k: int = 12) -> pd.Series:
    vals = returns.values
    out = np.full(len(vals), np.nan)
    for i in range(window, len(vals)):
        chunk = vals[i - window:i]
        chunk = chunk[~np.isnan(chunk)]
        losses = np.sort(-chunk[chunk < 0])[::-1]
        if k + 1 >= len(losses) or losses[k] <= 0:
            continue
        out[i] = np.mean(np.log(losses[:k] / losses[k]))
    return pd.Series(out, index=returns.index)


def rolling_downside_beta(fin_ret: pd.Series, mkt_ret: pd.Series,
                          window: int = 252, min_obs: int = 63) -> pd.Series:
    out = pd.Series(np.nan, index=fin_ret.index)
    fv, mv = fin_ret.values, mkt_ret.values
    for i in range(window, len(fv)):
        f, m = fv[i - window:i], mv[i - window:i]
        mask = m < 0
        if mask.sum() < min_obs:
            continue
        var_m = m[mask].var()
        if var_m > 0:
            out.iloc[i] = np.cov(f[mask], m[mask])[0, 1] / var_m
    return out


def _resample_last(s: pd.Series, freq: str) -> pd.Series:
    return s if freq == "D" else s.resample(freq).last()


def _load_market_daily(market: str, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Daily dataframe on the native trading-day index, truncated to
    ``<= cutoff`` before any rolling indicator is computed."""
    cfg = MARKET_CFG[market]
    dp = DATA / market

    idx_raw = pd.read_csv(dp / "index.csv", index_col=0, parse_dates=True).sort_index()
    idx_raw = idx_raw[idx_raw.index <= cutoff]
    df_daily = idx_raw[["Close"]].copy().rename(columns={"Close": market})
    df_daily["returns"] = (df_daily[market].pct_change()
                              if not df_daily.empty
                              else pd.Series(dtype=float, index=df_daily.index))

    hy   = pd.read_csv(dp / "hy_spread.csv",         index_col=0, parse_dates=True).iloc[:, 0]
    yc   = pd.read_csv(dp / "yield_curve.csv",       index_col=0, parse_dates=True).iloc[:, 0]
    yc3m = pd.read_csv(dp / "yield_curve_10y3m.csv", index_col=0, parse_dates=True).iloc[:, 0]
    vix  = pd.read_csv(dp / "vix.csv",               index_col=0, parse_dates=True).iloc[:, 0]
    ciss = pd.read_csv(dp / "ciss.csv",              index_col=0, parse_dates=True).iloc[:, 0]
    nfci = pd.read_csv(dp / "nfci.csv",              index_col=0, parse_dates=True).iloc[:, 0]
    epu  = pd.read_csv(dp / "epu.csv",               index_col=0, parse_dates=True).iloc[:, 0]

    m2_raw    = pd.read_csv(dp / "m2.csv",     index_col=0, parse_dates=True)
    gdp_raw   = pd.read_csv(dp / "gdp.csv",    index_col=0, parse_dates=True).iloc[:, 0]
    unemp_raw = pd.read_csv(dp / "unrate.csv", index_col=0, parse_dates=True)
    cfnai     = pd.read_csv(dp / "cfnai.csv",  index_col=0, parse_dates=True).iloc[:, 0]
    cape      = pd.read_csv(dp / "shiller_cape.csv", index_col=0, parse_dates=True).iloc[:, 0]
    cape.name = "CAPE"

    m2_growth = (m2_raw["M2"].pct_change(12) * 100).rename("M2_Growth")
    u3   = unemp_raw["UNRATE"].rolling(3).mean()
    sahm = (u3 - u3.rolling(12).min()).rename("SahmIndicator")

    sectors_raw = pd.read_csv(dp / "sectors.csv", index_col=0, parse_dates=True)
    mkt_col, fin_col = cfg["mkt"], cfg["fin"]
    valid = [c for c in sectors_raw.columns if sectors_raw[c].notna().sum() >= 2000]
    if mkt_col not in valid:
        valid = [mkt_col] + valid
    sec_ret = sectors_raw[valid].ffill().dropna(subset=[mkt_col]).pct_change()
    mkt_ret = sec_ret[mkt_col].dropna()
    systemic_corr_daily = (sec_ret.drop(columns=[mkt_col], errors="ignore")
                           .rolling(window=60).corr(mkt_ret).mean(axis=1))

    if (dp / "ff_industry.csv").exists():
        ff_ind  = pd.read_csv(dp / "ff_industry.csv", index_col=0, parse_dates=True)
        ff_fact = pd.read_csv(dp / "ff_factors.csv",  index_col=0, parse_dates=True)
        patro_m = pd.DataFrame({
            "FinanceRet": ff_ind.iloc[:, cfg["ff_finance_col"]].astype(float),
            "MarketRet":  ff_fact.iloc[:, cfg["ff_market_col"]].astype(float),
        })
    else:
        fc = fin_col if fin_col in sectors_raw.columns and not sectors_raw[fin_col].isna().all() else valid[1]
        m_ = sectors_raw[[fc, mkt_col]].dropna(how="all").resample(_ME).last().pct_change().dropna()
        patro_m = pd.DataFrame({"FinanceRet": m_[fc], "MarketRet": m_[mkt_col]})
    patro_corr_m = patro_m["FinanceRet"].rolling(window=36, min_periods=12).corr(patro_m["MarketRet"])

    fc_db = fin_col if fin_col in sectors_raw.columns and not sectors_raw[fin_col].isna().all() else valid[1]
    db_df = sectors_raw[[fc_db, mkt_col]].dropna(how="all").ffill().pct_change().dropna()
    downside_beta = rolling_downside_beta(db_df[fc_db], db_df[mkt_col])

    r = df_daily["returns"]
    rm, rs = r.rolling(63).mean(), r.rolling(63).std()
    df_daily["RSkew"] = ((r - rm) ** 3).rolling(63).mean() / (rs ** 3)
    df_daily["RSV_Ratio"] = ((r.clip(upper=0) ** 2).rolling(63).sum()
                             / (r ** 2).rolling(63).sum().replace(0, np.nan))
    df_daily["HillTail"] = rolling_hill(r, window=252, k=12)
    neely = (df_daily[market] < df_daily[market].rolling(252).mean()).astype(int)

    return df_daily, {
        "hy": hy, "yc": yc, "yc3m": yc3m, "vix": vix, "ciss": ciss,
        "nfci": nfci, "epu": epu, "m2_growth": m2_growth, "sahm": sahm,
        "cape": cape, "cfnai": cfnai, "gdp": gdp_raw,
        "systemic_corr_d": systemic_corr_daily, "patro_corr_m": patro_corr_m,
        "downside_beta_d": downside_beta, "neely_d": neely,
    }


def _build_df_at_freq(market, df_daily, raws, freq, pandas_freq) -> pd.DataFrame:
    df = df_daily[[market, "returns"]].resample(pandas_freq).agg(
        {market: "last", "returns": "sum"}).dropna(subset=[market])

    df["HY_Spread"]   = _resample_last(raws["hy"],   pandas_freq).reindex(df.index, method="ffill")
    df["Yield_Curve"] = _resample_last(raws["yc"],   pandas_freq).reindex(df.index, method="ffill")
    df["YC_10Y3M"]    = _resample_last(raws["yc3m"], pandas_freq).reindex(df.index, method="ffill")
    df["VIX"]         = _resample_last(raws["vix"],  pandas_freq).reindex(df.index, method="ffill")
    df["CISS"]        = _resample_last(raws["ciss"], pandas_freq).reindex(df.index, method="ffill")
    df["NFCI"]        = _resample_last(raws["nfci"], pandas_freq).reindex(df.index, method="ffill")
    df["EPU"]         = _resample_last(raws["epu"],  pandas_freq).reindex(df.index, method="ffill")

    df["M2_Growth"]    = raws["m2_growth"].reindex(df.index, method="ffill")
    df["SahmIndicator"]= raws["sahm"].reindex(df.index, method="ffill")
    df["Shiller_CAPE"] = raws["cape"].reindex(df.index, method="ffill")
    df["CFNAI"]        = raws["cfnai"].reindex(df.index, method="ffill")

    df["GDP"] = raws["gdp"].reindex(df.index, method="ffill")
    df["Buffett"] = df[market] / df["GDP"]
    window_trend = {"W": 52 * 10, "M": 12 * 10, "D": 252 * 10}[freq]
    trend = df["Buffett"].rolling(window=window_trend, min_periods=max(window_trend // 5, 12)).mean()
    df["BuffettDeviation"] = (df["Buffett"] - trend) / trend

    for src_key, col in [("neely_d", "Neely"), ("downside_beta_d", "Downside_Beta"),
                         ("systemic_corr_d", "Systemic_Corr")]:
        if pandas_freq == "D":
            df[col] = raws[src_key].reindex(df.index).ffill()
        else:
            df[col] = raws[src_key].resample(pandas_freq).last().reindex(df.index).ffill()
    df["PatroCorr"] = raws["patro_corr_m"].reindex(df.index, method="ffill")
    for col in ["RSkew", "RSV_Ratio", "HillTail"]:
        if pandas_freq == "D":
            df[col] = df_daily[col].reindex(df.index).ffill()
        else:
            df[col] = df_daily[col].resample(pandas_freq).last().reindex(df.index).ffill()
    return df


def sweep_market_pre2021(market: str) -> pd.DataFrame:
    df_daily, raws = _load_market_daily(market, cutoff=CUTOFF)
    if df_daily.empty:
        print(f"    skipping {market}: no pre-2021 history")
        return pd.DataFrame(columns=["indicator", "p_value"])
    rows = []
    for dd in DRAWDOWN_THRESHOLDS:
        for horizon, freq in EXPERIMENT:
            pandas_freq = FREQ_MAP[freq]
            df = _build_df_at_freq(market, df_daily, raws, freq, pandas_freq)
            target_col = f"Crisis_next_{horizon}{freq}"
            df[target_col] = compute_crisis_target(df["returns"], horizon, dd)
            df_common = df.dropna(subset=INDICATOR_COLS + [target_col])
            if len(df_common) == 0:
                continue
            for ind in INDICATOR_COLS:
                p = fit_logit_one(df_common, ind, target_col)
                if p is not None:
                    rows.append({"indicator": ind, "p_value": p})
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────────
#  Comparison + figure
# ───────────────────────────────────────────────────────────────────────────

def _load_full() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(DATA / m / "results" / "results_sweep.csv")
                    for m in MARKETS], ignore_index=True)
    df = df[df["p_value"].notna()].copy()
    df["sig5"] = (df["p_value"] < 0.05).astype(int)
    return df


def main() -> None:
    print("Loading full-sample results from results_sweep.csv ...")
    full = _load_full()

    print("Re-fitting on the pre-2021 history (<= 2021-01-01) ...")
    parts = []
    for m in MARKETS:
        print(f"  pre-2021 sweep: {m}")
        parts.append(sweep_market_pre2021(m))
    recent = pd.concat(parts, ignore_index=True)
    recent["sig5"] = (recent["p_value"] < 0.05).astype(int)

    order = full.groupby("indicator")["p_value"].median().sort_values().index.tolist()
    full_g = full.groupby("indicator").agg(med_full=("p_value", "median"),
                                           sig_full=("sig5", "mean"))
    rec_g = recent.groupby("indicator").agg(med_rec=("p_value", "median"),
                                            sig_rec=("sig5", "mean"))
    comp = full_g.join(rec_g, how="left").reindex(order)
    comp.to_csv(OUT_CSV)
    print(f"Wrote: {OUT_CSV}")

    labels = [PLAIN[i] for i in comp.index]
    yy = np.arange(len(comp))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.0))

    ax = axes[0]
    ax.barh(yy - 0.2, comp["sig_full"] * 100, height=0.4, color="#3B6FB6", label="Full sample")
    ax.barh(yy + 0.2, comp["sig_rec"] * 100, height=0.4, color="#C53A3A", label="Pre-2021 (drop last 5y)")
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("Share significant at 5% (%)")
    ax.set_title("5%-significance share: full vs pre-2021")
    ax.legend(loc="lower right", fontsize=8.5); ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.barh(yy - 0.2, comp["med_full"] * 100, height=0.4, color="#3B6FB6", label="Full sample")
    ax.barh(yy + 0.2, comp["med_rec"] * 100, height=0.4, color="#C53A3A", label="Pre-2021 (drop last 5y)")
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8.5); ax.invert_yaxis()
    ax.axvline(5, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Median p-value (%)")
    ax.set_title("Median p-value: full vs pre-2021")
    ax.legend(loc="lower right", fontsize=8.5); ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Out-of-sample robustness: re-fit on the pre-2021 history "
                 "(dropping the most recent five years)", y=1.01, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")

    rho = comp[["med_full", "med_rec"]].corr(method="spearman").iloc[0, 1]
    print(f"  Spearman rho(full, pre-2021) on median p = {rho:.3f}")
    print(f"  Median p full / pre-2021 = {comp['med_full'].median():.4f} / "
          f"{comp['med_rec'].median():.4f}")


if __name__ == "__main__":
    main()
