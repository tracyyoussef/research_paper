# Crisis Indicators for Equity Markets

Code and data for the master's thesis *Crisis Indicators for Equity
Markets: Which Signals Actually Predict Drawdowns?*

The study evaluates nineteen published crisis-prediction indicators
(volatility, tail-risk, credit, valuation, macroeconomic and
systemic-risk) across eight international equity markets, sweeping six
drawdown thresholds (3 % to 15 %) and eleven forecasting horizons
(5 day to 24 month) — around ten thousand univariate logistic
regressions in total, ranked by the median *p*-value of the slope
coefficient.

## Repository layout

```
.
├── README.md
├── environment.yml                  conda environment (reproducible setup)
├── crisis_indicator_sweep.ipynb     main notebook: data prep + per-market sweep
├── analysis/
│   ├── analyze_and_plot.py          pools per-market sweeps -> tables + figures
│   ├── hill_tail_diagnostic.py      robustness check: Hill k-sweep
│   ├── logit_vs_probit.py           robustness check: Logit vs Probit
│   ├── out_of_sample.py             robustness check: pre-2021 re-fit
│   └── tables/                      aggregated summary CSVs
├── data/                            per-market raw inputs (index + indicators)
│   ├── SP500/ DAX/ Nikkei/ EuroStoxx/
│   └── FTSE100/ CAC40/ ASX200/ China/   (the eight markets analysed)
└── figures/                         generated figures
    ├── markets/                     per-market heatmaps
    └── diagnostics/                 robustness figures
```

## Reproducing the results

### 1. Per-market sweep

`crisis_indicator_sweep.ipynb` is the main code. Open it and run every cell in order.
The `MARKET` variable at the top of the "Grid config" cell selects the
market; set it to each of the eight markets in turn (SP500, DAX,
Nikkei, EuroStoxx, FTSE100, CAC40, ASX200, China) and re-run the sweep
cell. Each run writes `data/<MARKET>/results/results_sweep.csv`.

### 2. Aggregation, figures and robustness checks

Once every market's `results_sweep.csv` exists:

```bash
python3 analysis/analyze_and_plot.py         # summary tables + main figures
python3 analysis/hill_tail_diagnostic.py     # robustness: Hill k-sweep
python3 analysis/logit_vs_probit.py          # robustness: Logit vs Probit
python3 analysis/out_of_sample.py            # robustness: pre-2021 re-fit
```

`out_of_sample.py` re-runs the regression grid on the pre-2021 history
only, so it is self-contained and does not depend on the notebook.

## Methodology

The crisis target is a binary forward-looking event. At the target
frequency (daily, weekly, or monthly), let `r_t` be the period return
and `h` the horizon in periods:

```
future_cum = returns.rolling(h).sum().shift(-h)
y_t        = 1   if future_cum <= -delta
             0   otherwise
             NA  if the future is unobserved (last h rows)
```

For each (market x indicator x delta x horizon) we fit a univariate
Logit with a constant:

```
P(y_t = 1 | x_t) = Lambda(alpha + beta * x_t),  Lambda(z) = 1 / (1 + exp(-z))
```

and record the slope p-value, the McFadden pseudo-R^2, the odds ratio,
and the Wald z-statistic. Indicators are ranked by the median p-value
across the grid; the median is preferred over the mean because the
p-value distribution is heavily right-skewed.

Daily indicators (HY spread, yield curves, VIX, CISS, NFCI, EPU) are
aggregated to weekly/monthly frequency by the last observation within
the period; flow-type series (M2, GDP, CFNAI, Sahm rule, Shiller CAPE)
are forward-filled.

## Indicators evaluated

| Family | Indicators |
|---|---|
| Credit & Term-Structure | HY\_Spread, Yield\_Curve, YC\_10Y3M |
| Macro & Real-Activity   | SahmIndicator, M2\_Growth, CFNAI, EPU |
| Valuation & Trend       | Shiller\_CAPE, BuffettDeviation, Neely |
| Volatility & Tail-Risk  | VIX, RSkew, RSV\_Ratio, HillTail |
| Systemic-Risk & Contagion | CISS, NFCI, PatroCorr, Billio\_Corr, Downside\_Beta |

## Software

The recommended workflow uses `conda` and the `environment.yml` at the
repo root:

```bash
conda env create -f environment.yml
conda activate crisis-indicators
```

This creates an isolated environment with Python 3.11 and pinned
versions of pandas (>= 2.2), numpy (>= 1.26), statsmodels (>= 0.14),
scipy (>= 1.13), matplotlib (>= 3.8), JupyterLab and ipykernel, plus
`yfinance` and `pandas-datareader` for the optional data-download cell
in the notebook.

Without conda, `pip install pandas numpy statsmodels scipy matplotlib
jupyterlab ipykernel` is sufficient to run everything except the
optional download cell.

### Using the environment in VS Code

After `conda env create -f environment.yml`, open the project folder in
VS Code, open `crisis_indicator_sweep.ipynb`, click the kernel selector (top-right),
choose "Select Another Kernel..." -> "Python Environments" and pick
**crisis-indicators**.
