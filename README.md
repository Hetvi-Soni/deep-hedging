# Deep Hedging with Market Frictions

**AMS 520 — Machine Learning in Quantitative Finance, Fall 2026**

Classical option hedging (the Black–Scholes delta) assumes continuous, cost-free trading and a correct model.
This project trains a neural-network hedging policy that minimizes the tail risk (CVaR) of hedging P&L under
transaction costs, following Buehler et al. (2019), and tests it on real historical S&P 500 price paths.

## Project phases

| Phase | Weight | Scope | Status |
|---|---|---|---|
| I | 20% | Literature survey, data collection, data cleaning | Done |
| II | 40% | Exploratory data analysis, calibration, first hedging results | Planned |
| III | 40% | Out-of-model testing, no-transaction band network, RL formulation | Planned |

## Data (free, no credentials needed)

- **SPY** daily adjusted prices — Yahoo Finance via `yfinance`
- **VIX** index — Yahoo Finance (`^VIX`)
- **3-month T-bill rate** (`DTB3`) — FRED, Federal Reserve Bank of St. Louis

Sample period: January 2005 to September 2026. Train/calibration period ends 2018-12-31; later data is kept for testing.

## Repository structure

```
Phase1_Data_Collection_and_Cleaning.ipynb   Phase I report: collection, cleaning, checks, plots
data/raw/                                   untouched downloads
data/processed/                             cleaned daily data, hedging windows, cleaning log
figures/                                    plots used in the presentation
```

## How to run

Open the notebook in Google Colab or Jupyter and run all cells. The data downloads automatically.

```
pip install yfinance pandas numpy scipy matplotlib
```

## References

1. Buehler, Gonon, Teichmann, Wood (2019). *Deep Hedging.* arXiv:1802.03042
2. Imaki et al. (2021). *No-Transaction Band Network.* arXiv:2103.01775
3. Buehler, Murray, Wood (2022). *Deep Bellman Hedging.* arXiv:2207.00932
4. Heston (1993). *A closed-form solution for options with stochastic volatility.* Review of Financial Studies.
5. Whalley, Wilmott (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.* Mathematical Finance.
