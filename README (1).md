# Deep Hedging with Market Frictions

**AMS 520 — Machine Learning in Quantitative Finance, Fall 2026 · Hetvi Soni**

Classical option hedging (the Black–Scholes delta) assumes continuous, cost-free trading and a correct model.
This project trains a neural-network hedging policy that minimizes the tail risk (CVaR) of hedging P&L under
transaction costs, following Buehler et al. (2019), and tests it on simulated markets, real S&P 500 price paths,
and real S&P 500 (SPX) options.

## Project phases

| Phase | Weight | Scope | Status |
|---|---|---|---|
| I | 20% | Literature survey, data collection, data cleaning | Done |
| II | 40% | Exploratory data analysis, calibration, deep hedging results, extended analysis, real SPX options | Done |
| III | 40% | Implied-volatility inputs, no-transaction band network, RL formulation, data-driven simulator | Planned |

## Phase II key findings

- **Sanity check passed:** with no trading costs the network rediscovers the Black–Scholes delta (average gap 0.028 shares).
- **Simulated markets:** deep hedging lowers tail risk (CVaR 95%) versus Black–Scholes in every case, statistically significant
  across 5 training seeds — up to **12.9%** (GBM) and **3.4%** (Heston) at 0.5% transaction cost — mainly by trading less.
- **Stronger benchmarks:** without being given any formula, the network matches the classical cost-aware Whalley–Wilmott
  hedge and beats it under GBM.
- **Real SPY paths (2019+):** the improvement over Black–Scholes is statistically significant only at high costs; the
  GBM-trained hedger had the smallest tail losses in the 2020 COVID crash and the 2022 sell-off.
- **Real SPX options (92 monthly contracts, Databento OPRA):** hedging cuts tail risk 3–6× versus no hedge; with the same
  volatility information (VIX), deep hedging lowers CVaR by 8–13% versus Black–Scholes. The best overall strategy was
  Whalley–Wilmott with the option's own implied volatility, which motivates adding implied volatility as a network input in Phase III.

## Data

- **SPY** daily adjusted prices and **S&P 500 index** (`^GSPC`) — Yahoo Finance via `yfinance`
- **VIX** index — Yahoo Finance (`^VIX`), calculated by Cboe
- **3-month T-bill rate** (`DTB3`) — FRED, Federal Reserve Bank of St. Louis
- **SPX option quotes and contract definitions** — Databento, OPRA dataset (`OPRA.PILLAR`). Raw option data is **not included**
  in this repository (data licence); `Phase2b_Real_SPX_Options.ipynb` re-downloads it with your own Databento API key
  (cost about $1 of the free credit).

Sample period: January 2005 to September 2026. Calibration/training period ends 2018-12-31; later data is kept for testing.

## Repository structure

```
Phase1_Data_Collection_and_Cleaning.ipynb   Phase I: collection, cleaning, checks, plots
Phase2_EDA_and_Deep_Hedging.ipynb           Phase II: EDA, calibration, deep hedging, extended analysis
Phase2b_Real_SPX_Options.ipynb              Phase II: hedging real SPX options with real premiums
data/processed/                             cleaned daily data, hedging windows, cleaning log
figures/                                    plots used in the presentations
results/                                    result tables, summaries, trained networks (policy_*.pt)
```

## How to run

Open the notebooks in Google Colab (GPU recommended for Phase II) and run all cells in order.

```
pip install yfinance pandas numpy scipy matplotlib torch databento
```

- Phase II needs `data/processed/market_daily_clean.csv` from Phase I.
- Phase 2b also needs the trained networks in `results/` and a Databento API key stored as a Colab secret named
  `DATABENTO_API_KEY` (never in the notebook). Start with `DRY_RUN = True` to see the cost before downloading.

## References

1. Buehler, Gonon, Teichmann, Wood (2019). *Deep Hedging.* Quantitative Finance. arXiv:1802.03042
2. Imaki et al. (2021). *No-Transaction Band Network.* arXiv:2103.01775
3. Buehler, Murray, Wood (2022). *Deep Bellman Hedging.* arXiv:2207.00932
4. Black, Scholes (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
5. Heston (1993). *A closed-form solution for options with stochastic volatility.* Review of Financial Studies.
6. Leland (1985). *Option pricing and replication with transaction costs.* Journal of Finance.
7. Whalley, Wilmott (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.* Mathematical Finance.
8. Rockafellar, Uryasev (2000). *Optimization of conditional value-at-risk.* Journal of Risk.
9. Cont (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance.
