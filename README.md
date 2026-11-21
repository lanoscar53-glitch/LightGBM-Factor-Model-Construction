# LightGBM-Factor-Model-Construction

This is a report hoping to exam the actual effect of pure machine learning in current A stock market with data obtained from Tushare ranging from 2018.1 to 2026.10

## Final Result (2020-01 ~ 2026-10)

Walk-forward LightGBM-Bayes stock selection: **monthly retraining + monthly Bayesian hyperparameter retuning** on a trailing 12-month window (lagged 13 months for label realization, no look-ahead), 11 factors (5 kept from the referenced paper + 6 significant ones from our 16-factor system), Top20 constrained Markowitz (monthly, with industry/style/beta box constraints) and Top10 equal-weight (quarterly) portfolios, fee 1bp/side:

![final result](figures/wealth_paperv2_2020wf.png)

| Portfolio | Ann. Return | Sharpe | Max Drawdown | Win Rate | Rank IC (mean) |
|---|---|---|---|---|---|
| LGBv2_mw (Top20 Markowitz) | **16.4%** | **0.72** | -19.6% | 58.0% | 0.0416 (ICIR 0.39, t=3.52) |
| LGBv2_ew10 (Top10 equal-weight) | 6.7% | 0.38 | -29.3% | 53.1% | 0.0416 (ICIR 0.39, t=3.52) |
| CSI 300 (benchmark) | 1.4% | 0.17 | -40.6% | 50.6% | — |

GRS test: stat=7.69, p=0.001 — the linear factor model cannot explain portfolio returns (significant non-linear alpha), consistent with the referenced paper's claim of non-linear factor interactions captured by tree ensembles.

## Repository Structure

```
├── paper/     referenced paper (PDF)
├── src/       backtest code (walk-forward version)
├── results/   monthly returns (CSV) / holdings & style exposure (JSON) / metrics (JSON)
├── figures/   NAV chart with core-metric annotations
└── reports/   summary (MD) + factor definitions & mathematical forms (DOCX)
```

## Key Documents

- **Factor definitions & mathematical forms**: [reports/paper_v2因子计算方式与数学形式.docx](reports/paper_v2因子计算方式与数学形式.docx)
- **Methodology & full result summary**: [reports/backtest_report_paperv2_2020wf.md](reports/backtest_report_paperv2_2020wf.md)

## Methodology in Brief

- Universe: all A-shares ex ST / Beijing Stock Exchange / new listings (<370 days) / suspended; PIT financials with ann_date+1 enforcement (first-disclosure versions to avoid restatement look-ahead)
- Prices: post-adjusted (hfq), executed at t+1 open with one-word limit-up/down rules
- Label: sign of forward 12-month post-adjusted return (referenced-paper convention); training window strictly lags signal month by 13 months so all training labels are fully realized
- Hyperparameters: re-tuned monthly by Bayesian optimization (15 gp_minimize calls on a 12k subsample, 3-fold CV) alongside the monthly model retrain
- Data source: local quantitative database built from Tushare/Sina; monthly inputs cached as parquet. Full run takes ~22 min on cached data

*Referenced paper: Li, Xu, Li (2022), Procedia Computer Science 214:1234-1240 — 5-minute price-volume factors + LightGBM stock selection. Reproduced here at daily/monthly factor frequency with the paper's kept factors plus extensions.*
