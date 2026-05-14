# ARIMA-Based Stock Price Prediction & Trading Performance Evaluation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![ARIMA](https://img.shields.io/badge/Statsmodels-ARIMA-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**Can time series models predict stock price direction consistently enough to generate trading profits?** This project provides an empirical answer by evaluating ARIMA and LSTM models across six Indian stocks, comparing statistical accuracy against real trading outcomes.

---

## 🔍 What Makes This Project Unique

- **Honest failure analysis:** Documents where models perform worse than random guessing
- **Volatility-performance mapping:** Quantifies how volatility affects predictability
- **Real trading validation:** Compares model signals against actual StockGro execution data
- **Cost-aware evaluation:** Transaction costs explicitly modeled

Most projects stop at RMSE. This one explains why low error doesn't mean profitable trading.

---

## 📋 Problem Statement

Financial markets are non-stationary, volatile, and influenced by unpredictable events. Traditional time series models assume stability — a mismatch that limits real-world applicability.

**Core questions addressed:**

1. Can ARIMA and LSTM predict price **direction** (not just levels) consistently?
2. How does volatility affect prediction accuracy across sectors?
3. Do statistically accurate forecasts translate into trading profits after costs?

**Time Period:** January 2021 – December 2025  
**Train Set:** January 2021 – June 2025 (1,172 days)  
**Test Set:** July 2025 – December 2025 (131 days)  
**Stocks:** Oil India, Quick Heal, Jindal Drilling, ICICI Bank, Tata Steel, Sun Pharma

---
## 📂 Repository & Notebook Links

| Component | Link |
|:----------|:-----|
| **Main Analysis Notebook** | [![Kaggle](https://img.shields.io/badge/Kaggle-ARIMA_Prediction-20BEFF?logo=kaggle)](https://www.kaggle.com/code/mrrogueknight/arima-based-stock-price-prediction-and-performance) |
| **Interactive Dashboard** | [![Kaggle](https://img.shields.io/badge/Kaggle-Performance_Dashboard-20BEFF?logo=kaggle)](https://www.kaggle.com/code/mrrogueknight/arima-stock-forecasting-performance-dashboard) |
| **Full Project Repository** | [![GitHub](https://img.shields.io/badge/GitHub-Complete_Project-181717?logo=github)](https://github.com/MrRogueKnight/arima-stock-forecasting-and-performance-evaluation) |
| **Detailed Project Report (PDF)** | [![Report](https://img.shields.io/badge/PDF-Detailed_Report-e74c3c?logo=adobeacrobatreader)](https://github.com/MrRogueKnight/arima-stock-forecasting-and-performance-evaluation/blob/rogue/Report%20on%20Evaluating%20the%20Predictive%20Power%20of%20Time%20Series%20Models%20for%20Stock%20Price%20Forecasting%20and%20Trading%20Performance.pdf) |

**Navigation Guide:**
- **Main Analysis Notebook** → Complete ARIMA/LSTM modeling, backtesting, and evaluation
- **Interactive Dashboard** → Visual dashboard loading outputs from the main notebook
- **GitHub Repository** → Full source code, PDF report, and documentation
- **Detailed Project Report (PDF)** → Comprehensive 5,000+ word academic report
---
## 📊 Data & Setup

### Stock Selection (Sector-Diversified)

| Stock | Sector | Volatility (σ) |
|-------|--------|----------------|
| Jindal Drilling | Energy | 0.031 |
| Quick Heal | Technology | 0.030 |
| Oil India | Energy | 0.025 |
| Tata Steel | Metals | 0.021 |
| ICICI Bank | Banking | 0.014 |
| Sun Pharma | Pharmaceuticals | 0.013 |

**Source:** Yahoo Finance (`yfinance`)  
**Target:** Adjusted Close Price  
**Preprocessing:** Log returns, forward-fill missing values, business day alignment

**Stationarity check (ADF test):**
- Price levels: Non-stationary (p > 0.05)
- Log returns: Stationary (p < 0.001)
- **Conclusion:** Differencing order d=1 required

---

## 🧠 Modeling Approach

### ARIMA (1,1,1)

ARIMA captures linear dependencies through three parameters:

- **p=1 (autoregressive):** Uses one past value
- **d=1 (differencing):** Removes trend (confirmed by ADF)
- **q=1 (moving average):** Corrects one lag of forecast errors

**Model equation:**
```
ΔPrice_t = φ₁·ΔPrice_{t-1} + θ₁·ε_{t-1} + ε_t
```

Model selection used AIC; (1,1,1) was optimal or near-optimal for all stocks.

### LSTM

LSTM handles non-linear patterns with:
- **Sequence length:** 20 days
- **Architecture:** 2 LSTM layers (64→32 units) + Dropout (0.2)
- **Training:** Walk-forward validation (no leakage)
- **Scaling:** Min-Max normalization

### Ensemble

| Model | Weight | Rationale |
|-------|--------|-----------|
| ARIMA | 40% | Stable trend capture |
| LSTM | 60% | Non-linear pattern recognition |

Weights are fixed based on validation performance.

---

## 📉 Results & Evaluation

### Performance Metrics

| Stock | RMSE | MAPE (%) | Direction Accuracy (%) |
|-------|------|----------|------------------------|
| Tata Steel | 5.80 | 2.68 | 44.6 |
| Oil India | 14.62 | 2.99 | 45.4 |
| Quick Heal | 21.97 | 6.12 | 46.2 |
| ICICI Bank | 27.26 | 1.58 | 40.0 |
| Jindal Drilling | 33.89 | 5.02 | 49.2 |
| Sun Pharma | 49.12 | 2.33 | 49.2 |
| **Average** | **25.44** | **3.45** | **45.8** |

### Critical Interpretation

> **Average directional accuracy (45.8%) is marginally below or near random guessing (~50%).**

Even with low MAPE (e.g., Sun Pharma at 2.33%), the model fails to consistently predict price movement direction. This is the core limitation: **price level accuracy does not imply directional trading ability.**

**Why this matters:**

In financial modeling, a model can achieve low prediction error while still being unusable for trading.  
This project demonstrates that **directional accuracy—not price error—is the true bottleneck** in building profitable strategies.

This suggests that the limitation is not just model choice, but the inherently low signal-to-noise ratio in financial time series.

### Residual Analysis (Ljung-Box Test)

```
Residuals fail the white-noise assumption (Ljung-Box p < 0.05), indicating remaining autocorrelation and unmodeled structure.
```

**Interpretation:** The model missed systematic patterns in the data. This explains the poor directional accuracy — predictable information remains unmodeled.

### Volatility Alone Does Not Explain Directional Failure

| Volatility Level | Average Direction Accuracy |
|-----------------|---------------------------|
| Low (σ < 0.015) | ~45% |
| Medium (σ 0.020-0.025) | ~45% |
| High (σ > 0.028) | ~47% |

The model failed to generate a consistent edge across all regimes. Differences across volatility regimes were not strong enough to produce a consistent predictive edge.

---

## 💰 Trading Performance

### Model-Based Backtest

| Metric | Value |
|--------|-------|
| Initial Capital | ₹100,000 |
| Final Capital | ₹99,942 |
| Total Return | -0.06% |
| Buy & Hold Return (avg) | -7.11% |
| **Relative Outperformance** | **+7.05%** |

The model-based strategy lost less than buy-and-hold but still failed to generate positive returns.

### Actual StockGro Trading Results

| Metric | Value |
|--------|-------|
| Gross Profit | ₹10,903 |
| Transaction Charges | ₹4,733 (43% of gross) |
| **Net Profit** | **₹6,170** |

**Note on trading period:**

Model predictions were generated for the full test period (July–December 2025).  
However, actual trading was executed only on selected days (May 11–12, conducted as a separate live trading validation after model development).

This distinction is important:
- Model results reflect **systematic backtesting**
- Trading results reflect **real-world execution under limited opportunities**

Therefore, trading performance should be interpreted as **practical validation**, not a continuous model-driven strategy.

**Key observation:** The actual trading net profit (₹6,170) significantly exceeded model-based backtest results (-₹58). This gap highlights that **execution decisions, timing, and discretionary judgment contributed more to profitability than model-generated signals.**

This highlights that human-in-the-loop decision making can outperform purely model-driven execution when signal quality is low and noise dominates.

### Winner vs Loser (Direction Accuracy)

| Stock | Direction Accuracy | Classification |
|-------|-------------------|----------------|
| Sun Pharma | 61.0% | Winner |
| Tata Steel | 50.0% | Neutral |
| Oil India | 49.2% | Loser |
| Jindal Drilling | 43.2% | Loser |
| ICICI Bank | 41.5% | Loser |
| Quick Heal | 40.7% | Loser |

Only Sun Pharma provided a genuine predictive edge (>55% accuracy).

---

## 💡 Key Insights

### 1. Price Accuracy is a Misleading Metric

Sun Pharma had 2.33% MAPE (excellent) but 49% direction accuracy (random). **Low error does not mean tradable signal.**

### 2. Real Trading Outperformed Model Signals

The ₹6,170 net profit came from execution decisions, not pure model following. The model provided context, not commands.

### 3. Transaction Costs are Not Trivial

At 43% of gross profit, costs dominated outcomes. High-frequency signal generation is economically unsustainable.

### 4. The Gap Between Statistical and Trading Success

This project highlights the gap between statistical accuracy and trading usefulness — a critical distinction often overlooked in financial modeling.

---

## 📈 When Should You Use This Model?

- For **trend estimation**, not precise timing
- As a **supporting signal** alongside fundamentals or sentiment
- In **low-volatility environments** where price movements are smoother
- For **risk context**, not decision automation

## 📉 When It Fails

- Short-term trading where direction accuracy is critical
- Regime shifts or sudden market shocks
- High-noise environments with weak signal-to-noise ratio
- When signal-to-noise ratio is low (most real-world financial data)
- Any strategy relying purely on model predictions without filtering

### Practical Takeaway for Traders

> **Use ARIMA-LSTM ensembles for price level context, not trading signals. Combine with volatility filtering and strict cost controls. Never rely on direction predictions alone.**

---

## ⚠️ Limitations

| Limitation | Impact |
|------------|--------|
| No exogenous variables | Misses news, macro, sentiment |
| Linear ARIMA component | Cannot capture regime changes |
| Fixed ensemble weights | Not adaptive to market conditions |
| High transaction cost assumption | May overstate cost impact |
| Small stock universe (n=6) | Limits generalizability |

---

## 🚀 Future Improvements

**Short-term:**
- Add GARCH for volatility forecasting
- Dynamic ensemble weighting based on recent performance
- Reduce trading frequency to manage costs

**Medium-term:**
- Incorporate sector ETFs and Nifty as features
- Binary classification (up/down) instead of price regression
- Expand stock universe to 20+ securities

**Long-term:**
- Transformer architectures for sequence modeling
- Reinforcement learning for execution optimization
- Sentiment integration from news/social media

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Collection | `yfinance` |
| Processing | `pandas`, `numpy` |
| Statistics | `statsmodels` (ADF, ARIMA, Ljung-Box) |
| Deep Learning | `tensorflow`, `keras` (LSTM) |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Jupyter Notebook, Kaggle |

---

## 📝 Conclusion

**What this project proved:**

- ARIMA-LSTM ensembles provide reasonable price level estimates (MAPE typically ~2–6%, depending on stock volatility)
- They cannot consistently predict direction (accuracy near random, ~40-50%)
- Volatility alone does not explain directional failure — the problem is structural
- Trading success depends more on execution and costs than model accuracy

**The core limitation is not technical but fundamental:** Financial markets contain patterns that linear and even simple non-linear models cannot capture. Directional prediction remains unsolved.

**Final takeaway:**

> This project reinforces a fundamental reality: financial markets are not just noisy — they are adversarial. Simple statistical models can extract structure, but not enough to sustain a trading edge without additional context, filtering, and disciplined execution.

This project demonstrates the gap between theoretical modeling and practical trading — a key challenge in applied financial machine learning.

At its core, the limitation is not computational but informational: financial markets contain weak, unstable signals embedded in high noise, making consistent directional prediction inherently difficult.

---

## 📬 Connect

**Author:** Prashant Ranjan  
**GitHub:** [MrRogueKnight](https://github.com/MrRogueKnight)

---

*This project was completed as a Time Series Analysis Capstone. Code and analysis are open for learning and improvement.*
