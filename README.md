# 🧠 Market Regime Intelligence

> Unsupervised discovery, anomaly detection, Markov modeling, and supervised forecasting of financial market regimes.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Notebooks](https://img.shields.io/badge/Notebooks-4-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Overview

Financial markets cycle through distinct behavioral states — trending, mean-reverting, volatile, calm — each with its own risk/return profile. **Market Regime Intelligence** is an end-to-end pipeline that:

1. **Discovers** hidden market regimes from raw price data using unsupervised learning
2. **Validates** regimes by detecting anomalies within and across clusters
3. **Models** how regimes evolve over time using Markov chain theory
4. **Forecasts** the next regime using a tuned XGBoost classifier

The result is a framework for **regime-aware decision-making** — knowing not just where the market is, but where it's likely to go next.

---

## 🔄 Pipeline

```
Market Data (yfinance)
        │
        ▼
┌─────────────────────────┐
│  01 · Regime Discovery  │  KMeans + PCA → Labeled regime clusters
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  02 · Anomaly Detection │  Isolation Forest + LOF → Global & per-regime flags
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  03 · Markov Modeling   │  Transition matrix · Stationary dist · Expected durations
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  04 · Regime Forecast   │  XGBoost + Optuna → Predict next regime label
└─────────────────────────┘
```

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_Regime_Classification` | Identifies market regimes with **KMeans**. Performs statistical characterization of each cluster (mean returns, volatility, Sharpe). Includes **PCA** for 2D visualization of the regime space. |
| 02 | `02_Anomaly_Detection` | Detects structural outliers using **Isolation Forest** (global) and **LOF** (local). Anomalies are analyzed both across the full dataset and within each individual regime. |
| 03 | `03_Markov_Chain` | Fits a **discrete-time Markov chain** to the regime sequence. Outputs: transition matrix, stationary distribution, multi-step probabilities, and expected sojourn time per regime. |
| 04 | `04_XGBC_Regime_Prediction` | Trains an **XGBoost classifier** to predict the next-period regime. Hyperparameters are optimized with **Optuna** (TPE sampler). Reports accuracy, confusion matrix, and feature importance. |

---

## 📊 Data

Market data is fetched programmatically via [`yfinance`](https://github.com/ranaroussi/yfinance) — no manual downloads required.

Features engineered from raw OHLCV data include returns, rolling volatility, momentum indicators, and more (see notebooks for details).

---

## 🧪 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | `yfinance`, `pandas`, `numpy` |
| Statistics | `scipy` |
| ML / Clustering | `scikit-learn` (KMeans, IsolationForest, LOF, PCA) |
| Boosting | `xgboost` |
| Tuning | `optuna` |
| Visualization | `matplotlib`, `seaborn` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/JohnsoN98X/market-regime-intelligence.git
cd market-regime-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order

```
01_Regime_Classification   →   identifies regimes, saves labels
02_Anomaly_Detection       →   flags outliers per cluster
03_Markov_Chain            →   models regime transitions
04_XGBC_Regime_Prediction  →   trains forecasting model
```

> Each notebook is self-contained and saves intermediate outputs to `data/` so the next stage can pick up cleanly.

---

## 📁 Repository Structure

```
market-regime-intelligence/
│
├── notebooks/
│   ├── 01_Regime_Classification.ipynb
│   ├── 02_Anomaly_Detection.ipynb
│   ├── 03_Markov_Chain.ipynb
│   └── 04_XGBC_Regime_Prediction.ipynb
│
├── data/
│   ├── raw/          ← Downloaded market data
│   └── processed/    ← Feature-engineered datasets
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📈 Key Outputs

| Stage | Output |
|-------|--------|
| Regime Classification | Cluster labels + PCA scatter plot per regime |
| Anomaly Detection | Anomaly scores and flags (global & per-regime) |
| Markov Modeling | Transition heatmap, stationary distribution bar chart |
| Regime Forecasting | Classification report, confusion matrix, feature importance plot |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
Models and outputs are provided "as is" without warranty of any kind.
**Do not use this codebase as the sole basis for financial decisions.**
The author bears no responsibility for outcomes resulting from its use.
