# Monte Carlo Backtester

A Monte Carlo simulation engine for backtesting trading strategies, built in Python.
Supports multiple stochastic models for return simulation and computes standard quantitative risk metrics.

## Models
- **Geometric Brownian Motion (GBM)** — continuous diffusion baseline (Black-Scholes)
- **Merton Jump-Diffusion** — GBM with Poisson-distributed price jumps for tail risk
- **GARCH(1,1)** — time-varying volatility with clustering effects
- **Historical Bootstrap** — non-parametric resampling of empirical return distributions

## Risk Metrics
- **Sharpe Ratio**
- **Value at Risk (VaR)** — parametric and historical
- **Conditional VaR (CVaR / Expected Shortfall)**
- **Maximum Drawdown**

## Data Sources
- **Live data**: Yahoo Finance via `yfinance`
- **Synthetic**: Parameterized return generation for model testing

## Usage
```bash
python -m cli.main --ticker AAPL --model gbm --sims 10000 --days 252
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
monte-carlo-backtester/
├── data/           # Data fetching and synthetic generation
├── models/         # Stochastic return models
├── simulation/     # Monte Carlo engine and risk metrics
├── backtest/       # Strategy runner and base strategy class
├── cli/            # Command-line entry point
├── plots/          # Visualization
└── tests/          # Unit tests

## Background
This project was built to explore stochastic modeling of financial returns.
The mathematical foundation is Itô calculus — specifically the connection between
physical Brownian motion and geometric diffusion processes in asset pricing.
