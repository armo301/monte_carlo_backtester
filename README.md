# Monte Carlo Backtester

Monte Carlo simulator with multiple stochastic models (GBM, Jump-Diffusion, GARCH) for return simulation, as well as bootstrap model. 

Computes quantitative risk metrics, eg. Sharpe ratio, VaR, CVaR
Also returns return and max drawdown distributions.

Can use real data through yfinance or synthetic data by inputting mu (expected return) and sigma (volatility).

Provides 3 sample trading strategies: buy and hold (common baseline), SMA (momentum trading), mean reversion.
## Sample Usage

```bash
# Simulate AAPL using GARCH, 10,000 paths, one trading year
python -m cli.main --ticker AAPL --model garch --sims 10000 --days 252

python -m cli.main --ticker AAPL --model jump --sims 10000 --days 252 --strategy sma

# Synthetic mode (input mu and sigma)
python -m cli.main --mu 0.10 --sigma 0.20 --model gbm --sims 10000 --days 252

python -m cli.main --returns my_strategy.csv --model bootstrap --sims 10000

# See all options
python -m cli.main --help
```

## Setup

```bash
git clone https://github.com/armo301/monte-carlo-backtester.git
cd monte-carlo-backtester
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
