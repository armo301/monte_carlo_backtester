import numpy as np
import pandas as pd
import yfinance as yf

def fetch_returns(ticker: str, period: str = '2y') -> tuple:
    """
    Fetches historical price data from Yahoo Finance and computes
    log returns

    Parameters:
        ticker : str -> stock ticker symbol (e.g. 'AAPL', 'TSLA', 'SPY')
    period : str -> lookback period. Valid values: '1y', '2y', '5y', '10y', 'max'

    Returns:
        tuple : (returns, prices, info)
        returns : np.ndarray of log returns
        prices  : np.ndarray of closing prices
        info    : dict with ticker metadata
    """
    print(f"Fetching {period} of data for {ticker}...")

    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period)

    if hist.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. "
                         f"Check that the ticker symbol is valid.")

    prices = hist['Close'].values
    returns = np.log(prices[1:] / prices[:-1])

    info = {
            'ticker':      ticker.upper(),
            'period':      period,
            'n_days':      len(returns),
            'start_date':  hist.index[0].strftime('%Y-%m-%d'),
            'end_date':    hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': prices[0],
            'end_price':   prices[-1],
            'mean_return': np.mean(returns),
            'volatility':  np.std(returns, ddof=1) * np.sqrt(252)
        }

    print(f"Retrieved {info['n_days']} trading days "
          f"({info['start_date']} to {info['end_date']})")
    print(f"Current price: ${info['end_price']:.2f} | "
          f"Annualized vol: {info['volatility']:.2%}")

    return returns, prices, info


def generate_synthetic_returns(mu: float, sigma: float,
                                n_days: int = 504,
                                S0: float = 100.0,
                                dt: float = 1/252) -> tuple:
    """
    Generates synthetic log returns from a normal distribution
    Used for testing models without real data

    Parameters:
        mu : float -> annualized drift (e.g. 0.10 for 10%)
        sigma : float -> annualized volatility (e.g. 0.20 for 20%)
        n_days : int -> number of trading days to generate. Default 504 (2 years)
        dt : float -> time step in years.

    Returns:
        tuple : (returns, prices, info) 
            -> Same structure as fetch_returns for interchangeability
    """
    daily_mu = mu * dt
    daily_sigma = sigma * np.sqrt(dt)
    returns = np.random.normal(daily_mu, daily_sigma, n_days)

    # Build synthetic price series starting at $100
    prices = np.empty(n_days + 1)
    prices[0] = S0
    for i in range(1, n_days + 1):
        prices[i] = prices[i-1] * np.exp(returns[i-1])

    info = {
        'ticker':      'SYNTHETIC',
        'period':      f'{n_days} days',
        'n_days':      n_days,
        'start_date':  'N/A',
        'end_date':    'N/A',
        'start_price': S0,
        'end_price':   prices[-1],
        'mean_return': np.mean(returns),
        'volatility':  np.std(returns, ddof=1) * np.sqrt(252)
    }

    return returns, prices, info

def load_returns_from_csv(filepath: str) -> tuple:
    """
    Loads strategy returns from a CSV file for Use Case 2.
    CSV should have one column of decimal returns (e.g. 0.0082, -0.0043).

    Parameters:
        filepath: str -> path to CSV file.

    Returns:
        tuple: (returns, info)
        returns: np.ndarray of log returns
        info: dict with metadata
    """
    df = pd.read_csv(filepath, header=None)
    returns = df.iloc[:, 0].values.astype(float)

    info = {
        'ticker':      filepath.split('/')[-1].replace('.csv', '').upper(),
        'period':      f'{len(returns)} days',
        'n_days':      len(returns),
        'start_date':  'N/A',
        'end_date':    'N/A',
        'start_price': 100.0,
        'end_price':   100.0,
        'mean_return': np.mean(returns),
        'volatility':  np.std(returns, ddof=1) * np.sqrt(252)
    }

    print(f"Loaded {len(returns)} days of returns from {filepath}")
    print(f"Mean return: {info['mean_return']:.6f} | "
          f"Annualized vol: {info['volatility']:.2%}")

    return returns, info