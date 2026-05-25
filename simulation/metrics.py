import numpy as np


def compute_returns(paths: np.ndarray) -> np.ndarray:
    """
    Computes simple returns for each simulated path
    return_i = (final_price_i - start_price_i) / start_price_i

    Parameters:
        paths : np.ndarray -> shape (n_sims, n_steps + 1), each row is one simulated path

    Returns:
    np.ndarray -> shape (n_sims,), one return per simulation
    """
    return (paths[:, -1] - paths[:, 0]) / paths[:, 0]


def var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Computes Value at Risk at the given confidence level

    VaR is quoted as a positive number representing a loss
    e.g. VaR of 0.15 means a 15% loss at the given confidence level.

    Parameters
        returns : np.ndarray -> array of simulated returns (one per simulation)
        confidence : float -> confidence level (e.g. 0.95 for 95% VaR)

    Returns:
        float -> VaR as a positive loss percentage.
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Computes Conditional VaR (Expected Shortfall) at the given confidence level.

    Always >= VaR at the same confidence level.

    Parameters:
        returns : np.ndarray -> array of simulated returns (one per simulation)
        confidence : float -> confidence level (e.g. 0.95 for 95% CVaR)

    Returns:
        float -> CVaR as a positive loss percentage
    """
    var_threshold = -var(returns, confidence)
    tail_returns = returns[returns < var_threshold]

    if len(tail_returns) == 0:
        return var(returns, confidence)

    return -np.mean(tail_returns)


def max_drawdown(path: np.ndarray) -> float:
    """
    Computes maximum drawdown for a single price path
    Max drawdown = largest peak to trough decline as a fraction of peak

    Parameters:
    path : np.ndarray -> shape (n_steps + 1,). A single simulated price path

    Returns:
        float -> max drawdown as a positive fraction (e.g. 0.35 = 35% drawdown)
    """
    peak = np.maximum.accumulate(path)
    drawdowns = (peak - path) / peak
    return np.max(drawdowns)


def max_drawdown_distribution(paths: np.ndarray) -> dict:
    """
    Computes max drawdown for every simulated path and returns
    summary statistics of the distribution

    Parameters:
        paths : np.ndarray -> shape (n_sims, n_steps + 1)

    Returns:
        dict -> summary statistics of the max drawdown distribution
    """
    drawdowns = np.array([max_drawdown(path) for path in paths])

    return {
        'mean':   np.mean(drawdowns),
        'median': np.median(drawdowns),
        'p5':     np.percentile(drawdowns, 5),
        'p95':    np.percentile(drawdowns, 95),
        'worst':  np.max(drawdowns)
    }


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.05,
                 dt: float = 1/252) -> float:
    """
    Computes annualized Sharpe Ratio from a sequence of period returns
    Used when a trading strategy produces a time series of returns

    Sharpe = (mean_return - risk_free_rate) / std_return
    Both mean and std are annualized

    Parameters:
        returns : np.ndarray
            ->Time series of period returns (e.g. daily returns from a strategy)
        risk_free_rate : float 
            ->Annualized risk free rate. Default 0.05 (5% ~ current T-bill rate)
        dt : float -> Length of one period in years. Default 1/252 (one trading day)

    Returns:
        float -> annualized Sharpe ratio
    """
    mean_return = np.mean(returns) / dt
    std_return = np.std(returns, ddof=1) / np.sqrt(dt)
    return (mean_return - risk_free_rate) / std_return


def summarize(paths: np.ndarray, risk_free_rate: float = 0.05,
              confidence: float = 0.95) -> dict:
    """
    Computes all path-level metrics at once and returns a summary dict
    This is the main function called by the simulation engine

    Parameters:
        paths : np.ndarray -> shape (n_sims, n_steps + 1)
        risk_free_rate : float -> annualized risk free rate for Sharpe computation
        confidence : float -> confidence level for VaR and CVaR

    Returns:
        dict -> all metrics in one place
    """
    returns = compute_returns(paths)
    dd = max_drawdown_distribution(paths)

    return {
        'mean_return':    np.mean(returns),
        'median_return':  np.median(returns),
        'std_return':     np.std(returns),
        'var':            var(returns, confidence),
        'cvar':           cvar(returns, confidence),
        'sharpe':         sharpe_ratio(returns, risk_free_rate),
        'max_drawdown': {
            'mean':   dd['mean'],
            'median': dd['median'],
            'p5':     dd['p5'],
            'p95':    dd['p95'],
            'worst':  dd['worst']
        },
        'prob_profit':    np.mean(returns > 0),
        'p5_return':      np.percentile(returns, 5),
        'p95_return':     np.percentile(returns, 95),
    }


def print_summary(summary: dict, S0: float, confidence: float = 0.95) -> None:
    """
    Prints a formatted summary of all metrics to the terminal

    Parameters:
        summary : dict -> Output of summarize()
        S0 : float -> Starting price, used to convert returns to dollar amounts
        confidence : float -> Confidence level used, for display purposes
    """
    c = int(confidence * 100)

    print("\n" + "=" * 50)
    print("  MONTE CARLO SIMULATION RESULTS")
    print("=" * 50)

    print("\n  RETURN DISTRIBUTION")
    print(f"  Mean return:         {summary['mean_return']:>8.2%}")
    print(f"  Median return:       {summary['median_return']:>8.2%}")
    print(f"  Std of returns:      {summary['std_return']:>8.2%}")
    print(f"  5th pct return:      {summary['p5_return']:>8.2%}")
    print(f"  95th pct return:     {summary['p95_return']:>8.2%}")
    print(f"  Probability profit:  {summary['prob_profit']:>8.2%}")

    print(f"\n  RISK METRICS ({c}% confidence)")
    print(f"  VaR:                 {summary['var']:>8.2%}  "
          f"(${summary['var'] * S0:,.2f})")
    print(f"  CVaR:                {summary['cvar']:>8.2%}  "
          f"(${summary['cvar'] * S0:,.2f})")
    print(f"  Sharpe Ratio:        {summary['sharpe']:>8.2f}  "
      f"(strategy mode only -- requires daily return stream)")

    print(f"\n  MAX DRAWDOWN DISTRIBUTION")
    print(f"  Mean:                {summary['max_drawdown']['mean']:>8.2%}")
    print(f"  Median:              {summary['max_drawdown']['median']:>8.2%}")
    print(f"  5th pct:             {summary['max_drawdown']['p5']:>8.2%}")
    print(f"  95th pct:            {summary['max_drawdown']['p95']:>8.2%}")
    print(f"  Worst case:          {summary['max_drawdown']['worst']:>8.2%}")

    print("=" * 50)