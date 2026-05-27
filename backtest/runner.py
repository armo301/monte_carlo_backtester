import numpy as np
from backtest.strategy import BaseStrategy
from simulation.metrics import summarize, print_summary, sharpe_ratio, sharpe_ratio_mc

class BacktestResult:
    """
    Stores the results of running a strategy across simulated paths

    Attributes:
        strategy: the strategy that was run
        pnl_paths: shape (n_sims, n_steps+1) —> portfolio value over time
        strategy_returns: shape (n_sims, n_steps) —> daily returns of strategy
        summary: risk metrics computed from strategy returns
        benchmark: summary of buy and hold for comparison
    """

    def __init__(self, strategy, pnl_paths, strategy_returns, summary, benchmark):
        self.strategy         = strategy
        self.pnl_paths        = pnl_paths
        self.strategy_returns = strategy_returns
        self.summary          = summary
        self.benchmark        = benchmark

    def print_comparison(self, S0: float, confidence: float = 0.95) -> None:
        """
        Prints strategy results alongside buy and hold benchmark.
        """
        print(f"\n{'='*55}")
        print(f"  BACKTEST RESULTS — {self.strategy}")
        print(f"{'='*55}")
        print_summary(self.summary, S0, confidence)

        print(f"\n{'='*55}")
        print(f"  BENCHMARK — Buy and Hold")
        print(f"{'='*55}")
        print_summary(self.benchmark, S0, confidence)

        # Head to head comparison
        print(f"\n{'='*55}")
        print(f"  HEAD TO HEAD COMPARISON")
        print(f"{'='*55}")
        strat_ret  = self.summary['mean_return']
        bench_ret  = self.benchmark['mean_return']
        strat_var  = self.summary['var']
        bench_var  = self.benchmark['var']
        strat_dd   = self.summary['max_drawdown']['median']
        bench_dd   = self.benchmark['max_drawdown']['median']
        strat_prob = self.summary['prob_profit']
        bench_prob = self.benchmark['prob_profit']

        print(f"  {'Metric':<25} {'Strategy':>10} {'Benchmark':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Mean return':<25} {strat_ret:>10.2%} {bench_ret:>10.2%}")
        print(f"  {'VaR (95%)':<25} {strat_var:>10.2%} {bench_var:>10.2%}")
        print(f"  {'Median max drawdown':<25} {strat_dd:>10.2%} {bench_dd:>10.2%}")
        print(f"  {'Prob of profit':<25} {strat_prob:>10.2%} {bench_prob:>10.2%}")
        print(f"{'='*55}")


class BacktestRunner:
    """
    Runs a trading strategy across all simulated price path
    and computes risk metrics on the resulting P&L distribution

    This is the core of Use Case 1:
        simulate many futures -> apply strategy to each -> measure risk
    """

    def __init__(self, risk_free_rate: float = 0.05, dt: float = 1/252):
        self.risk_free_rate = risk_free_rate
        self.dt = dt

    def run(self, paths: np.ndarray, strategy: BaseStrategy,
            S0: float, confidence: float = 0.95) -> BacktestResult:
        """
        Runs strategy on every simulated path and computes metrics

        Parameters:
            paths: np.ndarray -> shape (n_sims, n_steps + 1), simulated price paths
            strategy: BaseStrategy -> any strategy implementing generate_signals()
            S0: float -> starting portfolio value (typically last known price)
            confidence: float -> confidence level for VaR and CVaR

        Returns:
            BacktestResult -> P&L paths, strategy returns, metrics, and benchmark
        """
        n_sims, n_steps_plus_1 = paths.shape
        n_steps = n_steps_plus_1 - 1

        print(f"\nRunning backtest: {strategy} on {n_sims:,} paths...")

        # Run strategy on every path
        pnl_paths        = np.empty((n_sims, n_steps_plus_1))
        strategy_returns = np.empty((n_sims, n_steps))

        for i in range(n_sims):
            pnl, returns = self._run_single_path(paths[i], strategy, S0)
            pnl_paths[i]        = pnl
            strategy_returns[i] = returns

        # Compute metrics on strategy P&L paths
        summary = summarize(pnl_paths, self.risk_free_rate, confidence)

        # Compute Sharpe correctly using daily return stream
        # Average daily returns across all simulations then annualize
        summary['sharpe'] = sharpe_ratio_mc(pnl_paths, self.risk_free_rate, self.dt
        )

        # Run buy and hold as benchmark for comparison
        from backtest.strategy import BuyAndHold
        benchmark_summary = self._run_benchmark(paths, S0, confidence)

        print("Backtest complete.")

        return BacktestResult(
            strategy=strategy,
            pnl_paths=pnl_paths,
            strategy_returns=strategy_returns,
            summary=summary,
            benchmark=benchmark_summary
        )

    def _run_single_path(self, price_path: np.ndarray,
                          strategy: BaseStrategy,
                          S0: float) -> tuple:
        """
        Runs strategy on a single price path
        Returns (pnl_path, daily_returns)

        On long days  (signal=1):  earn the stock's return
        On flat days  (signal=0):  earn the daily risk free rate
        On short days (signal=-1): earn the negative of the stock's return
        """
        n = len(price_path)
        signals = strategy.generate_signals(price_path)

        pnl = np.empty(n)
        pnl[0] = S0
        daily_returns = np.empty(n - 1)

        for t in range(1, n):
            # Stock log return this step
            stock_return = np.log(price_path[t] / price_path[t-1])

            if signals[t-1] == 1:
                # Long — earn stock return
                r = stock_return
            elif signals[t-1] == -1:
                # Short — earn negative stock return
                r = -stock_return
            else:
                # Flat — earn daily risk free rate
                r = self.risk_free_rate * self.dt

            daily_returns[t-1] = r
            pnl[t] = pnl[t-1] * np.exp(r)

        return pnl, daily_returns

    def _run_benchmark(self, paths: np.ndarray,
                        S0: float, confidence: float) -> dict:
        """
        Runs buy and hold on all paths as benchmark
        Buy and hold P&L is just the price path itself scaled to start at S0
        """
        from backtest.strategy import BuyAndHold
        bah = BuyAndHold()
        n_sims = paths.shape[0]
        bah_pnl = np.empty_like(paths)

        for i in range(n_sims):
            bah_pnl[i], _ = self._run_single_path(paths[i], bah, S0)

        summary = summarize(bah_pnl, self.risk_free_rate, confidence)

        # Compute Sharpe for benchmark too
        summary['sharpe'] = sharpe_ratio_mc(bah_pnl, self.risk_free_rate, self.dt)

        return summary