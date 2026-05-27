import argparse
import sys
import numpy as np

from simulation.engine import SimulationEngine, SimulationConfig
from simulation.metrics import print_summary
from backtest.strategy import BuyAndHold, SMAcrossover, MeanReversion
from backtest.runner import BacktestRunner
from plots.visualizer import plot_simulation


def parse_args():
    """
    Defines and parses all command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation engine for backtesting trading strategies.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Data source (mutually exclusive) ---
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        '--ticker', type=str,
        help='Stock ticker for real data (e.g. AAPL, TSLA, SPY)'
    )
    data_group.add_argument(
        '--returns', type=str,
        help='Path to CSV file of historical strategy returns\n'
             'for Use Case 2 (e.g. my_strategy.csv)'
    )

    # --- Synthetic data (only used with --ticker) ---
    parser.add_argument('--mu',    type=float, help='Annualized drift for synthetic data (e.g. 0.10)')
    parser.add_argument('--sigma', type=float, help='Annualized volatility for synthetic data (e.g. 0.20)')

    # --- Model ---
    parser.add_argument(
        '--model', type=str, default='gbm',
        choices=['gbm', 'jump', 'garch', 'bootstrap'],
        help='Stochastic model to use (default: gbm)'
    )

    # --- Simulation parameters ---
    parser.add_argument('--sims',   type=int,   default=10000, help='Number of simulations (default: 10000)')
    parser.add_argument('--days',   type=int,   default=252,   help='Days to simulate (default: 252)')
    parser.add_argument('--period', type=str,   default='2y',  help='Historical data period (default: 2y)')
    parser.add_argument('--seed',   type=int,   default=None,  help='Random seed for reproducibility')

    # --- Risk parameters ---
    parser.add_argument('--confidence', type=float, default=0.95, help='VaR/CVaR confidence level (default: 0.95)')
    parser.add_argument('--rf',         type=float, default=0.05, help='Risk free rate (default: 0.05)')

    # --- Jump diffusion parameters ---
    parser.add_argument('--lam',     type=float, default=5.0,   help='Jump rate lambda (default: 5.0)')
    parser.add_argument('--mu-j',    type=float, default=-0.10, help='Mean log jump size (default: -0.10)')
    parser.add_argument('--sigma-j', type=float, default=0.15,  help='Jump size volatility (default: 0.15)')

    # --- Bootstrap parameters ---
    parser.add_argument('--block-size', type=int, default=20, help='Block size for bootstrap (default: 20)')

    # --- Strategy ---
    parser.add_argument(
        '--strategy', type=str, default=None,
        choices=['bah', 'sma', 'meanrev'],
        help='Trading strategy to backtest:\n'
             '  bah     — Buy and Hold\n'
             '  sma     — SMA Crossover\n'
             '  meanrev — Mean Reversion'
    )

    # --- Strategy parameters ---
    parser.add_argument('--fast',      type=int,   default=20,  help='SMA fast window (default: 20)')
    parser.add_argument('--slow',      type=int,   default=50,  help='SMA slow window (default: 50)')
    parser.add_argument('--window',    type=int,   default=20,  help='Mean reversion window (default: 20)')
    parser.add_argument('--threshold', type=float, default=1.5, help='Mean reversion threshold (default: 1.5)')

    # --- Output ---
    parser.add_argument('--no-plot', action='store_true', help='Skip chart generation')

    return parser.parse_args()


def build_strategy(args):
    """
    Instantiates the correct strategy from parsed arguments
    Returns None if no strategy was specified
    """
    if args.strategy is None:
        return None
    elif args.strategy == 'bah':
        return BuyAndHold()
    elif args.strategy == 'sma':
        return SMAcrossover(fast=args.fast, slow=args.slow)
    elif args.strategy == 'meanrev':
        return MeanReversion(window=args.window, threshold=args.threshold)


def run_use_case_2(args):
    """
    Use Case 2: load strategy returns from CSV, simulate directly
    No stock ticker needed — the strategy's return history is the input
    """
    from data.fetcher import load_returns_from_csv
    from models.bootstrap import Bootstrap
    from models.gbm import GBM
    from models.garch import GARCH
    from models.jump_diffusion import JumpDiffusion

    print(f"\nUse Case 2: loading strategy returns from {args.returns}")
    returns, info = load_returns_from_csv(args.returns)

    # Build and calibrate model on strategy returns
    if args.model == 'gbm':
        model = GBM()
    elif args.model == 'jump':
        model = JumpDiffusion(lam=args.lam, mu_j=args.mu_j,
                              sigma_j=args.sigma_j)
    elif args.model == 'garch':
        model = GARCH()
    elif args.model == 'bootstrap':
        model = Bootstrap(block_size=args.block_size)

    if args.seed is not None:
        np.random.seed(args.seed)

    model.calibrate(returns)

    S0 = 100.0  # normalized starting value
    paths = model.simulate_paths(S0, args.days, args.sims)

    from simulation.metrics import summarize
    summary = summarize(paths, args.rf, args.confidence)
    print_summary(summary, S0, args.confidence)

    if not args.no_plot:
        plot_simulation(paths, summary, info, args)


def main():
    args = parse_args()

    # --- Use Case 2: CSV returns ---
    if args.returns is not None:
        run_use_case_2(args)
        return

    # --- Use Case 1: simulate stock paths ---
    # Build config — handle synthetic mode
    if args.ticker:
        config = SimulationConfig(
            model=args.model,
            ticker=args.ticker,
            n_sims=args.sims,
            n_days=args.days,
            period=args.period,
            random_seed=args.seed,
            confidence=args.confidence,
            risk_free=args.rf,
            lam=args.lam,
            mu_j=args.mu_j,
            sigma_j=args.sigma_j,
            block_size=args.block_size
        )
    elif args.mu is not None and args.sigma is not None:
        config = SimulationConfig(
            model=args.model,
            mu=args.mu,
            sigma=args.sigma,
            n_sims=args.sims,
            n_days=args.days,
            random_seed=args.seed,
            confidence=args.confidence,
            risk_free=args.rf,
            lam=args.lam,
            mu_j=args.mu_j,
            sigma_j=args.sigma_j,
            block_size=args.block_size
        )
    else:
        print("Error: provide --ticker for real data, --returns for CSV, "
            "or --mu and --sigma for synthetic data.")
        sys.exit(1)

    # Run simulation
    engine = SimulationEngine()
    result = engine.run(config)

    # Print metrics
    print_summary(result.summary, result.info['end_price'], config.confidence)

    # Run backtest if strategy specified
    strategy = build_strategy(args)
    if strategy is not None:
        runner = BacktestRunner(risk_free_rate=args.rf)
        backtest = runner.run(result.paths, strategy, result.info['end_price'],
                              config.confidence)
        backtest.print_comparison(result.info['end_price'], config.confidence)

        if not args.no_plot:
            plot_simulation(backtest.pnl_paths, backtest.summary,
                            result.info, config)
    else:
        if not args.no_plot:
            plot_simulation(result.paths, result.summary,
                            result.info, config)


if __name__ == '__main__':
    main()