import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def plot_simulation(paths: np.ndarray, summary: dict, info: dict,
                    config, n_paths_to_plot: int = 200) -> None:
    """
    Generates a 2x2 grid of charts from simulation results:
        1. Fan chart of simulated price paths
        2. Final price distribution
        3. Return distribution with normal overlay
        4. Max drawdown distribution

    Parameters:
    paths: np.ndarray -> shape (n_sims, n_steps + 1)
    summary: dict -> output of metrics.summarize()
    info: dict -> output of data fetcher — ticker, dates, etc
    config: SimulationConfig -> simulation configuration for display purposes
    n_paths_to_plot: int -> number of paths to draw on fan chart, 
        plotting all 10,000 would be too slow and cluttered
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Monte Carlo Simulation — {info['ticker']} — "
        f"{config.model.upper()} — {config.n_sims:,} simulations",
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _plot_fan_chart(ax1, paths, info, config)
    _plot_final_price_distribution(ax2, paths, summary, info)
    _plot_return_distribution(ax3, paths, summary, info)
    _plot_drawdown_distribution(ax4, paths, summary)

    plt.savefig(f"simulation_{info['ticker']}_{config.model}.png",
                dpi=150, bbox_inches='tight')
    print(f"\nChart saved as simulation_{info['ticker']}_{config.model}.png")
    plt.show()


def _plot_fan_chart(ax, paths: np.ndarray, info: dict, config) -> None:
    """
    Plots simulated price paths as a fan chart
    Shows distribution of possible futures visually
    """
    n_sims, n_steps = paths.shape
    t = np.arange(n_steps)

    # Plot random sample of paths in light gray
    sample_idx = np.random.choice(n_sims, size=min(200, n_sims), replace=False)
    for idx in sample_idx:
        ax.plot(t, paths[idx], color='lightgray', alpha=0.3, linewidth=0.5)

    # Plot percentile bands
    p5  = np.percentile(paths, 5,  axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    mean = np.mean(paths, axis=0)

    ax.plot(t, p5,   color='red',    linewidth=1.5, label='5th percentile',  linestyle='--')
    ax.plot(t, p95,  color='green',  linewidth=1.5, label='95th percentile', linestyle='--')
    ax.plot(t, p50,  color='blue',   linewidth=1.5, label='Median')
    ax.plot(t, mean, color='orange', linewidth=1.5, label='Mean')

    # Shade the 5th-95th percentile region
    ax.fill_between(t, p5, p95, alpha=0.1, color='blue')

    ax.set_title('Simulated Price Paths', fontweight='bold')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Price ($)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_final_price_distribution(ax, paths: np.ndarray,
                                    summary: dict, info: dict) -> None:
    """
    Histogram of final prices across all simulations
    Marks starting price, mean, median, and VaR threshold
    """
    final_prices = paths[:, -1]
    S0 = paths[0, 0]

    ax.hist(final_prices, bins=100, color='steelblue',
            edgecolor='white', linewidth=0.3, alpha=0.7)

    # Mark key levels
    ax.axvline(S0, color='black',  linewidth=2,   label=f'Start ${S0:.2f}',          linestyle='-')
    ax.axvline(np.mean(final_prices), color='orange', linewidth=2,
               label=f"Mean ${np.mean(final_prices):.2f}", linestyle='--')
    ax.axvline(np.median(final_prices), color='blue', linewidth=2,
               label=f"Median ${np.median(final_prices):.2f}", linestyle='--')

    # VaR threshold in price terms
    var_price = S0 * (1 - summary['var'])
    ax.axvline(var_price, color='red', linewidth=2,
               label=f"VaR(95%) ${var_price:.2f}", linestyle=':')

    ax.set_title('Final Price Distribution', fontweight='bold')
    ax.set_xlabel('Final Price ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_return_distribution(ax, paths: np.ndarray,
                               summary: dict, info: dict) -> None:
    """
    Histogram of final returns with normal distribution overlay
    Visually demonstrates fat tails vs normal distribution assumption
    Marks VaR and CVaR thresholds
    """
    returns = (paths[:, -1] - paths[:, 0]) / paths[:, 0]

    ax.hist(returns, bins=100, color='steelblue', edgecolor='white',
            linewidth=0.3, alpha=0.7, density=True, label='Simulated returns')

    # Overlay normal distribution for comparison
    x = np.linspace(returns.min(), returns.max(), 300)
    normal_curve = stats.norm.pdf(x, np.mean(returns), np.std(returns))
    ax.plot(x, normal_curve, color='black', linewidth=2,
            linestyle='--', label='Normal distribution')

    # Mark VaR and CVaR
    ax.axvline(-summary['var'],  color='red',    linewidth=2,
               linestyle='--', label=f"VaR(95%)  {summary['var']:.2%}")
    ax.axvline(-summary['cvar'], color='darkred', linewidth=2,
               linestyle=':',  label=f"CVaR(95%) {summary['cvar']:.2%}")

    # Shade CVaR region
    cvar_mask = x < -summary['cvar']
    ax.fill_between(x, normal_curve, where=cvar_mask,
                    color='red', alpha=0.2, label='CVaR region')

    ax.set_title('Return Distribution vs Normal', fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_drawdown_distribution(ax, paths: np.ndarray, summary: dict) -> None:
    """
    Histogram of max drawdowns across all simulated paths
    Marks median and 95th percentile drawdown
    """
    from simulation.metrics import max_drawdown
    drawdowns = np.array([max_drawdown(path) for path in paths])

    ax.hist(drawdowns, bins=100, color='salmon',
            edgecolor='white', linewidth=0.3, alpha=0.7)

    ax.axvline(summary['max_drawdown']['median'], color='blue', linewidth=2,
               linestyle='--',
               label=f"Median {summary['max_drawdown']['median']:.2%}")
    ax.axvline(summary['max_drawdown']['p95'], color='red', linewidth=2,
               linestyle='--',
               label=f"95th pct {summary['max_drawdown']['p95']:.2%}")
    ax.axvline(summary['max_drawdown']['worst'], color='darkred', linewidth=2,
               linestyle=':',
               label=f"Worst {summary['max_drawdown']['worst']:.2%}")

    ax.set_title('Max Drawdown Distribution', fontweight='bold')
    ax.set_xlabel('Max Drawdown')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)