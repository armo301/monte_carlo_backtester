import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from data.fetcher import fetch_returns, generate_synthetic_returns
from models.gbm import GBM
from models.jump_diffusion import JumpDiffusion
from models.garch import GARCH
from models.bootstrap import Bootstrap
from simulation.metrics import summarize, print_summary


@dataclass
class SimulationConfig:
    """
    Configuration object for a Monte Carlo simulation run
    All parameters in one place — passed to engine.run()

    Using a dataclass avoids writing a boilerplate __init__
    and gives us free __repr__ and type checking
    """
    # Required
    model: str  # 'gbm', 'jump', 'garch', 'bootstrap'

    # Data source — one of these must be provided
    ticker: Optional[str] = None    # real data via yfinance
    mu: Optional[float] = None  # synthetic data drift
    sigma: Optional[float] = None   # synthetic data volatility

    # Simulation parameters
    n_sims: int = 10000
    n_days: int = 252
    period: str = '2y'
    random_seed: Optional[int] = None

    # Risk metric parameters
    confidence: float = 0.95
    risk_free: float = 0.05

    # Jump-Diffusion parameters (only used when model='jump')
    lam: float = 5.0
    mu_j: float = -0.10
    sigma_j: float = 0.15

    # Bootstrap parameters (only used when model='bootstrap')
    block_size: int = 20


@dataclass
class SimulationResult:
    """
    Output of a simulation run
    Carries everything the CLI and visualizer need
    """
    config: SimulationConfig
    paths: np.ndarray
    summary: dict
    info: dict
    model: object


class SimulationEngine:
    """
    Orchestrates the full simulation pipeline:
        1. Fetch or generate data
        2. Create and calibrate model
        3. Run Monte Carlo simulation
        4. Compute risk metrics
        5. Return SimulationResult
    """

    def run(self, config: SimulationConfig) -> SimulationResult:
        """
        Executes a full simulation run from config to results

        Parameters:
        config : SimulationConfig -> full configuration for the simulation

        Returns:
        SimulationResult
            Paths, metrics, and metadata in one object.
        """
        # Set random seed if provided for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Step 1: Get data
        returns, prices, info = self._get_data(config)

        # Step 2: Build and calibrate model
        model = self._build_model(config)
        model.calibrate(returns)

        # Step 3: Run simulation
        S0 = prices[-1]
        print(f"\nRunning {config.n_sims:,} simulations "
              f"({config.n_days} days) using {config.model.upper()}...")

        paths = model.simulate_paths(S0, config.n_days, config.n_sims)

        # Step 4: Compute metrics
        summary = summarize(paths, config.risk_free, config.confidence)

        print("Done.")

        return SimulationResult(
            config=config,
            paths=paths,
            summary=summary,
            info=info,
            model=model
        )

    def _get_data(self, config: SimulationConfig) -> tuple:
        """
        Fetches real data or generates synthetic data based on config
        Raises ValueError if neither ticker nor mu/sigma are provided
        """
        if config.ticker is not None:
            return fetch_returns(config.ticker, config.period)

        elif config.mu is not None and config.sigma is not None:
            return generate_synthetic_returns(config.mu, config.sigma)

        else:
            raise ValueError(
                "Must provide either 'ticker' for real data "
                "or both 'mu' and 'sigma' for synthetic data."
            )

    def _build_model(self, config: SimulationConfig) -> object:
        """
        Instantiates the correct model based on config.model string
        Raises ValueError for unrecognized model names
        """
        if config.model == 'gbm':
            return GBM()

        elif config.model == 'jump':
            return JumpDiffusion(
                lam=config.lam,
                mu_j=config.mu_j,
                sigma_j=config.sigma_j
            )

        elif config.model == 'garch':
            return GARCH()

        elif config.model == 'bootstrap':
            return Bootstrap(block_size=config.block_size)

        else:
            raise ValueError(
                f"Unknown model '{config.model}'. "
                f"Choose from: 'gbm', 'jump', 'garch', 'bootstrap'"
            )