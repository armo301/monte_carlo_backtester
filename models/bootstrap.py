import numpy as np
from models.base import BaseModel

class Bootstrap(BaseModel):
    """
    Historical bootstrap resampling model
    Non-parametric —-> makes no distributional assumptions
    Randomly resamples from empirical historical returns

    Two modes:
        - Simple bootstrap: resample individual days independently
        - Block bootstrap: resample chunks of consecutive days
          to preserve autocorrelation structure
    
    Core limitation: can only reproduce history, cannot generate
    scenarios more extreme than those already in the data.
    """

    def __init__(self, block_size: int = 20, dt = 1 / 252):
        """
        Parameters:
            block_size : # consecutive days per block for bootstrapping
                         default to 20 (1 trading month), 1 for simple bootstrap
        """
        super().__init__(dt)
        self.block_size = block_size
        self.returns = None
        self.mu = None

    def calibrate(self, returns: np.ndarray) -> None:
        """
        Stores historical returns for resampling
        No parameter estimation needed beyond storing the data
        """
        self.returns = returns
        self.mu = np.mean(self.returns) / self.dt

    def step(self, current_price: float) -> float:
        """
        Simulates one step by drawing a single random historical return
        Simple bootstrap only — ignores block structure
        """
        r = np.random.choice(self.returns)
        return current_price * np.exp(r)
    
    def simulate_path(self, S0: float, n_steps: int) -> np.ndarray:
        """
        Overrides base class to use block bootstrap
        Resamples blocks of consecutive returns to preserve
        autocorrelation structure
        """
        path = np.empty(n_steps + 1)
        path[0] = S0

        sampled_returns = self.sample_returns(n_steps)

        for i in range(1, n_steps + 1):
            path[i] = path[i-1] * np.exp(sampled_returns[i-1])

        return path
    
    def simulate_paths(self, S0: float, n_steps: int, n_sims: int) -> np.ndarray:
        """
        Vectorized simulation of multiple bootstrap paths.
        Returns array of shape (n_sims, n_steps + 1).
        """
        paths = np.empty((n_sims, n_steps + 1))
        paths[:, 0] = S0

        return_matrix = np.array([self.sample_returns(n_steps) for _ in range(n_sims)])

        price_relatives = np.exp(return_matrix)
        paths[:, 1:] = S0 * np.cumprod(price_relatives, axis = 1)

        return paths
    
    def sample_returns(self, n_steps: int) -> np.ndarray:
        """
        Samples a sequence of returns of length n_steps using block bootstrap

        Randomly picks starting points in the historical return
        series and takes consecutive blocks from there until
        we have enough returns to fill n_steps
        """
        n_hist = len(self.returns)
        sampled = []

        while len(sampled) < n_steps:
            start = np.random.randint(0, n_hist - self.block_size + 1)
            block = self.returns[start: start + self.block_size]
            sampled.extend(block)
        
        return np.array(sampled[:n_steps])
    
    def __repr__(self):
        return (f"Bootstrap(block_size={self.block_size}, "
                f"n_historical={len(self.returns)}, mu={self.mu: .4f})"
                if self.returns is not None
                else "Boostrap(uncalibrated)")
