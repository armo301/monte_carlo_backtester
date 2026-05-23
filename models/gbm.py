import numpy as np
from models.base import BaseModel

class GBM(BaseModel):
    """
    Geometric Brownian Motion Model
    Assumes constant drift and volatility

    Discretized step:
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon)
        -> epsilon ~ N(0,1)
    """

    def __init__(self, dt: float = 1 / 252):
        super().__init__(dt)
        self.mu = None
        self.sigma = None

    def calibrate(self, returns: np.ndarray) -> None:
        """
        Estimate mu and sigma from historical log returns
        Converts from per-step units to annualized units by dividing by dt (for mu)
        and sqrt(dt) (for sigma)
        Uses ddof=1 for sample standard deviation
        """
        self.mu = np.mean(returns) / self.dt
        self.sigma = np.std(returns, ddof=1) / np.sqrt(self.dt)
    
    def step(self, current_price: float) -> float:
        """
        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon)
        """
        epsilon = np.random.standard_normal()
        return current_price * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt
                                       + self.sigma * np.sqrt(self.dt) * epsilon)
    
    def simulate_paths(self, S0: float, n_steps: int, n_sims: int) -> np.ndarray:
        """
        Vectorized simulation of multiple paths simultaneously
        Returns array of shap (n_sims, n_steps + 1)
        Vectorized matrix more efficient than calling simulate_path() n times
        """
        epsilon = np.random.standard_normal((n_sims, n_steps)) #Draws entire matrix

        daily_returns = ((self.mu - 0.5 * self.sigma**2) * self.dt
                            + self.sigma * np.sqrt(self.dt) * epsilon) 
        
        price_relatives = np.exp(daily_returns)
        paths = S0 * np.concatenate([np.ones((n_sims, 1)), 
                                     np.cumprod(price_relatives, axis=1)], 
                                     axis=1) #all the simulation paths computed
        
        return paths
    
    def __repr__(self):
        return (f"GBM(mu={self.mu: .4f}, sigma={self.sigma: .4f}, dt={self.dt})"
                if self.mu is not None
                else "GBM(uncalibrated)"
        )

