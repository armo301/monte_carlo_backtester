import numpy as np
from models.base import BaseModel

class JumpDiffusion(BaseModel):
    """
    Merton Jump Diffusion Model
    Extension of GBM -> Adds compounding poisson jump process

    Discretized Step:
        S(t+dt) = S(t) * exp((mu_adj - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon) * J^N
        -> mu_adj = mu - lambda*(exp(mu_j + 0.5*sigma_j^2) - 1)
        -> epsilon ~ N(0,1)
        -> N ~ Poisson(lambda*dt)
        -> J = exp(N(mu_j, sigma_j))
    """

    def __init__(self, lam: float = 1.0, mu_j: float = -0.10,
                 sigma_j: float = 0.15, dt: float = 1 / 252):
        """
        Parameters:
            lam     : expected number of jumps per year
            mu_j    : mean log jump size
            sigma_j : standard deviation of log jump size
        """
        super().__init__(dt)
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.mu = None
        self.sigma = None

    def calibrate(self, returns: np.ndarray) -> None:
        """
        Estimates mu and sigma from historical log returns
        Jump parameters (lam, mu_j, sigma_j) set at initialization
        and not estimated here due to the complexity of joint calibration.
        """
        self.mu = np.mean(returns) / self.dt
        self.sigma = np.std(returns, ddof=1) / np.sqrt(self.dt)

    def drift_correction(self) -> float:
        """
        Adjusts mu to account for expected return impact of jumps
        Without this, adding jumps would change overall drift of process

        correction = lambda * (E[J] - 1)
                    = lambda * (exp(mu_j + 0.5*sigma_j^2) - 1)
        """
        expected_jump = np.exp(self.mu_j + 0.5 * self.sigma_j**2)
        return self.lam * (expected_jump - 1)

    def step(self, curent_price: float) -> float:
        #diffusion part
        epsilon = np.random.standard_normal
        mu_adj = self.mu - self.drift_correction()
        diffusion = np.exp((mu_adj - 0.5 * self.sigma**2) * self.dt
                            + self.sigma * np.sqrt(self.dt) * epsilon)
        
        #jump part -> draws number of jumps this step from poisson distribution
        n_jumps = np.random.poisson(self.lam * self.dt)

        if n_jumps > 0:
            log_jump = np.random.normal(self.mu_j, self.sigma_j, n_jumps)
            jump_multiplier = np.exp(np.sum(log_jump))
        else:
            jump_multiplier = 1.0
        
        return curent_price * diffusion * jump_multiplier
    
    def simulate_paths(self, S0: float, n_steps: int, n_sims: int) -> np.ndarray:
        """
        Vectorized simulation of multiple Jump-Diffusion paths.
        Returns array of shape (n_sims, n_steps + 1).
        """
        mu_adj = self.mu - self.drift_correction()

        #diffusion part
        epsilon = np.random.standard_normal((n_sims, n_steps))
        diffusion_returns = ((mu_adj - 0.5 * self.sigma**2) * self.dt
                                + self.sigma * np.sqrt(self.dt) * epsilon)
        
        #jump part
        n_jumps = np.random.poisson(self.lam * self.dt, (n_sims, n_steps))

        log_jumps = np.random.normal(self.mu_j, self.sigma_j, (n_sims, n_steps))
        jump_returns = n_jumps * log_jumps

        #combine and build paths
        total_returns = diffusion_returns + jump_returns
        price_relatives = np.exp(total_returns)
        paths = S0 * np.concatenate([np.ones((n_sims, 1)), 
                                        np.cumprod(price_relatives, axis=1)], 
                                        axis=1)
        return paths
    
    def __repr__(self):
        return (f"JumpDiffusion(mu={self.mu: .4f}, sigma={self.sigma: .4f}, "
                f"lam={self.lam}, mu_j={self.mu_j}, sigma_j={self.sigma_j})"
                if self.mu is not None
                else "JumpDiffusion(uncalibrated)")
        