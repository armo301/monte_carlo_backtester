import numpy as np
from arch import arch_model
from models.base import BaseModel

class GARCH(BaseModel):
    """
    GARCH(1,1) model with time-varying volatility
    
    Variance equation:
        sigma^2_t = omega + alpha*epsilon^2_{t-1} + beta*sigma^2_{t-1}

    Parameters are estimated via MLE using the arch library
    Stationarity requires alpha + beta < 1
    """
    
    def __init__(self, dt: float = 1 / 252):
        super().__init__(dt)
        self.mu = None
        self.omega = None 
        self.alpha = None
        self.beta = None
        self.sigma2_0 = None

    def calibrate(self, returns: np.ndarray) -> None:
        """
        Fits GARCH(1,1) parameters via MLE using the arch library
        Stores annualized mu and initial variance for simulation
        """
        scaled_returns = returns * 100
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, mean='constant')
        result = model.fit(disp='off')

        self.mu = np.mean(returns) / self.dt
        self.omega = result.params['omega'] / (100**2)
        self.alpha = result.params['alpha[1]']
        self.beta = result.params['beta[1]']

        self.sigma2_0 = self.omega / (1 - self.alpha - self.beta)
        
        print(f"Calibrated: omega={self.omega: .6f}, alpha={self.alpha: .4f}, "
              f"beta={self.beta: .4f}, alpha+beta={self.alpha+self.beta: .4f}")
        
    def variance_step(self, sigma2_prev: float, epsilon_prev: float) -> float:
        """
        Computes next period variance using GARCH(1,1) equation.
        sigma^2_t = omega + alpha*epsilon^2_{t-1} + beta*sigma^2_{t-1}
        """
        return self.omega + self.alpha * epsilon_prev**2 + self.beta * sigma2_prev
    
    def step(self, current_price: float, sigma2_prev: float, epsilon_prev: float) -> tuple:
        """
        Simulates one GARCH time step forward
        Returns (next_price, next_sigma2, next_epsilon)
        Must return variance and epsilon to maintain state across steps
        """
        sigma2_t = self.variance_step(sigma2_prev, epsilon_prev)
        sigma_t = np.sqrt(sigma2_t)

        z = np.random.standard_normal()
        epsilon_t = sigma_t * z
        r_t = self.mu * self.dt + epsilon_t 

        next_price = current_price * np.exp(r_t)
        return next_price, sigma2_t, epsilon_t
    
    def simulate_path(self, S0: float, n_steps: int) -> np.ndarray:
        """
        Overrides base simulate_path because GARCH requires
        sequential state (sigma2, epsilon) that can't use the
        base class step() signature
        """
        path = np.empty(n_steps + 1)
        path[0] = S0

        sigma2 = self.sigma2_0
        epsilon = 0.0

        for i in range(1, n_steps + 1):
            path[i], sigma2, epsilon = self.step(path[i-1], sigma2, epsilon)

        return path
    
    def simulate_paths(self, S0: float, n_steps: float, n_sims: float) -> np.ndarray:
        """
        Simulates multiple GARCH paths
        Partially vectorized across simulations but sequential across 
        time steps due to GARCH state dependency
        """
        paths = np.empty((n_sims, n_steps + 1))
        paths[:, 0] = S0

        sigma2 = np.full(n_sims, self.sigma2_0)
        epsilon = np.zeros(n_sims)

        for t in range(1, n_steps + 1):
            sigma2 = self.omega + self.alpha * epsilon**2 + self.beta * sigma2
            sigma = np.sqrt(sigma2)

            z = np.random.standard_normal(n_sims)
            epsilon = sigma * z
            r_t = self.mu * self.dt + epsilon 

            paths[:, t] = paths[:, t-1] * np.exp(r_t)

        return paths
    
    def __repr__(self):
        return (f"GARCH(mu={self.mu: .4f}, omega={self.omega: .6f}, "
                f"alpha={self.alpha: .4f}, beta={self.beta: .4f})"
                if self.mu is not None
                else "GARCH(uncalibrated)")
