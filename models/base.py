from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for stochastic models
    Each model will implement this interface
    """

    def __init__(self, dt: float = 1/252):
        """
        dt: float --> length of one time step in years
        1/252 represents 1 trading day (252 trading days in a year)
        """
        self.dt = dt
    
    @abstractmethod
    def calibrate(self, returns: np.ndarray) -> None:
        """
        Estimates model parameters from historical log return

        Parameters:
            returns : np.ndarray --> Array of historical log returns:
                r_t = ln(S_t / S_{t-1})
        """
        pass

    @abstractmethod
    def step(self, current_price: float) -> float:
        """
        Simulates a single time step forward

        Parameters:
            current_price : float --> The price at the current time stop
        
        Returns: 
            float --> the price at the next time step
        """
        pass

    @abstractmethod
    def simulate_path(self, S0: float, n_steps: int) -> np.ndarray:
        """
        Simulates a full price path by calling step() repeatedly

        Parameters:
            S0 : float --> Starting price 
            n_steps : int --> Number of steps to simulate

        Returns:
            np.ndarray --> Array of length n_steps + 1 containing full price path, including S0
        """

        path = np.empty(n_steps + 1)
        path[0] = S0

        for i in range(1, n_steps + 1):
            path[i] = self.step(path[i - 1])

        return path