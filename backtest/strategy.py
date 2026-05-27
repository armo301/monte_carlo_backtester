import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    Each strategy must implement generate_signals()

    Signals convention:
         1 = long  
         0 = flat  (hold cash, earn risk free rate)
         -1 = short 
    """

    @abstractmethod
    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        """
        Generates a signal for each time step given a price path

        Parameters:
            prices : np.ndarray -> shape (n_steps + 1,) -> a single simulated price path

        Returns:
            np.ndarray -> shape (n_steps + 1,) -> Array of signals: 1, 0, or -1
        """
        pass

    def __repr__(self):
        return self.__class__.__name__


class BuyAndHold(BaseStrategy):
    """
    Always long, baseline strategy
    Every other strategy should be benchmarked against this
    """

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        return np.ones(len(prices))


class SMAcrossover(BaseStrategy):
    """
    Buy when fast moving average crosses above slow moving average 
    Sell when fast crosses back below slow -> momentum strategy

    Logic: if the recent average price is rising faster than the
    long term average, momentum is positive — go long

    Parameters:
        fast : int -> fast moving average window in days, default 20 (one month)
        slow : int -> slow moving average window in days, default 50 (two months)
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError(
                f"Fast window ({fast}) must be less than slow window ({slow})."
            )
        self.fast = fast
        self.slow = slow

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        n = len(prices)
        signals = np.zeros(n)

        # Compute moving averages
        fast_ma = self._rolling_mean(prices, self.fast)
        slow_ma = self._rolling_mean(prices, self.slow)

        # Generate signals where both MAs are available
        # Before slow window is filled we stay flat (signal = 0)
        for i in range(self.slow, n):
            if fast_ma[i] > slow_ma[i]:
                signals[i] = 1
            else:
                signals[i] = 0

        return signals

    def _rolling_mean(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Computes rolling mean with same length as input
        Values before the window is filled are set to NaN
        """
        result = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            result[i] = np.mean(prices[i - window + 1: i + 1])
        return result

    def __repr__(self):
        return f"SMAcrossover(fast={self.fast}, slow={self.slow})"


class MeanReversion(BaseStrategy):
    """
    Buys when price drops significantly below its rolling mean,
    sells when it rises significantly above

    Logic: assumes prices revert to their historical average over time
    Opposite philosophy to momentum — buys weakness, sells strength

    Parameters:
        window : int -> rolling mean window in days, default 20
        
        threshold : float -> number of standard deviations from mean to trigger signal,
            default 1.5
    """

    def __init__(self, window: int = 20, threshold: float = 1.5):
        self.window = window
        self.threshold = threshold

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        n = len(prices)
        signals = np.zeros(n)

        for i in range(self.window, n):
            window_prices = prices[i - self.window: i]
            mean = np.mean(window_prices)
            std = np.std(window_prices, ddof=1)

            if std == 0:
                continue

            z_score = (prices[i] - mean) / std

            # Price is significantly below mean — expect reversion up — buy
            if z_score < -self.threshold:
                signals[i] = 1
            # Price is significantly above mean — expect reversion down — sell
            elif z_score > self.threshold:
                signals[i] = -1
            # Price is near mean — stay flat
            else:
                signals[i] = 0

        return signals

    def __repr__(self):
        return f"MeanReversion(window={self.window}, threshold={self.threshold})"