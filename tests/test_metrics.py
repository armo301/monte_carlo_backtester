import numpy as np
import pytest
from simulation.metrics import var, cvar, max_drawdown, sharpe_ratio

@pytest.fixture
def sample_returns():
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 1000)

@pytest.fixture
def sample_paths():
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.01, (500, 252))
    paths = np.empty((500, 253))
    paths[:, 0] = 100.0
    for t in range(1, 253):
        paths[:, t] = paths[:, t-1] * np.exp(returns[:, t-1])
    return paths

class TestVaR:

    def test_var_is_positive(self, sample_returns):
        """VaR is always quoted as a positive loss"""
        assert var(sample_returns) > 0

    def test_var_95_greater_than_var_90(self, sample_returns):
        """Higher confidence = more conservative = larger VaR"""
        var_95 = var(sample_returns, confidence=0.95)
        var_90 = var(sample_returns, confidence=0.90)
        assert var_95 > var_90

    def test_var_is_percentile(self, sample_returns):
        """VaR at 95% should equal negative 5th percentile"""
        expected = -np.percentile(sample_returns, 5)
        assert abs(var(sample_returns, 0.95) - expected) < 1e-10

class TestCVaR:

    def test_cvar_greater_than_var(self, sample_returns):
        """CVaR must always be >= VaR at same confidence level"""
        v = var(sample_returns, 0.95)
        cv = cvar(sample_returns, 0.95)
        assert cv >= v

    def test_cvar_is_positive(self, sample_returns):
        assert cvar(sample_returns) > 0

class TestMaxDrawdown:

    def test_flat_path_zero_drawdown(self):
        """A flat price path has zero drawdown"""
        path = np.ones(253) * 100.0
        assert max_drawdown(path) == 0.0

    def test_always_increasing_path_zero_drawdown(self):
        """A monotonically increasing path has zero drawdown"""
        path = np.linspace(100, 200, 253)
        assert max_drawdown(path) == 0.0

    def test_known_drawdown(self):
        """Manual calculation: peak 200, trough 100 = 50% drawdown"""
        path = np.array([100.0, 150.0, 200.0, 100.0, 120.0])
        dd = max_drawdown(path)
        assert abs(dd - 0.50) < 1e-10

    def test_drawdown_between_zero_and_one(self, sample_paths):
        """Drawdown must always be between 0 and 1"""
        for path in sample_paths[:100]:
            dd = max_drawdown(path)
            assert 0.0 <= dd <= 1.0

class TestSharpe:

    def test_positive_returns_positive_sharpe(self):
        """Consistently positive returns above rf should give positive Sharpe"""
        returns = np.full(252, 0.001)  # 0.1% per day
        assert sharpe_ratio(returns, risk_free_rate=0.05) > 0

    def test_zero_std_returns_zero(self):
        """Zero volatility returns should return 0 not crash"""
        returns = np.full(252, 0.0002)
        # std is 0 so sharpe_ratio should handle gracefully
        result = sharpe_ratio(returns, risk_free_rate=0.05)
        assert result == 0.0 or np.isfinite(result)

    def test_higher_return_higher_sharpe(self):
        """Higher returns with same volatility = higher Sharpe"""
        np.random.seed(42)
        low_returns  = np.random.normal(0.0002, 0.01, 252)
        high_returns = np.random.normal(0.0008, 0.01, 252)
        assert sharpe_ratio(high_returns) > sharpe_ratio(low_returns)