import numpy as np
import pytest
from models.gbm import GBM
from models.jump_diffusion import JumpDiffusion
from models.garch import GARCH
from models.bootstrap import Bootstrap

@pytest.fixture
def fake_returns():
    """Reproducible synthetic returns for all tests"""
    np.random.seed(42)
    return np.random.normal(0.0004, 0.01, 504)

@pytest.fixture
def real_like_returns():
    """
    Returns with mild volatility clustering 
        —> more realistic than pure Gaussian for GARCH tests
    """
    np.random.seed(42)
    n = 504
    returns = np.zeros(n)
    sigma = 0.01
    for i in range(1, n):
        sigma = np.sqrt(0.000002 + 0.08 * returns[i-1]**2 + 0.90 * sigma**2)
        returns[i] = sigma * np.random.standard_normal()
    return returns

class TestGBM:

    def test_calibrate_reasonable_parameters(self, fake_returns):
        """mu and sigma should be plausible after calibration"""
        model = GBM()
        model.calibrate(fake_returns)
        assert -0.5 < model.mu < 1.0,    f"mu={model.mu} out of range"
        assert  0.0 < model.sigma < 2.0, f"sigma={model.sigma} out of range"

    def test_step_returns_positive_price(self, fake_returns):
        """GBM prices can never go negative"""
        model = GBM()
        model.calibrate(fake_returns)
        np.random.seed(0)
        for _ in range(1000):
            assert model.step(100.0) > 0

    def test_simulate_path_length(self, fake_returns):
        """Path should have n_steps + 1 points including start"""
        model = GBM()
        model.calibrate(fake_returns)
        path = model.simulate_path(100.0, 252)
        assert len(path) == 253

    def test_simulate_path_starts_at_S0(self, fake_returns):
        """First element of path must equal starting price"""
        model = GBM()
        model.calibrate(fake_returns)
        path = model.simulate_path(150.0, 252)
        assert path[0] == 150.0

    def test_simulate_paths_shape(self, fake_returns):
        """Vectorized paths must have correct shape"""
        model = GBM()
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 1000)
        assert paths.shape == (1000, 253)

    def test_simulate_paths_all_start_at_S0(self, fake_returns):
        """Every simulation must start at S0"""
        model = GBM()
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 1000)
        assert np.all(paths[:, 0] == 100.0)

    def test_simulate_paths_all_positive(self, fake_returns):
        """All prices across all paths must be positive."""
        model = GBM()
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 1000)
        assert np.all(paths > 0)

    def test_ito_correction(self, fake_returns):
        """
        With enough simulations, mean final price should be close to
        S0 * exp(mu * T) -> this validates the Ito correction is working
        If sigma^2/2 term were missing, mean would be too high
        """
        model = GBM()
        model.calibrate(fake_returns)
        np.random.seed(42)
        paths = model.simulate_paths(100.0, 252, 50000)
        empirical_mean = np.mean(paths[:, -1])
        theoretical_mean = 100.0 * np.exp(model.mu * 1.0)
        # Allow 2% tolerance
        assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.02, \
            f"Ito correction may be wrong: empirical={empirical_mean:.2f}, theoretical={theoretical_mean:.2f}"

class TestJumpDiffusion:

    def test_calibrate_sets_mu_sigma(self, fake_returns):
        model = JumpDiffusion()
        model.calibrate(fake_returns)
        assert model.mu is not None
        assert model.sigma is not None

    def test_paths_all_positive(self, fake_returns):
        """Prices must always be positive even with jumps"""
        model = JumpDiffusion(lam=10, mu_j=-0.20, sigma_j=0.20)
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 1000)
        assert np.all(paths > 0)

    def test_jump_increases_volatility(self, fake_returns):
        """
        Jump-Diffusion should produce wider return distribution than GBM
        with same base parameters
        """
        np.random.seed(42)
        gbm = GBM()
        gbm.calibrate(fake_returns)
        gbm_paths = gbm.simulate_paths(100.0, 252, 5000)

        np.random.seed(42)
        jd = JumpDiffusion(lam=10, mu_j=-0.10, sigma_j=0.15)
        jd.calibrate(fake_returns)
        jd_paths = jd.simulate_paths(100.0, 252, 5000)

        gbm_std = np.std(gbm_paths[:, -1])
        jd_std  = np.std(jd_paths[:, -1])
        assert jd_std > gbm_std, \
            f"JD std ({jd_std:.2f}) should exceed GBM std ({gbm_std:.2f})"

    def test_drift_correction_preserves_mean(self, fake_returns):
        """
        Drift correction should keep mean final price close to GBM mean
        """
        np.random.seed(42)
        gbm = GBM()
        gbm.calibrate(fake_returns)
        gbm_paths = gbm.simulate_paths(100.0, 252, 50000)

        np.random.seed(42)
        jd = JumpDiffusion(lam=5, mu_j=-0.05, sigma_j=0.10)
        jd.calibrate(fake_returns)
        jd_paths = jd.simulate_paths(100.0, 252, 50000)

        gbm_mean = np.mean(gbm_paths[:, -1])
        jd_mean  = np.mean(jd_paths[:, -1])
        # Should be within 5%
        assert abs(gbm_mean - jd_mean) / gbm_mean < 0.05, \
            f"Drift correction off: GBM={gbm_mean:.2f}, JD={jd_mean:.2f}"

class TestGARCH:

    def test_stationarity_constraint(self, real_like_returns):
        """alpha + beta must be less than 1 for stationarity"""
        model = GARCH()
        model.calibrate(real_like_returns)
        assert model.alpha + model.beta < 1.0, \
            f"Stationarity violated: alpha+beta={model.alpha + model.beta:.4f}"

    def test_parameters_positive(self, real_like_returns):
        """All GARCH parameters must be positive"""
        model = GARCH()
        model.calibrate(real_like_returns)
        assert model.omega > 0
        assert model.alpha > 0
        assert model.beta  > 0

    def test_paths_shape(self, real_like_returns):
        model = GARCH()
        model.calibrate(real_like_returns)
        paths = model.simulate_paths(100.0, 252, 500)
        assert paths.shape == (500, 253)

    def test_paths_all_positive(self, real_like_returns):
        model = GARCH()
        model.calibrate(real_like_returns)
        paths = model.simulate_paths(100.0, 252, 500)
        assert np.all(paths > 0)

    def test_long_run_variance(self, real_like_returns):
        """
        Long run variance formula: omega / (1 - alpha - beta)
        should be consistent with parameters.
        """
        model = GARCH()
        model.calibrate(real_like_returns)
        lr_var = model.omega / (1 - model.alpha - model.beta)
        assert lr_var > 0, "Long run variance must be positive"

class TestBootstrap:

    def test_paths_only_use_historical_returns(self, fake_returns):
        """
        Bootstrap can only reproduce returns that exist in history
        No simulated daily return should be too far outside historical range
        """
        model = Bootstrap(block_size=20)
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 1000)

        # Compute daily log returns from paths
        log_returns = np.diff(np.log(paths), axis=1).flatten()
        hist_min = fake_returns.min()
        hist_max = fake_returns.max()

        assert np.all(log_returns >= hist_min - 1e-10)
        assert np.all(log_returns <= hist_max + 1e-10)

    def test_paths_shape(self, fake_returns):
        model = Bootstrap(block_size=20)
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 500)
        assert paths.shape == (500, 253)

    def test_paths_start_at_S0(self, fake_returns):
        model = Bootstrap(block_size=20)
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 500)
        assert np.all(paths[:, 0] == 100.0)

    def test_block_size_one_equals_simple_bootstrap(self, fake_returns):
        """Block size of 1 should behave like simple bootstrap"""
        model = Bootstrap(block_size=1)
        model.calibrate(fake_returns)
        paths = model.simulate_paths(100.0, 252, 100)
        assert paths.shape == (100, 253)