"""Portfolio calculation functions."""

import numpy as np
from typing import Dict, List, Tuple, Optional

from .config import RISK_FREE_RATE, ASSET_CLASSES


# ============================================================================
# Weight-Based Metrics (calculated from historical data)
# ============================================================================

def calculate_weight_volatility(
    historical_weights: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Calculate the standard deviation of weights over time for each asset class.

    Args:
        historical_weights: Dict mapping asset_class to list of historical weight values

    Returns:
        Dict mapping asset_class to its weight volatility (std dev in %)
    """
    volatilities = {}
    for asset_class, weights in historical_weights.items():
        if len(weights) > 1:
            volatilities[asset_class] = round(float(np.std(weights, ddof=1)), 4)
        else:
            volatilities[asset_class] = 0.0
    return volatilities


def calculate_weight_statistics(
    historical_weights: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive statistics for weight time series.

    Args:
        historical_weights: Dict mapping asset_class to list of historical weight values

    Returns:
        Dict with statistics per asset class: mean, std, min, max, current
    """
    stats = {}
    for asset_class, weights in historical_weights.items():
        if len(weights) > 0:
            stats[asset_class] = {
                "mean": round(float(np.mean(weights)), 4),
                "std": round(float(np.std(weights, ddof=1)) if len(weights) > 1 else 0.0, 4),
                "min": round(float(np.min(weights)), 4),
                "max": round(float(np.max(weights)), 4),
                "current": round(float(weights[-1]), 4),  # Most recent
            }
        else:
            stats[asset_class] = {"mean": 0, "std": 0, "min": 0, "max": 0, "current": 0}
    return stats


def historical_resample_simulation(
    historical_weights: Dict[str, List[float]],
    n_scenarios: int = 100,
    seed: Optional[int] = None
) -> Dict[str, any]:
    """
    Simulate future portfolio scenarios by resampling historical weight patterns.

    Randomly samples from historical weight observations to generate
    potential future allocation scenarios.

    Args:
        historical_weights: Dict mapping asset_class to list of historical weight values
        n_scenarios: Number of scenarios to generate (default: 100)
        seed: Random seed for reproducibility

    Returns:
        Dict with:
            - scenarios: List of n_scenarios portfolio allocations
            - weight_ranges: 5th and 95th percentile for each asset
            - expected_weights: Mean weight across all scenarios
    """
    if seed is not None:
        np.random.seed(seed)

    asset_classes = list(historical_weights.keys())
    n_periods = len(list(historical_weights.values())[0])

    # Build weight matrix: rows = time periods, cols = assets
    weight_matrix = np.array([historical_weights[ac] for ac in asset_classes]).T

    # Resample: randomly pick n_scenarios rows from historical data
    indices = np.random.choice(n_periods, size=n_scenarios, replace=True)
    sampled_scenarios = weight_matrix[indices]

    # Calculate statistics across scenarios
    scenarios = []
    for i in range(n_scenarios):
        scenario = {ac: round(float(sampled_scenarios[i, j]), 4)
                   for j, ac in enumerate(asset_classes)}
        scenarios.append(scenario)

    # Weight ranges (5th to 95th percentile)
    weight_ranges = {}
    for j, ac in enumerate(asset_classes):
        p5 = round(float(np.percentile(sampled_scenarios[:, j], 5)), 4)
        p95 = round(float(np.percentile(sampled_scenarios[:, j], 95)), 4)
        weight_ranges[ac] = {"p5": p5, "p95": p95}

    # Expected (mean) weights
    expected_weights = {ac: round(float(np.mean(sampled_scenarios[:, j])), 4)
                       for j, ac in enumerate(asset_classes)}

    return {
        "n_scenarios": n_scenarios,
        "scenarios": scenarios,
        "weight_ranges": weight_ranges,
        "expected_weights": expected_weights,
    }


def calculate_portfolio_weight_risk(
    current_weights: Dict[str, float],
    historical_weights: Dict[str, List[float]]
) -> Dict[str, any]:
    """
    Calculate risk metrics based on how current weights compare to historical patterns.

    Args:
        current_weights: Current portfolio weights
        historical_weights: Historical weight data

    Returns:
        Dict with risk assessment per asset class
    """
    stats = calculate_weight_statistics(historical_weights)
    risk_assessment = {}

    for asset, weight in current_weights.items():
        if asset in stats:
            s = stats[asset]
            # Z-score: how many std devs from historical mean
            z_score = (weight - s["mean"]) / s["std"] if s["std"] > 0 else 0.0

            risk_assessment[asset] = {
                "current_weight": weight,
                "historical_mean": s["mean"],
                "historical_std": s["std"],
                "z_score": round(z_score, 2),
                "at_historical_high": weight >= s["max"] * 0.95,
                "at_historical_low": weight <= s["min"] * 1.05 if s["min"] > 0 else weight < 0.5,
            }

    return risk_assessment


# ============================================================================
# CMA-Based Metrics (legacy - uses Capital Market Assumptions)
# ============================================================================

def calculate_expected_return(weights: Dict[str, float], cmas: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculate portfolio expected return as weighted average of asset returns.

    Args:
        weights: Dict of {asset_class: weight%}
        cmas: Dict of {asset_class: (expected_return%, volatility%)}

    Returns:
        Expected portfolio return in %
    """
    total_return = 0.0
    for asset, weight in weights.items():
        if asset in cmas:
            expected_return = cmas[asset][0]
            total_return += (weight / 100.0) * expected_return
    return round(total_return, 4)


def calculate_volatility(weights: Dict[str, float], cmas: Dict[str, Tuple[float, float]],
                         correlation_matrix: np.ndarray = None) -> float:
    """
    Calculate portfolio volatility using mean-variance approach.

    Args:
        weights: Dict of {asset_class: weight%}
        cmas: Dict of {asset_class: (expected_return%, volatility%)}
        correlation_matrix: Optional correlation matrix. If None, assumes identity (no correlation).

    Returns:
        Portfolio volatility in %
    """
    # Build weight and volatility vectors in consistent order
    weight_vector = []
    vol_vector = []

    for asset in ASSET_CLASSES:
        w = weights.get(asset, 0.0) / 100.0
        v = cmas.get(asset, (0, 0))[1] / 100.0
        weight_vector.append(w)
        vol_vector.append(v)

    weight_vector = np.array(weight_vector)
    vol_vector = np.array(vol_vector)

    # If no correlation matrix provided, use identity (conservative assumption)
    if correlation_matrix is None:
        correlation_matrix = np.eye(len(ASSET_CLASSES))

    # Build covariance matrix: Cov(i,j) = vol_i * vol_j * corr(i,j)
    cov_matrix = np.outer(vol_vector, vol_vector) * correlation_matrix

    # Portfolio variance = w' * Cov * w
    portfolio_variance = weight_vector @ cov_matrix @ weight_vector
    portfolio_volatility = np.sqrt(portfolio_variance) * 100  # Convert back to %

    return round(portfolio_volatility, 4)


def calculate_sharpe_ratio(expected_return: float, volatility: float,
                           risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        expected_return: Portfolio expected return in %
        volatility: Portfolio volatility in %
        risk_free_rate: Risk-free rate in %

    Returns:
        Sharpe ratio
    """
    if volatility == 0:
        return 0.0
    sharpe = (expected_return - risk_free_rate) / volatility
    return round(sharpe, 4)


def calculate_portfolio_metrics(weights: Dict[str, float],
                                 cmas: Dict[str, Tuple[float, float]],
                                 correlation_matrix: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate all portfolio metrics.

    Args:
        weights: Dict of {asset_class: weight%}
        cmas: Dict of {asset_class: (expected_return%, volatility%)}
        correlation_matrix: Optional correlation matrix

    Returns:
        Dict with expected_return, volatility, sharpe_ratio
    """
    expected_return = calculate_expected_return(weights, cmas)
    volatility = calculate_volatility(weights, cmas, correlation_matrix)
    sharpe = calculate_sharpe_ratio(expected_return, volatility)

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
    }


# ============================================================================
# Monte Carlo Simulation (with correlation matrix)
# ============================================================================

def monte_carlo_portfolio_simulation(
    initial_value: float,
    weights: Dict[str, float],
    expected_returns: Dict[str, float],
    volatilities: Dict[str, float],
    correlation_matrix: np.ndarray,
    n_simulations: int = 1000,
    time_horizon_years: int = 1,
    seed: Optional[int] = None
) -> Dict[str, any]:
    """
    Run Monte Carlo simulation for portfolio value over time.

    Uses correlated random returns based on the correlation matrix.

    Args:
        initial_value: Starting portfolio value
        weights: Dict of {asset_class: weight%}
        expected_returns: Dict of {asset_class: annual_return%}
        volatilities: Dict of {asset_class: annual_volatility%}
        correlation_matrix: Correlation matrix for asset classes
        n_simulations: Number of simulation paths (default: 1000)
        time_horizon_years: Projection period in years (default: 1)
        seed: Random seed for reproducibility

    Returns:
        Dict with simulation results including VaR, expected values, percentiles
    """
    if seed is not None:
        np.random.seed(seed)

    n_assets = len(ASSET_CLASSES)
    n_months = time_horizon_years * 12

    # Build vectors in consistent order
    weight_vector = np.array([weights.get(ac, 0.0) / 100.0 for ac in ASSET_CLASSES])
    return_vector = np.array([expected_returns.get(ac, 0.0) / 100.0 / 12 for ac in ASSET_CLASSES])  # Monthly
    vol_vector = np.array([volatilities.get(ac, 0.0) / 100.0 / np.sqrt(12) for ac in ASSET_CLASSES])  # Monthly

    # Cholesky decomposition for correlated random numbers
    try:
        cholesky = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If correlation matrix is not positive definite, use diagonal
        cholesky = np.eye(n_assets)

    # Run simulations
    final_values = []
    paths = []

    for sim in range(n_simulations):
        portfolio_value = initial_value
        path = [portfolio_value]

        for month in range(n_months):
            # Generate correlated random returns
            z = np.random.standard_normal(n_assets)
            correlated_z = cholesky @ z

            # Asset returns for this month
            asset_returns = return_vector + vol_vector * correlated_z

            # Portfolio return (weighted sum)
            portfolio_return = np.sum(weight_vector * asset_returns)

            # Update portfolio value
            portfolio_value = portfolio_value * (1 + portfolio_return)
            path.append(portfolio_value)

        final_values.append(portfolio_value)
        if sim < 10:  # Store first 10 paths for visualization
            paths.append(path)

    final_values = np.array(final_values)

    # Calculate statistics
    expected_value = np.mean(final_values)
    std_dev = np.std(final_values)

    # Value at Risk (VaR) - 5% worst case
    var_5 = np.percentile(final_values, 5)
    var_1 = np.percentile(final_values, 1)

    # Conditional VaR (Expected Shortfall) - average of worst 5%
    cvar_5 = np.mean(final_values[final_values <= var_5])

    # Percentiles
    percentiles = {
        "p1": round(np.percentile(final_values, 1), 2),
        "p5": round(np.percentile(final_values, 5), 2),
        "p10": round(np.percentile(final_values, 10), 2),
        "p25": round(np.percentile(final_values, 25), 2),
        "p50": round(np.percentile(final_values, 50), 2),
        "p75": round(np.percentile(final_values, 75), 2),
        "p90": round(np.percentile(final_values, 90), 2),
        "p95": round(np.percentile(final_values, 95), 2),
        "p99": round(np.percentile(final_values, 99), 2),
    }

    return {
        "initial_value": initial_value,
        "time_horizon_years": time_horizon_years,
        "n_simulations": n_simulations,
        "expected_final_value": round(expected_value, 2),
        "expected_return_pct": round((expected_value / initial_value - 1) * 100, 2),
        "std_deviation": round(std_dev, 2),
        "var_5_pct": round(var_5, 2),
        "var_5_pct_loss": round((initial_value - var_5) / initial_value * 100, 2),
        "var_1_pct": round(var_1, 2),
        "cvar_5_pct": round(cvar_5, 2),
        "min_value": round(np.min(final_values), 2),
        "max_value": round(np.max(final_values), 2),
        "percentiles": percentiles,
        "sample_paths": [[round(v, 2) for v in path] for path in paths[:5]],
    }


def compare_monte_carlo_scenarios(
    initial_value: float,
    current_weights: Dict[str, float],
    new_weights: Dict[str, float],
    expected_returns: Dict[str, float],
    volatilities: Dict[str, float],
    correlation_matrix: np.ndarray,
    n_simulations: int = 1000,
    time_horizon_years: int = 1
) -> Dict[str, any]:
    """
    Compare Monte Carlo results between current and proposed allocations.

    Args:
        initial_value: Starting portfolio value
        current_weights: Current portfolio weights
        new_weights: Proposed new weights
        expected_returns: Expected returns per asset class
        volatilities: Volatilities per asset class
        correlation_matrix: Correlation matrix
        n_simulations: Number of simulations
        time_horizon_years: Projection period

    Returns:
        Dict comparing current vs proposed allocation outcomes
    """
    # Run simulation for current allocation
    current_sim = monte_carlo_portfolio_simulation(
        initial_value=initial_value,
        weights=current_weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        n_simulations=n_simulations,
        time_horizon_years=time_horizon_years,
        seed=42  # Same seed for fair comparison
    )

    # Run simulation for new allocation
    new_sim = monte_carlo_portfolio_simulation(
        initial_value=initial_value,
        weights=new_weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        n_simulations=n_simulations,
        time_horizon_years=time_horizon_years,
        seed=42
    )

    return {
        "time_horizon_years": time_horizon_years,
        "n_simulations": n_simulations,
        "current_allocation": {
            "expected_final_value": current_sim["expected_final_value"],
            "expected_return_pct": current_sim["expected_return_pct"],
            "var_5_pct": current_sim["var_5_pct"],
            "var_5_pct_loss": current_sim["var_5_pct_loss"],
        },
        "new_allocation": {
            "expected_final_value": new_sim["expected_final_value"],
            "expected_return_pct": new_sim["expected_return_pct"],
            "var_5_pct": new_sim["var_5_pct"],
            "var_5_pct_loss": new_sim["var_5_pct_loss"],
        },
        "impact": {
            "expected_value_change": round(new_sim["expected_final_value"] - current_sim["expected_final_value"], 2),
            "expected_return_change_pct": round(new_sim["expected_return_pct"] - current_sim["expected_return_pct"], 2),
            "var_improvement": round(new_sim["var_5_pct"] - current_sim["var_5_pct"], 2),
            "risk_change": "lower" if new_sim["var_5_pct_loss"] < current_sim["var_5_pct_loss"] else "higher",
        },
    }
