"""Function tools for the Treasury Asset Allocation Agent."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery
from google.adk.tools import FunctionTool

from .config import (
    PROJECT_ID, DATASET_ID, PORTFOLIO_TABLE, CMA_TABLE,
    ASSET_CLASSES, POLICY_LIMITS
)
from .calculators import (
    calculate_portfolio_metrics,
    calculate_weight_volatility,
    calculate_weight_statistics,
    historical_resample_simulation,
    calculate_portfolio_weight_risk,
    monte_carlo_portfolio_simulation,
    compare_monte_carlo_scenarios,
)
from .market_data import (
    fetch_fred_yields,
    fetch_etf_historical_returns,
    calculate_correlation_matrix,
    calculate_historical_volatility,
    calculate_historical_expected_returns,
    get_market_implied_returns,
    ETF_PROXIES,
)
from .config import RISK_FREE_RATE


def parse_month_string(month_str: str) -> str:
    """
    Parse flexible date strings into YYYY-MM-DD format.

    Supports formats like:
    - "October 2024", "Oct 2024"
    - "2024-10", "2024-10-01"
    - "current", "now", "latest"
    """
    if not month_str or month_str.lower() in ["current", "now", "latest"]:
        return None  # Will use MAX(snapshot_date)

    month_str = month_str.strip()

    # Try YYYY-MM-DD format
    try:
        dt = datetime.strptime(month_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-01")
    except ValueError:
        pass

    # Try YYYY-MM format
    try:
        dt = datetime.strptime(month_str, "%Y-%m")
        return dt.strftime("%Y-%m-01")
    except ValueError:
        pass

    # Try "Month YYYY" format (full month name)
    try:
        dt = datetime.strptime(month_str, "%B %Y")
        return dt.strftime("%Y-%m-01")
    except ValueError:
        pass

    # Try "Mon YYYY" format (abbreviated month name)
    try:
        dt = datetime.strptime(month_str, "%b %Y")
        return dt.strftime("%Y-%m-01")
    except ValueError:
        pass

    raise ValueError(f"Cannot parse date: {month_str}. Use formats like 'October 2024', 'Oct 2024', or '2024-10'")


# Initialize BigQuery client
bq_client = bigquery.Client(project=PROJECT_ID)


def get_current_portfolio() -> Dict[str, float]:
    """
    Fetch current portfolio weights from BigQuery.

    Returns:
        Dict mapping asset class names to their current weight percentages.
        Example: {"Money Market Funds": 23.6, "Corporate Bonds": 17.9, ...}
    """
    query = f"""
    SELECT asset_class, weight_pct
    FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
    WHERE snapshot_date = (
        SELECT MAX(snapshot_date)
        FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
    )
    """
    results = bq_client.query(query).result()
    portfolio = {row.asset_class: float(row.weight_pct) for row in results}
    if not portfolio:
        raise ValueError("No portfolio data found in BigQuery")
    return portfolio


def get_cma_data() -> Dict[str, Tuple[float, float]]:
    """
    Fetch Capital Market Assumptions from BigQuery.

    Returns:
        Dict mapping asset class to (expected_return%, volatility%).
        Example: {"Money Market Funds": (4.25, 0.5), ...}
    """
    query = f"""
    SELECT asset_class, expected_return_pct, volatility_pct
    FROM `{PROJECT_ID}.{DATASET_ID}.{CMA_TABLE}`
    """
    results = bq_client.query(query).result()
    cmas = {row.asset_class: (float(row.expected_return_pct), float(row.volatility_pct))
            for row in results}
    if not cmas:
        raise ValueError("No CMA data found in BigQuery. Please populate the capital_market_assumptions table.")
    return cmas


def get_portfolio_breakdown(month: str = None) -> Dict[str, any]:
    """
    Get full portfolio breakdown with dollar amounts and percentages.

    Args:
        month: Optional month string (e.g., "October 2024", "Oct 2024", "2024-10").
               If None or "current", uses the most recent month.

    Returns:
        Dict with:
            - snapshot_date: The date of the snapshot
            - total_value: Total portfolio value in dollars
            - holdings: List of {asset_class, amount, weight_pct} for each asset
    """
    # Parse the month string
    if month:
        try:
            parsed_date = parse_month_string(month)
        except ValueError as e:
            return {"error": str(e)}
    else:
        parsed_date = None

    # Build query
    if parsed_date:
        query = f"""
        SELECT snapshot_date, asset_class, amount, weight_pct
        FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
        WHERE snapshot_date = '{parsed_date}'
        ORDER BY amount DESC
        """
    else:
        query = f"""
        SELECT snapshot_date, asset_class, amount, weight_pct
        FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
        )
        ORDER BY amount DESC
        """

    try:
        results = bq_client.query(query).result()
        holdings = []
        total_value = 0.0
        snapshot_date = None

        for row in results:
            snapshot_date = str(row.snapshot_date)
            amount = float(row.amount)
            total_value += amount
            holdings.append({
                "asset_class": row.asset_class,
                "amount": amount,
                "weight_pct": round(float(row.weight_pct), 2),
            })

        return {
            "snapshot_date": snapshot_date,
            "total_value": round(total_value, 2),
            "total_value_formatted": f"${total_value:,.2f}",
            "holdings": holdings,
        }
    except Exception as e:
        return {"error": f"Failed to fetch portfolio: {str(e)}"}


def compare_portfolio_performance(start_month: str, end_month: str = None) -> Dict[str, any]:
    """
    Compare portfolio performance between two dates.

    Args:
        start_month: Starting month (e.g., "October 2024", "Oct 2024")
        end_month: Ending month. If None, uses most recent month.

    Returns:
        Dict with:
            - start_date, end_date: The comparison dates
            - start_total, end_total: Total portfolio values
            - total_change: Absolute dollar change
            - total_change_pct: Percentage change
            - asset_changes: Per-asset breakdown of changes
    """
    # Get start portfolio
    start_data = get_portfolio_breakdown(start_month)
    if "error" in start_data:
        return start_data

    # Get end portfolio
    end_data = get_portfolio_breakdown(end_month)
    if "error" in end_data:
        return end_data

    # Calculate total change
    start_total = start_data["total_value"]
    end_total = end_data["total_value"]
    total_change = end_total - start_total
    total_change_pct = (total_change / start_total * 100) if start_total > 0 else 0

    # Calculate per-asset changes
    start_holdings = {h["asset_class"]: h for h in start_data["holdings"]}
    end_holdings = {h["asset_class"]: h for h in end_data["holdings"]}

    asset_changes = []
    for asset_class in ASSET_CLASSES:
        start_h = start_holdings.get(asset_class, {"amount": 0, "weight_pct": 0})
        end_h = end_holdings.get(asset_class, {"amount": 0, "weight_pct": 0})

        start_amt = start_h["amount"] if isinstance(start_h, dict) else 0
        end_amt = end_h["amount"] if isinstance(end_h, dict) else 0
        change = end_amt - start_amt
        change_pct = (change / start_amt * 100) if start_amt > 0 else (100 if end_amt > 0 else 0)

        asset_changes.append({
            "asset_class": asset_class,
            "start_amount": start_amt,
            "end_amount": end_amt,
            "change": change,
            "change_pct": round(change_pct, 2),
            "start_weight_pct": start_h.get("weight_pct", 0),
            "end_weight_pct": end_h.get("weight_pct", 0),
        })

    # Sort by absolute change (largest first)
    asset_changes.sort(key=lambda x: abs(x["change"]), reverse=True)

    return {
        "start_date": start_data["snapshot_date"],
        "end_date": end_data["snapshot_date"],
        "start_total": start_total,
        "start_total_formatted": f"${start_total:,.2f}",
        "end_total": end_total,
        "end_total_formatted": f"${end_total:,.2f}",
        "total_change": round(total_change, 2),
        "total_change_formatted": f"${total_change:,.2f}",
        "total_change_pct": round(total_change_pct, 2),
        "asset_changes": asset_changes,
    }


def get_asset_class_performance(asset_class: str, start_month: str, end_month: str = None) -> Dict[str, any]:
    """
    Get performance of a specific asset class over a time period.

    Args:
        asset_class: The asset class to analyze (e.g., "Corporate Bonds")
        start_month: Starting month
        end_month: Ending month (defaults to most recent)

    Returns:
        Dict with:
            - asset_class: The asset being analyzed
            - start_date, end_date: The time period
            - start_amount, end_amount: Dollar values
            - change, change_pct: Absolute and percentage change
            - weight_change: Change in portfolio weight
            - monthly_data: List of monthly values
    """
    if asset_class not in ASSET_CLASSES:
        return {"error": f"Unknown asset class: {asset_class}. Valid options: {', '.join(ASSET_CLASSES)}"}

    # Parse dates
    try:
        start_date = parse_month_string(start_month)
        end_date = parse_month_string(end_month) if end_month else None
    except ValueError as e:
        return {"error": str(e)}

    # Build query for the time range
    if end_date:
        date_filter = f"snapshot_date >= '{start_date}' AND snapshot_date <= '{end_date}'"
    else:
        date_filter = f"snapshot_date >= '{start_date}'"

    query = f"""
    SELECT snapshot_date, amount, weight_pct
    FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
    WHERE asset_class = '{asset_class}' AND {date_filter}
    ORDER BY snapshot_date ASC
    """

    try:
        results = list(bq_client.query(query).result())

        if not results:
            return {"error": f"No data found for {asset_class} in the specified period"}

        monthly_data = []
        for row in results:
            monthly_data.append({
                "date": str(row.snapshot_date),
                "amount": float(row.amount),
                "weight_pct": round(float(row.weight_pct), 2),
            })

        start_row = monthly_data[0]
        end_row = monthly_data[-1]

        change = end_row["amount"] - start_row["amount"]
        change_pct = (change / start_row["amount"] * 100) if start_row["amount"] > 0 else 0
        weight_change = end_row["weight_pct"] - start_row["weight_pct"]

        return {
            "asset_class": asset_class,
            "start_date": start_row["date"],
            "end_date": end_row["date"],
            "start_amount": start_row["amount"],
            "start_amount_formatted": f"${start_row['amount']:,.2f}",
            "end_amount": end_row["amount"],
            "end_amount_formatted": f"${end_row['amount']:,.2f}",
            "change": round(change, 2),
            "change_formatted": f"${change:,.2f}",
            "change_pct": round(change_pct, 2),
            "start_weight_pct": start_row["weight_pct"],
            "end_weight_pct": end_row["weight_pct"],
            "weight_change": round(weight_change, 2),
            "num_months": len(monthly_data),
            "monthly_data": monthly_data,
        }
    except Exception as e:
        return {"error": f"Failed to fetch asset performance: {str(e)}"}


def simulate_rebalancing(
    source_asset: str,
    target_asset: str,
    change_pct: float
) -> Dict[str, any]:
    """
    Simulate expected returns from reallocating weights between asset classes.

    Uses CMA data from BigQuery for expected returns and volatility.

    Args:
        source_asset: Asset class to reduce (e.g., "Money Market Funds")
        target_asset: Asset class to increase (e.g., "Corporate Bonds")
        change_pct: Percentage points to move (e.g., 5.0 for 5%)

    Returns:
        Dict with:
            - current_weights, new_weights: Before/after allocations
            - current_metrics, new_metrics: Expected return, volatility, Sharpe
            - impact: Changes in metrics
            - validation: Policy compliance check
    """
    # Get current portfolio with amounts
    current_breakdown = get_portfolio_breakdown()
    if "error" in current_breakdown:
        return current_breakdown

    # Get CMA data from BigQuery
    cmas = get_cma_data()

    # Build current weights dict
    current_weights = {h["asset_class"]: h["weight_pct"] for h in current_breakdown["holdings"]}

    # Validate the change
    validation = validate_weight_change(source_asset, target_asset, change_pct, current_weights)

    if not validation["valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }

    # Apply the change
    new_weights = apply_weight_change(source_asset, target_asset, change_pct, current_weights)

    # Calculate metrics using CMA data
    current_metrics = calculate_portfolio_metrics(current_weights, cmas)
    new_metrics = calculate_portfolio_metrics(new_weights, cmas)

    # Calculate dollar impact
    total_value = current_breakdown["total_value"]
    source_current_amt = total_value * current_weights.get(source_asset, 0) / 100
    target_current_amt = total_value * current_weights.get(target_asset, 0) / 100
    move_amount = total_value * change_pct / 100

    return {
        "valid": True,
        "snapshot_date": current_breakdown["snapshot_date"],
        "total_portfolio_value": current_breakdown["total_value_formatted"],
        "rebalancing_move": {
            "from": source_asset,
            "to": target_asset,
            "amount": f"${move_amount:,.2f}",
            "percentage_points": change_pct,
        },
        "weight_changes": {
            source_asset: {
                "before": current_weights.get(source_asset, 0),
                "after": new_weights.get(source_asset, 0),
                "change": -change_pct,
            },
            target_asset: {
                "before": current_weights.get(target_asset, 0),
                "after": new_weights.get(target_asset, 0),
                "change": change_pct,
            },
        },
        "current_metrics": current_metrics,
        "new_metrics": new_metrics,
        "impact": {
            "expected_return_change": round(new_metrics["expected_return"] - current_metrics["expected_return"], 4),
            "volatility_change": round(new_metrics["volatility"] - current_metrics["volatility"], 4),
            "sharpe_ratio_change": round(new_metrics["sharpe_ratio"] - current_metrics["sharpe_ratio"], 4),
        },
        "warnings": validation["warnings"],
    }


def get_historical_weights() -> Dict[str, List[float]]:
    """
    Fetch full historical weight time series from BigQuery.

    Returns:
        Dict mapping asset class to list of historical weight values (chronological order).
        Example: {"Money Market Funds": [15.2, 16.1, 14.8, ...], ...}
    """
    query = f"""
    SELECT snapshot_date, asset_class, weight_pct
    FROM `{PROJECT_ID}.{DATASET_ID}.{PORTFOLIO_TABLE}`
    ORDER BY snapshot_date ASC
    """
    results = bq_client.query(query).result()

    # Group by asset class
    historical = {ac: [] for ac in ASSET_CLASSES}
    current_date = None
    date_weights = {}

    for row in results:
        date_str = str(row.snapshot_date)
        if current_date != date_str:
            if current_date is not None:
                # Add previous date's weights to history
                for ac in ASSET_CLASSES:
                    historical[ac].append(date_weights.get(ac, 0.0))
            current_date = date_str
            date_weights = {}

        date_weights[row.asset_class] = float(row.weight_pct)

    # Don't forget the last date
    if date_weights:
        for ac in ASSET_CLASSES:
            historical[ac].append(date_weights.get(ac, 0.0))

    if not any(historical.values()):
        raise ValueError("No historical data found in BigQuery")

    return historical


def get_weight_risk_metrics() -> Dict[str, any]:
    """
    Get weight-based risk metrics calculated from historical data.

    Returns:
        Dict with:
            - weight_volatility: Std dev of weights for each asset
            - weight_statistics: Mean, std, min, max, current for each asset
            - current_weights: Latest portfolio weights
    """
    historical = get_historical_weights()
    current = get_current_portfolio()

    return {
        "weight_volatility": calculate_weight_volatility(historical),
        "weight_statistics": calculate_weight_statistics(historical),
        "current_weights": current,
        "risk_assessment": calculate_portfolio_weight_risk(current, historical),
    }


def simulate_portfolio_scenarios(n_scenarios: int = 100) -> Dict[str, any]:
    """
    Run historical resampling simulation to generate portfolio scenarios.

    Uses historical weight patterns to simulate potential future allocations.

    Args:
        n_scenarios: Number of scenarios to generate (default: 100)

    Returns:
        Dict with:
            - n_scenarios: Number of scenarios generated
            - weight_ranges: 5th-95th percentile range for each asset
            - expected_weights: Mean weights across scenarios
            - sample_scenarios: First 5 scenarios as examples
    """
    historical = get_historical_weights()
    result = historical_resample_simulation(historical, n_scenarios=n_scenarios)

    # Only return first 5 scenarios to keep response size manageable
    return {
        "n_scenarios": result["n_scenarios"],
        "weight_ranges": result["weight_ranges"],
        "expected_weights": result["expected_weights"],
        "sample_scenarios": result["scenarios"][:5],
    }


def validate_weight_change(
    source_asset: str,
    target_asset: str,
    change_pct: float,
    current_weights: Dict[str, float]
) -> Dict[str, any]:
    """
    Validate a proposed weight change against policy limits.

    Args:
        source_asset: Asset class to reduce (e.g., "Money Market Funds")
        target_asset: Asset class to increase (e.g., "Corporate Bonds")
        change_pct: Percentage points to move (e.g., 5.0 for 5%)
        current_weights: Current portfolio weights

    Returns:
        Dict with:
            - valid: bool indicating if the change is allowed
            - errors: List of validation error messages
            - warnings: List of warning messages
    """
    errors = []
    warnings = []

    # Check assets exist
    if source_asset not in ASSET_CLASSES:
        errors.append(f"Unknown source asset: {source_asset}")
    if target_asset not in ASSET_CLASSES:
        errors.append(f"Unknown target asset: {target_asset}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Get current weights
    source_current = current_weights.get(source_asset, 0.0)
    target_current = current_weights.get(target_asset, 0.0)

    # Calculate proposed new weights
    source_new = source_current - change_pct
    target_new = target_current + change_pct

    # Check source doesn't go negative
    if source_new < 0:
        errors.append(f"Cannot reduce {source_asset} by {change_pct}%. Current weight is only {source_current}%.")

    # Check policy limits
    source_min, source_max = POLICY_LIMITS.get(source_asset, (0, 100))
    target_min, target_max = POLICY_LIMITS.get(target_asset, (0, 100))

    if source_new < source_min:
        errors.append(f"{source_asset} would fall below policy minimum of {source_min}% (proposed: {source_new}%)")

    if target_new > target_max:
        errors.append(f"{target_asset} would exceed policy maximum of {target_max}% (proposed: {target_new}%)")

    # Warnings for approaching limits
    if source_new <= source_min + 2 and source_new >= source_min:
        warnings.append(f"{source_asset} approaching policy minimum ({source_new}% vs {source_min}% min)")

    if target_new >= target_max - 2 and target_new <= target_max:
        warnings.append(f"{target_asset} approaching policy maximum ({target_new}% vs {target_max}% max)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def apply_weight_change(
    source_asset: str,
    target_asset: str,
    change_pct: float,
    current_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Apply a weight change and return new portfolio weights.

    Args:
        source_asset: Asset class to reduce
        target_asset: Asset class to increase
        change_pct: Percentage points to move
        current_weights: Current portfolio weights

    Returns:
        New portfolio weights after the change
    """
    new_weights = current_weights.copy()
    new_weights[source_asset] = round(current_weights.get(source_asset, 0.0) - change_pct, 2)
    new_weights[target_asset] = round(current_weights.get(target_asset, 0.0) + change_pct, 2)
    return new_weights


def compute_portfolio_comparison(
    current_weights: Dict[str, float],
    new_weights: Dict[str, float],
    cmas: Dict[str, Tuple[float, float]]
) -> Dict[str, any]:
    """
    Compute before/after portfolio metrics comparison.

    Args:
        current_weights: Current portfolio weights
        new_weights: Proposed new portfolio weights
        cmas: Capital Market Assumptions

    Returns:
        Dict with before_metrics, after_metrics, and changes
    """
    before_metrics = calculate_portfolio_metrics(current_weights, cmas)
    after_metrics = calculate_portfolio_metrics(new_weights, cmas)

    changes = {
        "expected_return": round(after_metrics["expected_return"] - before_metrics["expected_return"], 4),
        "volatility": round(after_metrics["volatility"] - before_metrics["volatility"], 4),
        "sharpe_ratio": round(after_metrics["sharpe_ratio"] - before_metrics["sharpe_ratio"], 4),
    }

    return {
        "before": before_metrics,
        "after": after_metrics,
        "changes": changes,
    }


def check_policy_breaches(weights: Dict[str, float]) -> List[Dict[str, any]]:
    """
    Check portfolio weights against all policy limits.

    Args:
        weights: Portfolio weights to check

    Returns:
        List of policy breaches, each with asset_class, current_weight, limit_type, limit_value
    """
    breaches = []

    for asset, weight in weights.items():
        if asset not in POLICY_LIMITS:
            continue

        min_limit, max_limit = POLICY_LIMITS[asset]

        if weight < min_limit:
            breaches.append({
                "asset_class": asset,
                "current_weight": weight,
                "limit_type": "minimum",
                "limit_value": min_limit,
            })

        if weight > max_limit:
            breaches.append({
                "asset_class": asset,
                "current_weight": weight,
                "limit_type": "maximum",
                "limit_value": max_limit,
            })

    return breaches


# ============================================================================
# Market Data & Monte Carlo Tools
# ============================================================================

def get_live_market_rates() -> Dict[str, any]:
    """
    Fetch live market rates from FRED (Federal Reserve Economic Data).

    Returns current Treasury yields, corporate bond spreads, and Fed Funds rate.
    Requires FRED_API_KEY environment variable (free from fred.stlouisfed.org).

    Returns:
        Dict with:
            - yields: Current market yields (1M Treasury, 10Y Treasury, Corporate, etc.)
            - implied_returns: Expected returns derived from current yields
            - as_of_date: Date of the data
    """
    yields = fetch_fred_yields()

    if "error" in yields:
        return yields

    implied_returns = get_market_implied_returns(yields)

    return {
        "yields": yields,
        "implied_returns": implied_returns,
        "data_source": "FRED (Federal Reserve Economic Data)",
    }


def get_market_based_assumptions() -> Dict[str, any]:
    """
    Get market-calibrated expected returns and volatilities.

    Combines:
    - Live yields from FRED for expected returns
    - Historical ETF returns for volatility and correlation

    Returns:
        Dict with expected_returns, volatilities, correlation_matrix, data_sources
    """
    # Get live yields from FRED
    yields = fetch_fred_yields()
    if "error" in yields:
        return yields

    # Get expected returns from FRED yields
    expected_returns = get_market_implied_returns(yields)
    if "error" in expected_returns:
        return expected_returns

    # Get historical returns from Yahoo Finance for volatility and correlation
    historical_returns = fetch_etf_historical_returns(years=3)
    if "error" in historical_returns:
        return historical_returns

    volatilities = calculate_historical_volatility(historical_returns)
    correlation_matrix = calculate_correlation_matrix(historical_returns, ASSET_CLASSES)

    return {
        "expected_returns": expected_returns,
        "volatilities": volatilities,
        "correlation_matrix": correlation_matrix.tolist(),
        "data_source": "FRED yields + Yahoo Finance ETF history",
        "etf_proxies": ETF_PROXIES,
    }


def run_monte_carlo_simulation(
    time_horizon_years: int = 1,
    n_simulations: int = 1000
) -> Dict[str, any]:
    """
    Run Monte Carlo simulation on the current portfolio.

    Simulates portfolio value paths using correlated returns based on
    historical correlations and market-implied expected returns.

    Args:
        time_horizon_years: Projection period (1, 3, or 5 years)
        n_simulations: Number of simulation paths (default: 1000)

    Returns:
        Dict with:
            - expected_final_value: Mean ending portfolio value
            - var_5_pct: 5% Value at Risk (worst 5% outcome)
            - percentiles: Distribution of outcomes
            - sample_paths: Example simulation paths
    """
    import numpy as np

    # Get current portfolio
    breakdown = get_portfolio_breakdown()
    if "error" in breakdown:
        return breakdown

    initial_value = breakdown["total_value"]
    current_weights = {h["asset_class"]: h["weight_pct"] for h in breakdown["holdings"]}

    # Get market-based assumptions
    assumptions = get_market_based_assumptions()
    expected_returns = assumptions["expected_returns"]
    volatilities = assumptions["volatilities"]

    # Build correlation matrix
    if assumptions["correlation_matrix"] is not None:
        correlation_matrix = np.array(assumptions["correlation_matrix"])
    else:
        correlation_matrix = np.eye(len(ASSET_CLASSES))

    # Run simulation
    result = monte_carlo_portfolio_simulation(
        initial_value=initial_value,
        weights=current_weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        n_simulations=n_simulations,
        time_horizon_years=time_horizon_years,
        seed=42
    )

    result["data_source"] = assumptions["data_source"]
    result["initial_value_formatted"] = f"${initial_value:,.2f}"
    result["expected_final_value_formatted"] = f"${result['expected_final_value']:,.2f}"
    result["var_5_pct_formatted"] = f"${result['var_5_pct']:,.2f}"

    return result


def simulate_rebalancing_monte_carlo(
    source_asset: str,
    target_asset: str,
    change_pct: float,
    time_horizon_years: int = 1,
    n_simulations: int = 1000
) -> Dict[str, any]:
    """
    Compare current vs proposed allocation using Monte Carlo simulation.

    This is the advanced version of simulate_rebalancing that uses:
    - Live market yields from FRED for expected returns
    - Historical ETF correlations for realistic diversification effects
    - Monte Carlo simulation for probability distributions of outcomes

    Args:
        source_asset: Asset class to reduce
        target_asset: Asset class to increase
        change_pct: Percentage points to move
        time_horizon_years: Projection period
        n_simulations: Number of simulations

    Returns:
        Dict comparing current vs proposed allocation with:
            - Expected outcomes
            - Value at Risk comparison
            - Probability of outperformance
    """
    import numpy as np

    # Get current portfolio
    breakdown = get_portfolio_breakdown()
    if "error" in breakdown:
        return breakdown

    initial_value = breakdown["total_value"]
    current_weights = {h["asset_class"]: h["weight_pct"] for h in breakdown["holdings"]}

    # Validate the change
    validation = validate_weight_change(source_asset, target_asset, change_pct, current_weights)
    if not validation["valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
        }

    # Apply the change
    new_weights = apply_weight_change(source_asset, target_asset, change_pct, current_weights)

    # Get market-based assumptions
    assumptions = get_market_based_assumptions()
    expected_returns = assumptions["expected_returns"]
    volatilities = assumptions["volatilities"]

    if assumptions["correlation_matrix"] is not None:
        correlation_matrix = np.array(assumptions["correlation_matrix"])
    else:
        correlation_matrix = np.eye(len(ASSET_CLASSES))

    # Run comparison
    comparison = compare_monte_carlo_scenarios(
        initial_value=initial_value,
        current_weights=current_weights,
        new_weights=new_weights,
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        n_simulations=n_simulations,
        time_horizon_years=time_horizon_years
    )

    move_amount = initial_value * change_pct / 100

    return {
        "valid": True,
        "snapshot_date": breakdown["snapshot_date"],
        "total_portfolio_value": f"${initial_value:,.2f}",
        "rebalancing_move": {
            "from": source_asset,
            "to": target_asset,
            "amount": f"${move_amount:,.2f}",
            "percentage_points": change_pct,
        },
        "weight_changes": {
            source_asset: {
                "before": current_weights.get(source_asset, 0),
                "after": new_weights.get(source_asset, 0),
            },
            target_asset: {
                "before": current_weights.get(target_asset, 0),
                "after": new_weights.get(target_asset, 0),
            },
        },
        "monte_carlo_comparison": comparison,
        "data_source": assumptions["data_source"],
        "interpretation": {
            "expected_value_change": f"${comparison['impact']['expected_value_change']:,.2f}",
            "risk_assessment": comparison["impact"]["risk_change"],
            "recommendation": "Proceed" if comparison["impact"]["expected_value_change"] > 0 and comparison["impact"]["risk_change"] != "higher" else "Review carefully",
        },
    }


# Export function tools for ADK
get_current_portfolio_tool = FunctionTool(get_current_portfolio)
get_portfolio_breakdown_tool = FunctionTool(get_portfolio_breakdown)
get_cma_data_tool = FunctionTool(get_cma_data)
get_historical_weights_tool = FunctionTool(get_historical_weights)
get_weight_risk_metrics_tool = FunctionTool(get_weight_risk_metrics)
simulate_portfolio_scenarios_tool = FunctionTool(simulate_portfolio_scenarios)
simulate_rebalancing_tool = FunctionTool(simulate_rebalancing)
compare_portfolio_performance_tool = FunctionTool(compare_portfolio_performance)
get_asset_class_performance_tool = FunctionTool(get_asset_class_performance)
validate_weight_change_tool = FunctionTool(validate_weight_change)
apply_weight_change_tool = FunctionTool(apply_weight_change)
compute_portfolio_comparison_tool = FunctionTool(compute_portfolio_comparison)
check_policy_breaches_tool = FunctionTool(check_policy_breaches)

# Market Data & Monte Carlo tools
get_live_market_rates_tool = FunctionTool(get_live_market_rates)
get_market_based_assumptions_tool = FunctionTool(get_market_based_assumptions)
run_monte_carlo_simulation_tool = FunctionTool(run_monte_carlo_simulation)
simulate_rebalancing_monte_carlo_tool = FunctionTool(simulate_rebalancing_monte_carlo)
