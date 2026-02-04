"""Function tools for the Treasury Asset Allocation Agent."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery
from google.adk.tools import FunctionTool
import pandas as pd
import xlsxwriter
import os
from fpdf import FPDF

from .config import (
    PROJECT_ID, DATASET_ID, PORTFOLIO_TABLE, CMA_TABLE,
    ASSET_CLASSES, POLICY_LIMITS,
    SECTOR_ALLOCATION, SECTOR_OUTLOOK, SECTOR_ICONS,
    ASSET_CLASSES, POLICY_LIMITS,
    SECTOR_ALLOCATION, SECTOR_OUTLOOK, SECTOR_ICONS,
    SECTOR_ALLOCATION, SECTOR_OUTLOOK, SECTOR_ICONS,
    SECTOR_ALLOCATION, SECTOR_OUTLOOK, SECTOR_ICONS,
    APPROX_DURATION, APPROX_CONVEXITY, CURRENT_YIELD_CURVE,
    SECTOR_ETF_MAPPING, MOCK_NEWS_FALLBACK, MACRO_REGIMES
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



# ============================================================================
# Thematic Analysis Tools
# ============================================================================

def get_portfolio_thematic_exposure() -> Dict[str, any]:
    """
    Get the portfolio's exposure to industry sectors (Thematic Analysis).

    NOTE: This primarily applies to Corporate Bonds holdings, as government 
    securities do not have traditional industry sectors.

    Returns:
        Dict with:
            - total_corporate_value: Total value of Corporate Bonds
            - sectors: List of {sector, amount, weight_of_corporates, outlook}
    """
    # Get current portfolio
    breakdown = get_portfolio_breakdown()
    if "error" in breakdown:
        return breakdown

    # Find Corporate Bonds amount
    corp_bonds = next((h for h in breakdown["holdings"] if h["asset_class"] == "Corporate Bonds"), None)
    
    if not corp_bonds:
        return {"error": "No Corporate Bonds found in portfolio to analyze themes."}

    total_corp_value = corp_bonds["amount"]
    
    # Calculate sector values based on mock allocation
    sector_breakdown = []
    
    for sector, weight in SECTOR_ALLOCATION.items():
        amount = total_corp_value * weight
        
        # Create compact format (e.g., $2.5B)
        if amount >= 1_000_000_000:
            fmt_short = f"${amount/1_000_000_000:.2f}B"
        elif amount >= 1_000_000:
            fmt_short = f"${amount/1_000_000:.1f}M"
        else:
            fmt_short = f"${amount:,.0f}"

        sector_breakdown.append({
            "sector": sector,
            "icon": SECTOR_ICONS.get(sector, "🔹"), 
            "amount": amount,
            "amount_formatted": f"${amount:,.2f}",
            "amount_formatted_short": fmt_short,
            "weight_of_corporates_pct": int(weight * 100), # Integer for cleaner table
            "outlook": SECTOR_OUTLOOK.get(sector, "Neutral")
        })
        
    # Sort by amount desc
    sector_breakdown.sort(key=lambda x: x["amount"], reverse=True)

    return {
        "analysis_date": breakdown["snapshot_date"],
        "asset_class_analyzed": "Corporate Bonds",
        "total_value": total_corp_value,
        "total_value_formatted": f"${total_corp_value:,.2f}",
        "sector_breakdown": sector_breakdown
    }


def get_sector_outlook(sector: str) -> str:
    """
    Returns the investment outlook for a specific sector.
    
    Dynmically checks the ACTIVE_REGIME.
    If the regime is "Normal", uses the standard outlook.
    If "Recession" or "Stagflation", returns the override outlook.
    """
    # Check for Regime Overrides
    regime_data = MACRO_REGIMES.get(ACTIVE_REGIME, MACRO_REGIMES["Normal"])
    
    # If the regime has a specific outlook override dict, look there first
    if "outlooks" in regime_data:
        override = regime_data["outlooks"].get(sector)
        if override:
            return f"[{ACTIVE_REGIME} Override] {override}"
            
    # Fallback to default/Normal
    if sector not in SECTOR_OUTLOOK:
        valid_sectors = ", ".join(SECTOR_OUTLOOK.keys())
        return f"Unknown sector: {sector}. Valid sectors are: {valid_sectors}"
        
    return SECTOR_OUTLOOK[sector]


    return SECTOR_OUTLOOK[sector]


# ============================================================================
# Stress Testing Tools
# ============================================================================

def run_stress_test_scenario(shock_bps: int) -> Dict[str, any]:
    """
    Run a deterministic stress test on the portfolio (Sensitivity Analysis).
    
    Calculates the estimated price change for each asset class based on its
    effective duration and the magnitude of the interest rate shock.
    
    Formula: Price Change % ≈ -Duration * (Shock_bps / 10000)

    Args:
        shock_bps: Interest rate shock in basis points (e.g., +100, -50).
                   Positive values mean rates rise (prices fall).
                   Negative values mean rates fall (prices rise).

    Returns:
        Dict with impact analysis, including dollar loss/gain per asset.
    """
    # Get current portfolio
    breakdown = get_portfolio_breakdown()
    if "error" in breakdown:
        return breakdown

    total_value = breakdown["total_value"]
    
    # Calculate impact per asset
    impact_analysis = []
    total_impact_amount = 0.0
    
    for holding in breakdown["holdings"]:
        asset = holding["asset_class"]
        amount = holding["amount"]
        duration = APPROX_DURATION.get(asset, 0.0)
        
        # Calculate % price change
        # If rates rise (+shock), prices fall (-change)
        pct_change = -duration * (shock_bps / 10000)
        
        # Calculate $ impact
        dollar_impact = amount * pct_change
        total_impact_amount += dollar_impact
        
        # Formatting for table
        new_amount = amount + dollar_impact
        
        # Compact format
        if abs(dollar_impact) >= 1_000_000_000:
            impact_fmt = f"${dollar_impact/1_000_000_000:.2f}B"
        elif abs(dollar_impact) >= 1_000_000:
            impact_fmt = f"${dollar_impact/1_000_000:.1f}M"
        else:
            impact_fmt = f"${dollar_impact:,.0f}"
            
        # Add sign for clarity
        if dollar_impact > 0:
            impact_fmt = f"+{impact_fmt}"
        
        impact_analysis.append({
            "asset_class": asset,
            "duration": duration,
            "shock_bps": shock_bps,
            "pct_change": round(pct_change * 100, 2),
            "dollar_impact": dollar_impact,
            "dollar_impact_formatted": impact_fmt,
            "new_amount": new_amount,
            # No specific icon for asset classes vs sectors, using generic bullet or skip
        })
        
    # Sort by absolute impact (risk drivers)
    impact_analysis.sort(key=lambda x: abs(x["dollar_impact"]), reverse=True)
    
    # Calculate portfolio-level stats
    portfolio_pct_change = (total_impact_amount / total_value) * 100
    new_total_value = total_value + total_impact_amount
    
    scenario_name = f"Rates {'Rise' if shock_bps > 0 else 'Fall'} {abs(shock_bps)} bps"
    
    return {
        "scenario_name": scenario_name,
        "shock_bps": shock_bps,
        "total_portfolio_value": f"${total_value:,.2f}",
        "estimated_impact_amount": f"${total_impact_amount:,.2f}",
        "estimated_impact_pct": round(portfolio_pct_change, 2),
        "new_portfolio_value": f"${new_total_value:,.2f}",
        "asset_impacts": impact_analysis,
        "interpretation": "Highly Sensitive" if abs(portfolio_pct_change) > 5 else "Moderately Sensitive" if abs(portfolio_pct_change) > 2 else "Low Sensitivity"
    }


# ============================================================================
# Macro Regime Tools (State Management)
# ============================================================================

# Global State for the Agent's Worldview
# Options: "Normal", "Recession", "Stagflation"
ACTIVE_REGIME = "Normal"

def set_macro_regime(regime: str) -> str:
    """
    Sets the macroeconomic regime for the simulation.
    
    This acts as a "World State Toggle". Changing this will:
    1. Override Sector Outlooks (e.g., Tech becomes Bearish in Recession).
    2. Shift the Yield Curve in all analytical models.
    3. Change the context for Investment Memos.

    Args:
        regime: One of ["Normal", "Recession", "Stagflation"].

    Returns:
        Confirmation string describing the new state.
    """
    global ACTIVE_REGIME
    
    # Normalize input
    regime = regime.capitalize()
    if regime not in MACRO_REGIMES:
        return f"Error: Invalid regime '{regime}'. Valid options: {list(MACRO_REGIMES.keys())}"
    
    ACTIVE_REGIME = regime
    desc = MACRO_REGIMES[regime]["description"]
    yield_shift = MACRO_REGIMES[regime]["yield_shift_bps"]
    
    return f"✅ World State Updated to: **{regime}**.\nContext: {desc}\nYield Curve Shift: {yield_shift:+}bps."

def get_current_regime() -> str:
    """Returns the currently active macro regime."""
    return f"Current Regime: **{ACTIVE_REGIME}**\n{MACRO_REGIMES[ACTIVE_REGIME]['description']}"

# ============================================================================
# Live Market Intelligence Tools
# ============================================================================

def get_market_news(query: str) -> Dict[str, any]:
    """
    Fetch live market news for a specific sector or asset class.
    
    Uses yfinance to pull real headlines for proxy ETFs (e.g., 'Technology' -> 'XLK').
    Returns a curated list of recent stories to provide context for market moves.

    Args:
        query: The sector or asset name (e.g., "Technology", "Corporate Bonds", "Energy").

    Returns:
        Dict containing source ETF, timestamp, and a list of news items (title, link).
    """
    # 1. Resolve Query to Ticker
    # Fuzzy match or direct lookup? Direct lookup for now.
    ticker_symbol = SECTOR_ETF_MAPPING.get(query)
    
    if not ticker_symbol:
        # Try finding a partial match
        for key, val in SECTOR_ETF_MAPPING.items():
            if key.lower() in query.lower():
                ticker_symbol = val
                break
    
    # Default to SPY if totally unknown, or return error? 
    # Let's return a helpful error to the agent so it knows.
    if not ticker_symbol:
        return {
            "error": f"Could not map '{query}' to a tracked sector ETF.",
            "available_sectors": list(SECTOR_ETF_MAPPING.keys())
        }

    # 2. Fetch News via yfinance
    try:
        ticker = yf.Ticker(ticker_symbol)
        news_items = ticker.news
        
        # 3. Format Output
        headlines = []
        if news_items:
            for item in news_items[:3]: # Top 3 only
                headlines.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link"),
                    "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime('%Y-%m-%d')
                })
        else:
            # Use Fallback if live fetch returns nothing (common on some networks)
            print(f"Warning: No live news found for {ticker_symbol}, using fallback.")
            fallback_titles = MOCK_NEWS_FALLBACK.get(query, ["Market is watching key economic data releases."])
            for title in fallback_titles:
                headlines.append({
                    "title": title,
                    "publisher": "Market Brief (Fallback)",
                    "link": "#",
                    "published": datetime.now().strftime('%Y-%m-%d')
                })

        return {
            "query": query,
            "proxy_ticker": ticker_symbol,
            "source": "Yahoo Finance (Live)" if news_items else "Internal Brief (Fallback)",
            "news": headlines
        }

    except Exception as e:
        # Failsafe: Return Mock Data so the demo never breaks
        print(f"Error fetching news for {ticker_symbol}: {e}. Using Fallback.")
        headlines = []
        fallback_titles = MOCK_NEWS_FALLBACK.get(query, ["Market is watching key economic data releases."])
        for title in fallback_titles:
            headlines.append({
                "title": title,
                "publisher": "Market Brief (Fallback)",
                "link": "#",
                "published": datetime.now().strftime('%Y-%m-%d')
            })
            
        return {
            "query": query,
            "proxy_ticker": ticker_symbol,
            "source": "Internal Brief (System Fallback)",
            "news": headlines
        }



# ============================================================================
# Analyst Validation Tools (Excel Generation)
# ============================================================================

def generate_validation_excel(shock_bps: int = 100) -> str:
    """
    Generates an 'Investment Banking Grade' Analyst Validation Excel.
    
    Features:
    1. Financial Engineering: Uses Convexity adjustments in stress formulas.
       Formula: Impact % ≈ (-Duration * Δy) + (0.5 * Convexity * (Δy)^2)
    2. Yield Curve Simulation: Visualizes the impact of the shift on the curve.
    3. VaR Distribution: Histograms of potential outcomes.
    4. Live Formulas: Analysts can toggle the shock scenario in the file.

    Args:
        shock_bps: Default shock scenario (e.g., 100).

    Returns:
        str: Absolute path to the generated Excel file.
    """
    filename = "Treasury_Analyst_Validation.xlsx"
    filepath = os.path.abspath(filename)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    workbook = writer.book

    # --- FORMATTING (Bloomberg Style) ---
    fmt_header = workbook.add_format({'bold': True, 'bg_color': '#003366', 'font_color': 'white', 'border': 1}) # Navy Blue
    fmt_currency = workbook.add_format({'num_format': '_($* #,##0.00_);_($* (#,##0.00);_($* "-"??_);_(@_)'})
    fmt_pct = workbook.add_format({'num_format': '0.00%'})
    fmt_bps = workbook.add_format({'num_format': '0 "bps"'})
    fmt_input = workbook.add_format({'bg_color': '#FFFFE0', 'border': 1, 'bold': True})
    fmt_formula_res = workbook.add_format({'num_format': '_($* #,##0.00_);_($* (#,##0.00);_($* "-"??_);_(@_)', 'bold': True, 'font_color': '#003366'})
    fmt_title = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#003366'})
    
    # ---------------------------------------------------------
    # 1. Gather Data & Assumptions
    # ---------------------------------------------------------
    breakdown = get_portfolio_breakdown()
    if "error" in breakdown: return f"Error: {breakdown['error']}"

    # Asset Data
    asset_rows = []
    for h in breakdown["holdings"]:
        asset_class = h["asset_class"]
        asset_rows.append({
            "Asset Class": asset_class,
            "Market Value ($)": h["amount"],
            "Weight (%)": h["weight_pct"] / 100.0,
            "Duration (Yrs)": APPROX_DURATION.get(asset_class, 0.0),
            "Convexity": APPROX_CONVEXITY.get(asset_class, 0.0)
        })
    df_assets = pd.DataFrame(asset_rows)

    # Yield Curve Data
    # Apply Regime Shift to the Curve
    regime_data = MACRO_REGIMES.get(ACTIVE_REGIME, MACRO_REGIMES["Normal"])
    shift_bps = regime_data.get("yield_shift_bps", 0)
    
    base_curve = pd.DataFrame(CURRENT_YIELD_CURVE)
    # Create a copy to avoid mutating the global constant if we weren't using a specific dict
    df_curve = base_curve.copy()
    
    # Apply the shift: New Yield = Old Yield + (Shift / 100)
    df_curve["Yield"] = df_curve["Yield"] + (shift_bps / 100.0)

    # ---------------------------------------------------------
    # Tab 1: Executive Dashboard (Visuals)
    # ---------------------------------------------------------
    ws_dash = workbook.add_worksheet("Executive_Dashboard")
    ws_dash.hide_gridlines(2)
    ws_dash.write(0, 0, "Treasury Portfolio: Strategic Overview", fmt_title)
    
    # KPIs
    ws_dash.write(2, 0, "Total AUM", fmt_header)
    ws_dash.write(3, 0, breakdown["total_value"], fmt_currency)
    ws_dash.write(2, 1, "Top Allocation", fmt_header)
    ws_dash.write(3, 1, df_assets.loc[df_assets['Market Value ($)'].idxmax()]['Asset Class'])
    
    # Charts from other tabs will be referenced here or simple ones created
    # Pie Chart
    df_assets.to_excel(writer, sheet_name='Hidden_Data', index=False)
    chart_pie = workbook.add_chart({'type': 'doughnut'})
    chart_pie.add_series({
        'name': 'Allocation',
        'categories': ['Hidden_Data', 1, 0, len(df_assets), 0],
        'values':     ['Hidden_Data', 1, 1, len(df_assets), 1],
        'data_labels': {'percentage': True}
    })
    chart_pie.set_title({'name': 'Portfolio Composition'})
    ws_dash.insert_chart('A6', chart_pie)

    # ---------------------------------------------------------
    # Tab 2: Financial Engineering (Stress Lab)
    # ---------------------------------------------------------
    ws_stress = workbook.add_worksheet("Stress_Simulation")
    ws_stress.hide_gridlines(2)
    writer.sheets['Stress_Simulation'] = ws_stress
    
    ws_stress.write(0, 0, "Sensitivity Analysis (Duration + Convexity)", fmt_title)
    
    # Control Panel
    ws_stress.write(2, 0, "Scenario Control", fmt_header)
    ws_stress.write(3, 0, "Shock (bps):")
    ws_stress.write(3, 1, shock_bps, fmt_input) # B4 is the Input
    ws_stress.write(3, 2, "<-- Adjust this driver")
    
    # Main Table
    start_row = 6
    headers = ["Asset Class", "Market Value ($)", "Duration", "Convexity", "Shock (bps)", "Linear Impact", "Convexity Adj", "Total Impact ($)", "New Value ($)"]
    for c, h in enumerate(headers):
        ws_stress.write(start_row, c, h, fmt_header)
        
    total_impact_cell_refs = []
    
    for i, row in df_assets.iterrows():
        r = start_row + 1 + i
        # Static Data
        ws_stress.write(r, 0, row["Asset Class"])
        ws_stress.write(r, 1, row["Market Value ($)"], fmt_currency)
        ws_stress.write(r, 2, row["Duration (Yrs)"])
        ws_stress.write(r, 3, row["Convexity"])
        
        # Formulas
        # Shock (Linked to Input B4)
        ws_stress.write_formula(r, 4, "=$B$4", fmt_bps)
        
        # Linear Impact % = -Duration * (Shock/10000)
        # Linear $ = Value * Linear %
        # Form: = B{r} * (-C{r} * ($B$4/10000))
        linear_formula = f"=B{r+1}*(-C{r+1}*($B$4/10000))"
        ws_stress.write_formula(r, 5, linear_formula, fmt_currency)
        
        # Convexity Adj $ = Value * (0.5 * Conv * (Shock/10000)^2)
        # Form: = B{r} * (0.5 * D{r} * ($B$4/10000)^2)
        convexity_formula = f"=B{r+1}*(0.5*D{r+1}*($B$4/10000)^2)"
        ws_stress.write_formula(r, 6, convexity_formula, fmt_currency)
        
        # Total Impact = Linear + Convexity
        total_formula = f"=F{r+1}+G{r+1}"
        ws_stress.write_formula(r, 7, total_formula, fmt_formula_res)
        total_impact_cell_refs.append(f"H{r+1}")
        
        # New Value
        ws_stress.write_formula(r, 8, f"=B{r+1}+H{r+1}", fmt_currency)

    # Total Row
    total_r = start_row + 1 + len(df_assets)
    ws_stress.write(total_r, 0, "TOTAL PORTFOLIO", fmt_header)
    ws_stress.write_formula(total_r, 1, f"=SUM(B{start_row+2}:B{total_r})", fmt_currency)
    ws_stress.write_formula(total_r, 7, f"=SUM(H{start_row+2}:H{total_r})", fmt_formula_res)
    ws_stress.write_formula(total_r, 8, f"=SUM(I{start_row+2}:I{total_r})", fmt_currency)
    
    ws_stress.set_column('A:A', 25)
    ws_stress.set_column('B:I', 18)

    # ---------------------------------------------------------
    # Tab 3: Yield Curve (Line Graph)
    # ---------------------------------------------------------
    ws_curve = workbook.add_worksheet("Yield_Curve_Analysis")
    ws_curve.hide_gridlines(2)
    writer.sheets['Yield_Curve_Analysis'] = ws_curve
    ws_curve.write(0, 0, "Term Structure of Interest Rates", fmt_title)
    
    # Headers
    ws_curve.write(2, 0, "Tenor", fmt_header)
    ws_curve.write(2, 1, "Years", fmt_header)
    ws_curve.write(2, 2, "Current Yield (%)", fmt_header)
    ws_curve.write(2, 3, "Projected Yield (%)", fmt_header) # Formula driven
    
    # Input Reference
    ws_curve.write(0, 3, "Shift (bps):")
    # Link back to main input or create local one? Let's link to Stress Sheet for consistency
    ws_curve.write_formula(0, 4, "=Stress_Simulation!$B$4", fmt_bps) 
    
    chart_curve = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
    
    for i, row in df_curve.iterrows():
        r = 3 + i
        ws_curve.write(r, 0, row["Tenor"])
        ws_curve.write(r, 1, row["Years"])
        ws_curve.write(r, 2, row["Yield"])
        # Projected = Current + (Shift/100)
        ws_curve.write_formula(r, 3, f"=C{r+1}+(Stress_Simulation!$B$4/100)")
        
    # Plotting
    # Current Series
    chart_curve.add_series({
        'name': 'Current Yield Curve',
        'categories': ['Yield_Curve_Analysis', 3, 1, 3+len(df_curve)-1, 1], # X = Years
        'values':     ['Yield_Curve_Analysis', 3, 2, 3+len(df_curve)-1, 2], # Y = Yield
        'line':       {'color': 'blue', 'width': 2.25},
        'marker':     {'type': 'circle', 'size': 5}
    })
    # Projected Series
    chart_curve.add_series({
        'name': 'Projected Scenario',
        'categories': ['Yield_Curve_Analysis', 3, 1, 3+len(df_curve)-1, 1],
        'values':     ['Yield_Curve_Analysis', 3, 3, 3+len(df_curve)-1, 3],
        'line':       {'color': 'red', 'dash_type': 'dash'},
        'marker':     {'type': 'triangle', 'size': 5}
    })
    
    chart_curve.set_x_axis({'name': 'Maturity (Years)', 'min': 0, 'max': 30})
    chart_curve.set_y_axis({'name': 'Yield (%)', 'major_gridlines': {'visible': True}})
    chart_curve.set_title({'name': 'Yield Curve Shift Analysis'})
    chart_curve.set_size({'width': 700, 'height': 450})
    
    ws_curve.insert_chart('F3', chart_curve)
    
    # ---------------------------------------------------------
    # Tab 4: Market Context (News table)
    # ---------------------------------------------------------
    # Recycle previous logic for Context tab
    news_rows = []
    for sector in ["Technology", "Financials", "Energy", "Corporate Bonds"]:
        news = get_market_news(sector)
        if "news" in news:
            for n in news["news"]:
                news_rows.append({"Asset": sector, "Headline": n["title"], "Source": n["publisher"], "Date": n["published"]})
    df_news = pd.DataFrame(news_rows)
    df_news.to_excel(writer, sheet_name='Market_Intelligence', index=False)
    ws_news = writer.sheets['Market_Intelligence']
    ws_news.set_column('A:A', 20)
    ws_news.set_column('B:B', 65)
    
    writer.close()
    return filepath



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

# Thematic Analysis tools
get_portfolio_thematic_exposure_tool = FunctionTool(get_portfolio_thematic_exposure)
get_sector_outlook_tool = FunctionTool(get_sector_outlook)

# Stress Testing tools
run_stress_test_scenario_tool = FunctionTool(run_stress_test_scenario)

# Live News tools
get_market_news_tool = FunctionTool(get_market_news)

# Excel Generation tools
generate_validation_excel_tool = FunctionTool(generate_validation_excel)


# ============================================================================
# Executive Reporting Tools (PDF Memo)
# ============================================================================

class TreasuryMemo(FPDF):
    def header(self):
        # Regime-Aware Header
        regime = get_current_regime_simple()
        
        self.set_font('Arial', 'B', 15)
        if regime in ["Recession", "Stagflation"]:
            # Crisis Header
            self.set_text_color(180, 0, 0) # Dark Red
            self.cell(0, 10, 'ILLINOIS STATE TREASURER - INVESTMENT MEMO', 0, 1, 'C')
            self.set_font('Arial', 'B', 10)
            self.cell(0, 10, f'⚠️ PROTOCOL ACTIVE: {regime.upper()} ⚠️', 0, 1, 'C')
        else:
            # Normal Header
            self.set_text_color(0, 51, 102) # Navy Blue
            self.cell(0, 10, 'ILLINOIS STATE TREASURER - INVESTMENT MEMO', 0, 1, 'C')
            self.set_font('Arial', '', 10)
            self.set_text_color(128, 128, 128) # Gray
            self.cell(0, 5, 'Daily Portfolio Strategy Note', 0, 1, 'C')
            
        self.ln(5)
        # Line break
        self.set_draw_color(0, 0, 0)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def get_current_regime_simple():
    # Helper to access global outside of class
    return ACTIVE_REGIME

def clean_text(text: str) -> str:
    """Sanitizes text for FPDF (Latin-1) compatibility."""
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Smart quotes
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2013': '-', '\u2014': '-',  # Dashes
        '\u2026': '...',               # Ellipsis
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Strip remaining non-latin-1 characters
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_investment_memo(recommendation: str, focus_sector: str = "Technology") -> str:
    """
    Generates a formal PDF Investment Memo for the Treasurer.
    
    Synthesizes:
    1. Active Regime Context (e.g. Recession warning).
    2. Live Market News (The "Why").
    3. Portfolio Risk Metrics (Stress Test Results).
    4. The Agent's Recommendation.

    Args:
        recommendation: The core advice (e.g. "Reduce Tech, Buy Treasuries").
        focus_sector: The sector to highlight in the news section.

    Returns:
        str: Absolute path to the generated PDF.
    """
    pdf = TreasuryMemo()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- 1. Meta Data ---
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 6, "Date:", 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d"), 0, 1)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(40, 6, "To:", 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, "Chief Investment Officer", 0, 1)

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(40, 6, "From:", 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, "AI Portfolio Assistant (Level 3)", 0, 1)
    
    pdf.ln(5)

    # --- 2. Executive Summary ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "1. Executive Recommendation", 0, 1, 'L', fill=True)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 11)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, clean_text(recommendation))
    pdf.ln(5)

    # --- 3. Macro Context (Regime + News) ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, f"2. Market Context ({ACTIVE_REGIME} Regime)", 0, 1, 'L', fill=True)
    pdf.ln(2)
    
    # Regime Desc
    regime_desc = MACRO_REGIMES[ACTIVE_REGIME]["description"]
    pdf.set_font('Arial', 'I', 11)
    pdf.multi_cell(0, 6, f"Macro Backdrop: {clean_text(regime_desc)}")
    pdf.ln(3)

    # Live News
    news = get_market_news(focus_sector)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, f"Recent Intelligence: {focus_sector}", 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if "news" in news:
        for item in news["news"][:3]: # Top 3
            title = item["title"]
            source = item["publisher"]
            pdf.cell(5) # Indent
            # Use multi_cell for wrapping titles
            current_y = pdf.get_y()
            pdf.set_font('Arial', '', 10)
            pdf.cell(2, 6, "-", 0, 0)
            pdf.set_xy(pdf.get_x(), current_y)
            pdf.multi_cell(0, 6, f"  {clean_text(title)} ({clean_text(source)})")
    
    pdf.ln(5)

    # --- 4. Risk Analysis (Stress Test) ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "3. Risk Assessment (Stress Test)", 0, 1, 'L', fill=True)
    pdf.ln(2)
    
    # Run a quick stress test for the memo
    # If Recession -> Test +100bps (Rate Spike Risk in volatility)? Or -100bps?
    # Let's show sensitivity to +100bps as standard risk metric
    shock_scenario = 100
    if ACTIVE_REGIME == "Stagflation": shock_scenario = 200 # Higher stress
    
    # We can't easily call the tool output text locally, but we can call the calculator logic
    # Simplified calc for the memo text
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, f"Sensitivity Analysis conducted at +{shock_scenario}bps shock scenario.")
    
    # Asset Table Header
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 6, "Asset Class", 1)
    pdf.cell(30, 6, "Est. Duration", 1)
    pdf.cell(40, 6, "Est. Price Impact", 1)
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    # Mock row data based on general knowledge or re-fetch config
    # Ideally should use portfolio breakdown, but for speed/simplicity:
    rows = [
        ("U.S. Treasuries",  APPROX_DURATION["U.S. Treasuries"],  f"-{APPROX_DURATION['U.S. Treasuries'] * (shock_scenario/100):.1f}%"),
        ("Corporate Bonds",  APPROX_DURATION["Corporate Bonds"],  f"-{APPROX_DURATION['Corporate Bonds'] * (shock_scenario/100):.1f}%"),
        ("Money Market",     APPROX_DURATION["Money Market Funds"], f"-{APPROX_DURATION['Money Market Funds'] * (shock_scenario/100):.1f}%"),
    ]
    
    for r in rows:
        pdf.cell(60, 6, r[0], 1)
        pdf.cell(30, 6, str(r[1]), 1)
        pdf.cell(40, 6, r[2], 1)
        pdf.ln()

    # --- 5. Sign Off ---
    pdf.ln(15)
    pdf.line(10, pdf.get_y(), 80, pdf.get_y())
    pdf.set_font('Arial', 'I', 10)
    signature_text = f"Generated by Treasury AI | Protocol: {ACTIVE_REGIME.upper()} | {datetime.now().strftime('%H:%M:%S')}"
    pdf.cell(0, 6, signature_text, 0, 1)

    # Output
    filename = f"Investment_Memo_{datetime.now().strftime('%Y%m%d')}.pdf"
    filepath = os.path.abspath(filename)
    pdf.output(filepath)
    
    return filepath


# Regime Switching
set_macro_regime_tool = FunctionTool(set_macro_regime)

# Investment Memo
generate_investment_memo_tool = FunctionTool(generate_investment_memo)






