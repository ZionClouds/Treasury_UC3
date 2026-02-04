"""
Market data integration for live yields and historical returns.

Data Sources:
- FRED (Federal Reserve Economic Data) - Live yields and spreads
- Yahoo Finance (via yfinance) - ETF historical returns for correlation matrix
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import FRED_API_KEY

# ETF proxies for each asset class
ETF_PROXIES = {
    "Money Market Funds": "SHV",      # iShares Short Treasury Bond
    "Corporate Bonds": "LQD",          # iShares Investment Grade Corporate Bond
    "U.S. Treasuries": "IEF",          # iShares 7-10 Year Treasury Bond
    "Repurchase Agreements": "BIL",    # SPDR Bloomberg 1-3 Month T-Bill
    "Bank Obligations": "SHV",         # Short-term proxy
    "Commercial Paper": "SHV",         # Short-term proxy
    "U.S. Agencies": "GOVT",           # iShares U.S. Treasury Bond
    "Supranational Bonds": "BNDX",     # Vanguard Total International Bond
    "Municipal Bonds": "MUB",          # iShares National Muni Bond
    "Foreign Bonds": "EMB",            # iShares JP Morgan USD Emerging Markets Bond
}

# FRED series IDs for yields
FRED_SERIES = {
    "1M_TREASURY": "DGS1MO",
    "3M_TREASURY": "DGS3MO",
    "1Y_TREASURY": "DGS1",
    "2Y_TREASURY": "DGS2",
    "5Y_TREASURY": "DGS5",
    "10Y_TREASURY": "DGS10",
    "CORPORATE_AAA": "DAAA",
    "CORPORATE_BAA": "DBAA",
    "FED_FUNDS": "FEDFUNDS",
}


def get_fred_api_key() -> Optional[str]:
    """Get FRED API key from config or environment variable."""
    return FRED_API_KEY or os.environ.get("FRED_API_KEY")


def fetch_fred_yields() -> Dict[str, float]:
    """
    Fetch current yields from FRED API using requests library.

    Returns:
        Dict mapping yield names to current values (in %)
    """
    try:
        import requests
    except ImportError:
        return {"error": "requests library not installed. Run: pip3 install requests"}

    api_key = get_fred_api_key()
    if not api_key:
        return {"error": "FRED_API_KEY not configured. Get free key at https://fred.stlouisfed.org/docs/api/api_key.html"}

    yields = {}
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    for name, series_id in FRED_SERIES.items():
        try:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc"
            }
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()

            if data.get("observations") and len(data["observations"]) > 0:
                value = data["observations"][0].get("value")
                if value and value != ".":  # FRED uses "." for missing data
                    yields[name] = round(float(value), 2)
                else:
                    yields[name] = None
            else:
                yields[name] = None
        except Exception as e:
            yields[name] = None

    yields["as_of_date"] = datetime.now().strftime("%Y-%m-%d")
    return yields


def fetch_etf_historical_returns(
    years: int = 3,
    frequency: str = "monthly"
) -> Dict[str, List[float]]:
    """
    Fetch historical returns for ETF proxies using Yahoo Finance.

    Args:
        years: Number of years of history to fetch
        frequency: "daily" or "monthly"

    Returns:
        Dict mapping asset class to list of periodic returns
    """
    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # Get unique ETF tickers
        unique_etfs = list(set(ETF_PROXIES.values()))

        # Download all ETF data at once
        tickers = " ".join(unique_etfs)
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        # Handle yfinance MultiIndex columns (newer versions return ('Close', 'TICKER'))
        etf_returns = {}
        for etf in unique_etfs:
            try:
                # Try to get Close price - handle both old and new yfinance column formats
                if isinstance(data.columns, tuple) or (hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1):
                    # MultiIndex columns: ('Close', 'ETF')
                    if ('Close', etf) in data.columns:
                        prices = data[('Close', etf)]
                    elif 'Close' in data.columns and etf in data['Close'].columns:
                        prices = data['Close'][etf]
                    else:
                        etf_returns[etf] = []
                        continue
                else:
                    # Single ticker - flat columns
                    prices = data['Close'] if 'Close' in data.columns else data['Adj Close']

                if frequency == "monthly":
                    monthly_prices = prices.resample("ME").last()
                    etf_returns[etf] = monthly_prices.pct_change().dropna().tolist()
                else:
                    etf_returns[etf] = prices.pct_change().dropna().tolist()
            except Exception as e:
                etf_returns[etf] = []

        # Map back to asset classes
        returns = {}
        for asset_class, etf in ETF_PROXIES.items():
            returns[asset_class] = etf_returns.get(etf, [])

        return returns

    except ImportError:
        return {"error": "yfinance not installed. Run: pip3 install yfinance"}
    except Exception as e:
        return {"error": f"Yahoo Finance error: {str(e)}"}


def calculate_correlation_matrix(
    returns: Dict[str, List[float]],
    asset_classes: List[str]
) -> np.ndarray:
    """
    Calculate correlation matrix from historical returns.

    Args:
        returns: Dict mapping asset class to list of returns
        asset_classes: Ordered list of asset classes

    Returns:
        Correlation matrix as numpy array
    """
    # Build returns matrix
    n_assets = len(asset_classes)

    # Find minimum length across all return series
    min_length = min(len(returns.get(ac, [])) for ac in asset_classes)

    if min_length < 12:  # Need at least 12 periods for meaningful correlation
        # Return identity matrix if insufficient data
        return np.eye(n_assets)

    # Build matrix with aligned returns
    returns_matrix = np.zeros((min_length, n_assets))
    for i, ac in enumerate(asset_classes):
        ac_returns = returns.get(ac, [])
        returns_matrix[:, i] = ac_returns[-min_length:]

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(returns_matrix.T)

    # Handle any NaN values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix


def calculate_historical_volatility(
    returns: Dict[str, List[float]],
    annualize: bool = True
) -> Dict[str, float]:
    """
    Calculate historical volatility from returns.

    Args:
        returns: Dict mapping asset class to list of returns
        annualize: Whether to annualize (assumes monthly returns)

    Returns:
        Dict mapping asset class to annualized volatility (in %)
    """
    volatilities = {}

    for asset_class, ret_list in returns.items():
        if isinstance(ret_list, list) and len(ret_list) > 1:
            vol = np.std(ret_list, ddof=1)
            if annualize:
                vol = vol * np.sqrt(12)  # Annualize monthly vol
            volatilities[asset_class] = round(vol * 100, 2)  # Convert to %
        else:
            volatilities[asset_class] = None

    return volatilities


def calculate_historical_expected_returns(
    returns: Dict[str, List[float]],
    annualize: bool = True
) -> Dict[str, float]:
    """
    Calculate expected returns from historical data.

    Args:
        returns: Dict mapping asset class to list of returns
        annualize: Whether to annualize (assumes monthly returns)

    Returns:
        Dict mapping asset class to annualized expected return (in %)
    """
    expected_returns = {}

    for asset_class, ret_list in returns.items():
        if isinstance(ret_list, list) and len(ret_list) > 0:
            mean_ret = np.mean(ret_list)
            if annualize:
                # Compound monthly returns to annual
                annual_ret = (1 + mean_ret) ** 12 - 1
            else:
                annual_ret = mean_ret
            expected_returns[asset_class] = round(annual_ret * 100, 2)  # Convert to %
        else:
            expected_returns[asset_class] = None

    return expected_returns


def get_market_implied_returns(
    yields: Dict[str, float]
) -> Dict[str, float]:
    """
    Derive expected returns from current market yields.

    Maps FRED yields to asset class expected returns.

    Args:
        yields: Dict of current yields from FRED

    Returns:
        Dict mapping asset class to expected return (in %)
    """
    if "error" in yields:
        return yields

    # Get yields - raise error if any are missing
    treasury_1y = yields.get("1Y_TREASURY")
    treasury_5y = yields.get("5Y_TREASURY")
    treasury_10y = yields.get("10Y_TREASURY")
    corp_baa = yields.get("CORPORATE_BAA")
    fed_funds = yields.get("FED_FUNDS")

    missing = []
    if treasury_5y is None: missing.append("5Y_TREASURY")
    if treasury_10y is None: missing.append("10Y_TREASURY")
    if corp_baa is None: missing.append("CORPORATE_BAA")
    if fed_funds is None: missing.append("FED_FUNDS")

    if missing:
        return {"error": f"FRED data missing for: {', '.join(missing)}"}

    return {
        "Money Market Funds": round(fed_funds + 0.1, 2),
        "Corporate Bonds": round(corp_baa, 2),
        "U.S. Treasuries": round((treasury_5y + treasury_10y) / 2, 2),
        "Repurchase Agreements": round(fed_funds + 0.05, 2),
        "Bank Obligations": round(fed_funds + 0.3, 2),
        "Commercial Paper": round(fed_funds + 0.2, 2),
        "U.S. Agencies": round(treasury_5y + 0.15, 2),
        "Supranational Bonds": round(treasury_5y + 0.1, 2),
        "Municipal Bonds": round(treasury_5y * 0.8, 2),
        "Foreign Bonds": round(treasury_10y + 1.5, 2),
        "as_of_date": yields.get("as_of_date"),
    }
