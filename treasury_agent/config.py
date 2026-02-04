"""Configuration constants for the Treasury Agent."""

import os

# GCP Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DATASET_ID = "treasury_uc3"
PORTFOLIO_TABLE = "portfolio_holdings"
CMA_TABLE = "capital_market_assumptions"

# FRED API Configuration (Federal Reserve Economic Data)
FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY environment variable is required")

# Asset Classes
ASSET_CLASSES = [
    "Money Market Funds",
    "Corporate Bonds",
    "U.S. Treasuries",
    "Repurchase Agreements",
    "Bank Obligations",
    "Commercial Paper",
    "U.S. Agencies",
    "Supranational Bonds",
    "Municipal Bonds",
    "Foreign Bonds",
]

# Policy Limits (min%, max%) per asset class
# TODO: Replace with actual policy limits from client
POLICY_LIMITS = {
    "Money Market Funds": (0.0, 50.0),
    "Corporate Bonds": (0.0, 35.0),
    "U.S. Treasuries": (0.0, 50.0),
    "Repurchase Agreements": (0.0, 30.0),
    "Bank Obligations": (0.0, 25.0),
    "Commercial Paper": (0.0, 25.0),
    "U.S. Agencies": (0.0, 30.0),
    "Supranational Bonds": (0.0, 15.0),
    "Municipal Bonds": (0.0, 10.0),
    "Foreign Bonds": (0.0, 10.0),
}

# Risk-free rate for Sharpe ratio calculation
RISK_FREE_RATE = 4.0  # %
