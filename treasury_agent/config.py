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

# Mock Sector Attribution (for Demo Purposes)
# Applies to "Corporate Bonds" asset class
SECTOR_ALLOCATION = {
    "Technology": 0.30,  # 30%
    "Financials": 0.25,  # 25%
    "Energy": 0.15,      # 15%
    "Healthcare": 0.15,  # 15%
    "Consumer Discretionary": 0.10, # 10%
    "Industrials": 0.05, # 5%
}

# Sector Icons for Visualization
SECTOR_ICONS = {
    "Technology": "🤖", # or 💻
    "Financials": "🏦",
    "Energy": "⚡",
    "Healthcare": "💊",
    "Consumer Discretionary": "🛍️",
    "Industrials": "🏭"
}

# Sector Themes / Qualitative Outlook
# Prefix with sentiment indicator: 🟢 (Positive), 🟡 (Neutral), 🔴 (Negative/Risk), 🟠 (Volatile)
SECTOR_OUTLOOK = {
    "Technology": "🟢 **Bullish**: Driven by AI infrastructure spending and cloud growth. Volatile but high growth potential.",
    "Financials": "🟡 **Neutral**: Benefiting from strict yield curve management. Regulatory headwinds remain.",
    "Energy": "🟠 **Volatile**: Facing geopolitical supply constraints. Strong cash flows but long-term transition risks.",
    "Healthcare": "🟢 **Positive**: Defensive during volatility. Aging demographics provide tailwind.",
    "Consumer Discretionary": "🟡 **Cautious**: Sensitive to rate hikes. Consumer spending holding up better than expected.",
    "Industrials": "🟢 **Stable**: Reshoring trends providing boost. Labor costs are a concern."
}

# Approximate Effective Duration (Sensitivity to Interest Rate Changes)
# Used for deterministic stress testing (e.g., +100bps shock)
APPROX_DURATION = {
    "Money Market Funds": 0.1,      # Very low duration (cash-like)
    "U.S. Treasuries": 5.2,         # Intermediate/Long duration mix
    "Corporate Bonds": 6.5,         # Credit adds to duration
    "U.S. Agencies": 4.0,           # Intermediate
    "Municipal Bonds": 7.0,         # Longer duration usually
    "Foreign Bonds": 5.5,           # Intermediate
    "Supranational Bonds": 3.5,     # Short-Intermediate
    "Repurchase Agreements": 0.05,  # Overnight/Short term
    "Commercial Paper": 0.25,       # < 270 days
    "Bank Obligations": 1.5,        # Short term notes
}

# Approximate Convexity (2nd Order Sensitivity)
# Formula: Price Change % ≈ (-Duration * Δy) + (0.5 * Convexity * (Δy)^2)
APPROX_CONVEXITY = {
    "Money Market Funds": 0.05,
    "U.S. Treasuries": 0.60,      # Positive convexity
    "Corporate Bonds": 0.75,      # Higher convexity due to credit
    "U.S. Agencies": 0.45,
    "Municipal Bonds": 0.85,
    "Foreign Bonds": 0.55,
    "Supranational Bonds": 0.30,
    "Repurchase Agreements": 0.00,
    "Commercial Paper": 0.01,
    "Bank Obligations": 0.10,
}

# Current Yield Curve (Mock Data for Visualization)
CURRENT_YIELD_CURVE = [
    {"Tenor": "1M", "Years": 0.08, "Yield": 5.45},
    {"Tenor": "3M", "Years": 0.25, "Yield": 5.38},
    {"Tenor": "6M", "Years": 0.50, "Yield": 5.20},
    {"Tenor": "1Y", "Years": 1.00, "Yield": 4.90},
    {"Tenor": "2Y", "Years": 2.00, "Yield": 4.65},
    {"Tenor": "5Y", "Years": 5.00, "Yield": 4.25},
    {"Tenor": "10Y", "Years": 10.0, "Yield": 4.10},
    {"Tenor": "30Y", "Years": 30.0, "Yield": 4.35},
]

# Sector to ETF Mapping for Live News (Proxy)
SECTOR_ETF_MAPPING = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Corporate Bonds": "LQD",
    "U.S. Treasuries": "GOVT",
}

# Macro Regime Scenarios (Stateful Overrides)
MACRO_REGIMES = {
    "Normal": {
        "description": "Soft landing. Moderate growth. AI boom continues.",
        # Uses default SECTOR_OUTLOOK and CURRENT_YIELD_CURVE
        "yield_shift_bps": 0
    },
    "Recession": {
        "description": "Hard landing. Deflationary. Flight to safety.",
        "outlooks": {
            "Technology": "🔴 **Bearish**: CapEx cuts and valuation compression.",
            "Financials": "🔴 **Bearish**: Loan defaults rise. Net Interest Margin squeeze.",
            "Energy": "🟠 **Weak**: Demand destruction lowers oil prices.",
            "Healthcare": "🟢 **Defensive**: Stable earnings in downturns.",
            "Consumer Discretionary": "🔴 **Bearish**: Consumer spending collapses.",
            "Industrials": "🟠 **Weak**: Manufacturing slowdown.",
             "Corporate Bonds": "🔴 **Negative**: Spreads widen significantly.",
             "U.S. Treasuries": "🟢 **Bullish**: Flight to quality drives yields down."
        },
        "yield_shift_bps": -150 # Fed cuts rates significantly (-1.5%)
    },
    "Stagflation": {
        "description": "High Inflation + Low Growth. 1970s redux.",
        "outlooks": {
             "Technology": "🔴 **Bearish**: High rates hurt long-duration assets.",
             "Financials": "🟡 **Neutral**: Higher rates help NIM, but defaults hurt.",
             "Energy": "🟢 **Bullish**: Real assets outperform. Supply shocks.",
             "Healthcare": "🟡 **Neutral**: Costs rise, but demand remains.",
             "Consumer Discretionary": "🔴 **Bearish**: Real wages fall.",
             "Industrials": "🟢 **Selective**: Infrastructure spending may continue.",
             "Corporate Bonds": "🟠 **Volatile**: High rates + Credit risk.",
             "U.S. Treasuries": "🔴 **Bearish**: Rates spike to fight inflation."
        },
         "yield_shift_bps": 200 # Rates spike (+2.0%)
    }
}

# Fallback News (in case live fetch fails)
MOCK_NEWS_FALLBACK = {
    "Technology": [
        "AI Infrastructure spending projected to double by 2026",
        "Semiconductor demand remains strong despite supply chain tweaks",
        "Major cloud providers announce new data center expansions"
    ],
    "Financials": [
        "Regional banks stabilize as yield curve un-inverts",
        "Fed stress tests show robust capital buffers",
        "Merger activity in fintech sector heating up"
    ],
    "Energy": [
        "Oil prices fluctuated on geopolitical supply concerns",
        "Renewable transition slowing down due to cost inputs",
        "Major producers stick to production discipline"
    ]
}
