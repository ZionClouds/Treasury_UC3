# Treasury Portfolio Intelligence Agent

A Dynamic Asset Allocation Agent for managing a ~$46B fixed income portfolio. Built with Google ADK (Agent Development Kit) and deployable to Vertex AI Agent Engine.

## What It Does

The agent takes plain English queries and provides:

- **Real-time portfolio breakdowns** with dollar amounts and percentages
- **Historical performance comparisons** between any two dates
- **Monte Carlo simulations** using live market data (1,000 paths)
- **Rebalancing impact analysis** with risk metrics (VaR, expected return, volatility)

## Features

| Feature | Description |
|---------|-------------|
| Portfolio Status | Returns all 10 asset classes with exact dollar amounts and percentages from BigQuery |
| Performance Comparison | Compares total value and per-asset changes between any two dates |
| Monte Carlo Simulation | Runs 1,000 simulations to show expected outcomes and risks |
| Market Context | Fetches live Treasury yields, corporate spreads, and Fed Funds rate |
| Policy Validation | Checks proposed changes against configurable policy limits |

## Architecture

```
treasury_agent/
├── agent.py           # Main ADK agent definition + system instructions
├── tools.py           # 17 function tools for the agent
├── market_data.py     # FRED API + Yahoo Finance integrations
├── calculators.py     # Portfolio math + Monte Carlo simulation
├── config.py          # GCP config, policy limits
└── __init__.py        # Package exports

scripts/
└── load_portfolio_to_bq.py  # ETL: CSV → BigQuery

data/
└── *.csv              # Source portfolio data
```

## Data Sources

### 1. BigQuery (Google Cloud)
Stores historical portfolio data with monthly snapshots: date, asset class, dollar amount, weight %.

### 2. FRED API (Federal Reserve Economic Data)
Live market yields for expected return calculations:
- `FEDFUNDS` - Federal Funds Rate → Money Market, Repos, Bank Obligations, Commercial Paper
- `DGS5` - 5-Year Treasury Yield → U.S. Agencies, Supranational, Municipal Bonds
- `DGS10` - 10-Year Treasury Yield → U.S. Treasuries, Foreign Bonds
- `DBAA` - Moody's BAA Corporate Yield → Corporate Bonds

### 3. Yahoo Finance
Historical ETF prices for volatility and correlation calculations (36 months of data).

## Asset Classes (10)

1. Money Market Funds
2. Corporate Bonds
3. U.S. Treasuries
4. Repurchase Agreements
5. Bank Obligations
6. Commercial Paper
7. U.S. Agencies
8. Supranational Bonds
9. Municipal Bonds
10. Foreign Bonds

## Setup

### Prerequisites

- Python 3.10+
- Google Cloud account with BigQuery enabled
- FRED API key (free from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
# Clone the repo
git clone https://github.com/aarnavputta/Treasury_UC3.git
cd Treasury_UC3

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
FRED_API_KEY=your-fred-api-key
EOF
```

### Load Data to BigQuery

```bash
python scripts/load_portfolio_to_bq.py
```

### Run Locally

```bash
adk run treasury_agent
```

## Deployment

### Deploy to Vertex AI Agent Engine

```bash
python deploy.py
```

This outputs a `REASONING_ENGINE_ID` - save it for the next step.

### Connect to Gemini Enterprise

1. Update `connect_gemini.sh` with your:
   - `PROJECT_ID`
   - `REASONING_ENGINE_ID` (from deploy.py output)
   - `APP_ID` (your Gemini Enterprise app ID)

2. Run:
```bash
./connect_gemini.sh
```

## Sample Prompts

```
# Portfolio breakdown
Give me the full portfolio breakdown with dollar amounts and percentages

# Historical comparison
How has the portfolio performed since October 2024?

# Asset performance
Show me Corporate Bonds performance from January 2023 to now

# Market rates
What are current market rates?

# Monte Carlo simulation
Run a Monte Carlo simulation on the current portfolio for 1 year

# Rebalancing analysis
Simulate moving 5% from Money Market Funds to Corporate Bonds using Monte Carlo

# Risk metrics
Show me the weight risk metrics and how current allocations compare to historical patterns

# Policy validation
What happens if I move 40% from Commercial Paper to U.S. Treasuries?
```

## Function Tools (17)

### Portfolio Data (BigQuery)
- `get_portfolio_breakdown` - Full breakdown with $ amounts and %
- `get_current_portfolio` - Current weights only
- `compare_portfolio_performance` - Compare between two dates
- `get_asset_class_performance` - Single asset over time
- `get_historical_weights` - Full weight time series

### Market Data (FRED + Yahoo)
- `get_live_market_rates` - Current yields from FRED
- `get_market_based_assumptions` - Combined yields + volatilities

### Simulation
- `run_monte_carlo_simulation` - 1,000 path simulation
- `simulate_rebalancing_monte_carlo` - Before/after comparison with Monte Carlo
- `simulate_rebalancing` - Basic rebalancing (CMA-based)
- `simulate_portfolio_scenarios` - Historical weight resampling

### Risk & Validation
- `get_weight_risk_metrics` - Weight volatility + z-scores
- `validate_weight_change` - Check against policy limits
- `check_policy_breaches` - Flag policy violations

### Calculation
- `apply_weight_change` - Compute new weights
- `compute_portfolio_comparison` - Before/after metrics
- `get_cma_data` - Fetch Capital Market Assumptions

## Policy Limits

| Asset Class | Min | Max |
|-------------|-----|-----|
| Money Market Funds | 0% | 50% |
| Corporate Bonds | 0% | 35% |
| U.S. Treasuries | 0% | 50% |
| Repurchase Agreements | 0% | 30% |
| Bank Obligations | 0% | 25% |
| Commercial Paper | 0% | 25% |
| U.S. Agencies | 0% | 30% |
| Supranational Bonds | 0% | 15% |
| Municipal Bonds | 0% | 10% |
| Foreign Bonds | 0% | 10% |

## Monte Carlo Details

The simulation uses:
- **Expected Returns**: Live FRED yields
- **Volatilities**: 3-year historical ETF returns
- **Correlations**: 10x10 matrix from ETF price co-movements
- **Method**: Cholesky decomposition for correlated random returns

Output includes: expected final value, 5% VaR, 1% VaR, and percentile distribution.

## License

MIT
