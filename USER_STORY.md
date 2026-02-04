# Treasury Portfolio Intelligence Agent - User Story

## Problem Statement

**The Illinois State Treasurer's office manages a ~$46 billion fixed income portfolio across 10 asset classes.** Portfolio managers need to:

1. **Monitor current holdings** - Know exactly where the money is allocated at any time
2. **Track historical performance** - See how allocations and values have changed over time
3. **Understand market conditions** - Get real-time rates that affect portfolio returns
4. **Evaluate rebalancing decisions** - Understand risk/return tradeoffs before moving money
5. **Ensure policy compliance** - Stay within mandated allocation limits
6. **Quantify risk** - Know the probability of losses under different scenarios

**The Problem:** Today, this requires pulling data from multiple systems, running Excel models, manually checking policy limits, and interpreting complex financial metrics. It's slow, error-prone, and requires specialized expertise.

**The Solution:** An AI agent that answers plain English questions about the portfolio, pulling live data from authoritative sources (FRED, Yahoo Finance, BigQuery) and running sophisticated simulations on demand.

---

## All 17 Tools

| # | Tool | Category | Purpose |
|---|------|----------|---------|
| 1 | `get_portfolio_breakdown` | Portfolio | Full breakdown with $ amounts and % for any month |
| 2 | `get_current_portfolio` | Portfolio | Current weights only (latest month) |
| 3 | `compare_portfolio_performance` | Portfolio | Compare total value between two dates |
| 4 | `get_asset_class_performance` | Portfolio | Single asset performance over time period |
| 5 | `get_historical_weights` | Portfolio | Full 126-month weight time series |
| 6 | `get_live_market_rates` | Market Data | Fetch current yields from FRED |
| 7 | `get_market_based_assumptions` | Market Data | Combined FRED yields + Yahoo Finance volatilities |
| 8 | `get_cma_data` | Market Data | Capital Market Assumptions from BigQuery |
| 9 | `run_monte_carlo_simulation` | Simulation | 1,000 path simulation on current portfolio |
| 10 | `simulate_rebalancing_monte_carlo` | Simulation | Compare current vs proposed with Monte Carlo |
| 11 | `simulate_rebalancing` | Simulation | Basic rebalancing simulation (CMA-based) |
| 12 | `simulate_portfolio_scenarios` | Simulation | Historical weight resampling (100 scenarios) |
| 13 | `get_weight_risk_metrics` | Risk | Weight volatility + z-scores vs historical |
| 14 | `validate_weight_change` | Compliance | Check against policy min/max limits |
| 15 | `check_policy_breaches` | Compliance | Flag any policy violations |
| 16 | `apply_weight_change` | Calculation | Compute new weights after move |
| 17 | `compute_portfolio_comparison` | Calculation | Before/after metrics comparison |

---

## User Story: Quarterly Portfolio Review

### Persona

**Sarah Chen**, Senior Portfolio Manager at the Illinois State Treasurer's Office. She's preparing for a quarterly investment committee meeting and needs to evaluate whether to rebalance the portfolio given current market conditions.

---

### Step 1: Get Current Portfolio Status

> *Sarah starts her day by checking the current state of the portfolio*

**Query:**
```
What's the current portfolio breakdown?
```

**Tools Used:** `get_portfolio_breakdown`

**What Happens:** Agent queries BigQuery for the latest month's data, returns all 10 asset classes with exact dollar amounts and percentages.

**Sample Response:**
```
Portfolio Breakdown (December 2025)
Total Value: $46,234,567,890

Asset Class               Amount              Weight
-----------------------------------------------------
Money Market Funds        $16,182,098,762     35.01%
Repurchase Agreements     $7,858,876,541      17.00%
Corporate Bonds           $6,935,185,183      15.00%
U.S. Treasuries           $5,548,148,147      12.00%
...
```

---

### Step 2: Check Historical Performance

> *Sarah wants to see how the portfolio has performed this quarter*

**Query:**
```
How has the portfolio performed since October 2024?
```

**Tools Used:** `compare_portfolio_performance`

**What Happens:** Agent compares two snapshots from BigQuery, calculates total value change and per-asset changes.

---

### Step 3: Review Market Conditions

> *Before making any decisions, Sarah checks current market rates*

**Query:**
```
What are current market rates?
```

**Tools Used:** `get_live_market_rates`

**What Happens:** Agent calls FRED API in real-time, fetches Fed Funds, Treasury yields (1M to 10Y), and corporate spreads.

**Sample Response:**
```
Live Market Rates (as of 2025-12-15)
Source: Federal Reserve Economic Data (FRED)

Short-Term Rates:
  Fed Funds Rate:     4.33%  (FEDFUNDS)
  1-Month Treasury:   4.28%  (DGS1MO)
  3-Month Treasury:   4.31%  (DGS3MO)

Medium-Term:
  5-Year Treasury:    4.15%  (DGS5)

Long-Term:
  10-Year Treasury:   4.42%  (DGS10)

Credit Spreads:
  AAA Corporate:      4.85%  (DAAA)
  BAA Corporate:      5.62%  (DBAA)
```

---

### Step 4: Get Risk Assumptions

> *Sarah wants to understand expected returns and volatilities*

**Query:**
```
Show me expected returns and volatilities for all asset classes
```

**Tools Used:** `get_market_based_assumptions`

**What Happens:** Agent combines FRED yields (for expected returns) with Yahoo Finance ETF historical data (for volatilities and correlations).

---

### Step 5: Analyze Risk Metrics

> *Sarah checks if any allocations are unusual compared to history*

**Query:**
```
Show me weight risk metrics and how current allocations compare to historical patterns
```

**Tools Used:** `get_weight_risk_metrics`, `get_historical_weights`

**What Happens:** Agent calculates z-scores showing how current weights compare to 10-year historical averages.

**Sample Response:**
```
Weight Risk Analysis

Asset Class           Current   Historical Avg   Z-Score   Status
--------------------------------------------------------------------
Money Market Funds    35.01%    28.50%          +1.8      Above Average
Corporate Bonds       15.00%    12.30%          +1.2      Slightly High
U.S. Treasuries       12.00%    18.40%          -2.1      Below Average
```

---

### Step 6: Evaluate a Rebalancing Move

> *Sarah considers moving money from Money Market to Treasuries*

**Query:**
```
What happens if I move 5% from Money Market Funds to U.S. Treasuries?
```

**Tools Used:** `validate_weight_change`, `simulate_rebalancing_monte_carlo`

**What Happens:**
1. Agent first validates the move against policy limits
2. Then runs 1,000 Monte Carlo simulations comparing current vs proposed allocation
3. Returns expected return change, VaR impact, and recommendation

**Sample Response:**
```
Rebalancing Analysis: Move 5% from Money Market to U.S. Treasuries

Policy Check: PASSED (both assets within limits)

Move Details:
  Amount: $2,311,728,394
  Money Market: 35.01% -> 30.01%
  U.S. Treasuries: 12.00% -> 17.00%

Monte Carlo Results (1,000 simulations, 1-year horizon):

                        Current      Proposed     Change
---------------------------------------------------------
Expected Return:        4.12%        4.28%        +0.16%
Expected Value:         $48.14B      $48.22B      +$78M
5% VaR (worst case):    $44.89B      $44.52B      -$370M
Volatility:             2.84%        3.21%        +0.37%

Assessment: Higher expected return (+$78M) but increased downside risk.
            Worst-case loss increases by $370M.
```

---

### Step 7: Check Policy Compliance

> *Sarah wants to ensure a larger move wouldn't violate policy*

**Query:**
```
Can I move 20% from Money Market to Corporate Bonds?
```

**Tools Used:** `validate_weight_change`, `check_policy_breaches`

**What Happens:** Agent checks proposed weights against policy limits, finds violation.

**Sample Response:**
```
Policy Violation Detected

Proposed Move: 20% from Money Market Funds to Corporate Bonds

Policy Limit Breach:
  Corporate Bonds: 35.00% (proposed) exceeds 35% maximum

Current: 15.00%
Proposed: 35.00%
Maximum Allowed: 35.00%

Recommendation: Reduce move to 19.99% or less to stay within policy.
```

---

### Step 8: Run Full Monte Carlo Simulation

> *Sarah wants a comprehensive risk analysis of the current portfolio*

**Query:**
```
Run a Monte Carlo simulation on the current portfolio for 1 year
```

**Tools Used:** `run_monte_carlo_simulation`

**What Happens:** Agent runs 1,000 simulations using:
- Current weights from BigQuery
- Expected returns from FRED
- Volatilities from Yahoo Finance ETF history
- Correlation matrix from ETF co-movements

**Sample Response:**
```
Monte Carlo Simulation Results
==============================

Portfolio: $46.23B | Horizon: 1 Year | Simulations: 1,000

Expected Outcome:
  Expected Value:     $48,142,345,678  (+4.12%)
  Median Value:       $48,089,234,567  (+4.01%)

Risk Metrics:
  5% VaR:            $44,892,345,678  (5% chance of losing >$1.34B)
  1% VaR:            $43,567,890,123  (1% chance of losing >$2.67B)
  Expected Shortfall: $43,123,456,789  (avg of worst 5%)

Distribution:
  Best 5%:   >$51.2B
  Top 25%:   >$49.4B
  Median:     $48.1B
  Bottom 25%: <$46.8B
  Worst 5%:  <$44.9B

Data Sources:
  - Portfolio: BigQuery (December 2025)
  - Yields: FRED API (as of 2025-12-15)
  - Volatility: Yahoo Finance (36-month ETF history)
```

---

### Step 9: Deep Dive on Single Asset

> *Sarah wants more detail on Corporate Bonds specifically*

**Query:**
```
Show me Corporate Bonds performance from January 2024 to now
```

**Tools Used:** `get_asset_class_performance`

**What Happens:** Agent queries BigQuery for monthly snapshots of just Corporate Bonds.

---

### Step 10: Final Decision

> *Sarah decides on a smaller, safer rebalancing move*

**Query:**
```
Simulate moving 2% from Money Market Funds to U.S. Treasuries using Monte Carlo
```

**Tools Used:** `simulate_rebalancing_monte_carlo`

**What Happens:** Agent runs comparison simulation, shows acceptable risk/return tradeoff.

---

## Summary: Tools Used in User Flow

| Step | User Intent | Tools Triggered |
|------|-------------|-----------------|
| 1 | See current holdings | `get_portfolio_breakdown` |
| 2 | Track quarterly performance | `compare_portfolio_performance` |
| 3 | Check market rates | `get_live_market_rates` |
| 4 | Understand return/risk assumptions | `get_market_based_assumptions` |
| 5 | Compare to historical patterns | `get_weight_risk_metrics` |
| 6 | Evaluate rebalancing | `validate_weight_change`, `simulate_rebalancing_monte_carlo` |
| 7 | Check policy limits | `validate_weight_change`, `check_policy_breaches` |
| 8 | Full risk simulation | `run_monte_carlo_simulation` |
| 9 | Single asset deep dive | `get_asset_class_performance` |
| 10 | Final decision analysis | `simulate_rebalancing_monte_carlo` |

---

## Test Prompts

### Portfolio Overview
```
What's in my portfolio?
Give me the full portfolio breakdown with dollar amounts
```

### Historical Performance
```
How has the portfolio performed since October 2024?
Show me Corporate Bonds performance from January 2023 to now
```

### Market Data
```
What are current market rates?
Show me expected returns and volatilities for all asset classes
```

### Monte Carlo Simulations
```
Run a Monte Carlo simulation on the current portfolio for 1 year
```

### Rebalancing Analysis
```
Simulate moving 5% from Money Market Funds to Corporate Bonds
What if I shift 3% from Repos to U.S. Treasuries?
```

### Policy Compliance
```
Can I move 20% from Money Market to Corporate Bonds?
Are there any policy breaches in the current allocation?
```
