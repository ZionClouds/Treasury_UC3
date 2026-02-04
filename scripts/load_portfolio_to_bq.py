#!/usr/bin/env python3
"""
Load Illinois Treasury portfolio data from CSV to BigQuery.

Transforms wide-format CSV (months as columns) to normalized time series format.
Schema: snapshot_date, asset_class, amount, weight_pct

Usage: python3 scripts/load_portfolio_to_bq.py
"""

import csv
import os
from datetime import datetime
from google.cloud import bigquery

# GCP Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
DATASET_ID = "treasury_uc3"
TABLE_ID = "portfolio_holdings"

# Path to CSV file
CSV_PATH = "data/Illinois State Treasurer_Total Debt Investments_summary_export (2).csv"


def parse_month_year(col_name: str) -> datetime:
    """Parse 'Month YYYY' format to datetime."""
    return datetime.strptime(col_name, "%B %Y")


def parse_amount(value: str) -> float:
    """Parse amount string to float, handling empty values."""
    if not value or value.strip() == "":
        return 0.0
    return float(value.replace(",", ""))


def load_csv_to_bigquery():
    """Load and transform CSV data, then insert into BigQuery."""

    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    # Read CSV
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Extract month columns (skip first column "Instrument")
        month_columns = header[1:]

        # Parse all rows
        rows_by_month = {}  # {month_str: {asset_class: amount}}
        total_row = None

        for row in reader:
            instrument = row[0]

            if instrument == "Amount: Total":
                # Store totals for weight calculation
                total_row = {month_columns[i]: parse_amount(row[i+1])
                            for i in range(len(month_columns))}
            else:
                # Asset class row
                for i, month in enumerate(month_columns):
                    if month not in rows_by_month:
                        rows_by_month[month] = {}
                    amount = parse_amount(row[i+1]) if i+1 < len(row) else 0.0
                    rows_by_month[month][instrument] = amount

    # Transform to normalized rows with weights
    bq_rows = []

    for month_str, assets in rows_by_month.items():
        try:
            snapshot_date = parse_month_year(month_str)
        except ValueError:
            print(f"Skipping invalid month: {month_str}")
            continue

        total_amount = total_row.get(month_str, 0.0) if total_row else 0.0

        for asset_class, amount in assets.items():
            weight_pct = (amount / total_amount * 100) if total_amount > 0 else 0.0

            bq_rows.append({
                "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                "asset_class": asset_class,
                "amount": amount,
                "weight_pct": round(weight_pct, 4),
            })

    print(f"Prepared {len(bq_rows)} rows for BigQuery")

    # Create table schema
    schema = [
        bigquery.SchemaField("snapshot_date", "DATE"),
        bigquery.SchemaField("asset_class", "STRING"),
        bigquery.SchemaField("amount", "FLOAT64"),
        bigquery.SchemaField("weight_pct", "FLOAT64"),
    ]

    # Create dataset if not exists
    dataset_ref = client.dataset(DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "us-central1"
        client.create_dataset(dataset)
        print(f"Created dataset {DATASET_ID}")

    # Create or replace table
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table, exists_ok=True)
    print(f"Table {TABLE_ID} ready")

    # Delete existing data and insert new
    delete_query = f"DELETE FROM `{table_ref}` WHERE TRUE"
    try:
        client.query(delete_query).result()
        print("Cleared existing data")
    except Exception as e:
        print(f"Table was empty or new: {e}")

    # Insert rows in batches
    errors = client.insert_rows_json(table_ref, bq_rows)

    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Successfully loaded {len(bq_rows)} rows to {table_ref}")

    # Print summary
    print("\nData summary:")
    print(f"  Date range: {min(r['snapshot_date'] for r in bq_rows)} to {max(r['snapshot_date'] for r in bq_rows)}")
    print(f"  Asset classes: {len(set(r['asset_class'] for r in bq_rows))}")
    print(f"  Months: {len(rows_by_month)}")


if __name__ == "__main__":
    load_csv_to_bigquery()
