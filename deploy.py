#!/usr/bin/env python3
"""
Deploy Treasury Agent to Google Agent Engine (Vertex AI).

Usage:
    python3 deploy.py

Requirements:
    - .env file with FRED_API_KEY
    - gcloud authenticated with appropriate permissions
    - Staging bucket must exist: gs://{PROJECT_ID}-treasury-agent-staging
"""

import io
import logging
import os
from typing import Any

# Load .env file BEFORE importing treasury_agent (which needs env vars)
from dotenv import load_dotenv
load_dotenv()

import vertexai
from vertexai import agent_engines

from treasury_agent import root_agent

# Configuration - Set GOOGLE_CLOUD_PROJECT in your .env file
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required. Set it in your .env file.")
REGION = "us-central1"
STAGING_BUCKET = f"gs://{PROJECT_ID}-treasury-agent-staging"
DISPLAY_NAME = "treasury_portfolio_agent"

# Environment variables for logging and telemetry
ENV_VARS = {
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    "FRED_API_KEY": os.environ.get("FRED_API_KEY", ""),
    "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
}

# Initialize Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=STAGING_BUCKET
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_requirements() -> list[str]:
    """Read requirements.txt and return non-empty, non-comment lines."""
    path = "requirements.txt"
    try:
        with open(path, "r") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    except FileNotFoundError:
        logger.warning("requirements.txt not found; proceeding without extra requirements.")
        return []


NON_SERIALIZABLE_TYPES: tuple[type, ...] = (io.TextIOBase, logging.Handler)


def _scrub_in_place(obj: Any, seen: set[int] | None = None) -> Any:
    """Remove non-serializable attributes that break deployment."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return obj
    seen.add(oid)

    # Strip common non-serializable attributes
    for attr in ("logger", "_logger", "log_stream", "handler", "stream", "stdout", "stderr"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                if isinstance(val, NON_SERIALIZABLE_TYPES):
                    setattr(obj, attr, None)
            except Exception:
                pass

    # Handle dicts
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, NON_SERIALIZABLE_TYPES):
                obj[k] = None
            else:
                _scrub_in_place(v, seen)
        return obj

    # Handle lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        for item in list(obj):
            _scrub_in_place(item, seen)
        return obj

    # Handle object attributes
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if isinstance(v, NON_SERIALIZABLE_TYPES):
                try:
                    setattr(obj, k, None)
                except Exception:
                    pass
            else:
                _scrub_in_place(v, seen)

    return obj


if __name__ == "__main__":
    logger.info(f"Deploying Treasury Agent to project: {PROJECT_ID}")
    logger.info(f"Region: {REGION}")
    logger.info(f"Staging bucket: {STAGING_BUCKET}")

    # Get agent instance
    agent_instance = root_agent() if callable(root_agent) else root_agent

    # Remove non-serializable attributes
    _scrub_in_place(agent_instance)
    for t in getattr(agent_instance, "tools", []) or []:
        _scrub_in_place(t)

    # Wrap in AdkApp
    app = agent_engines.AdkApp(agent=agent_instance, enable_tracing=True)

    # Load requirements
    reqs = load_requirements()
    if reqs:
        logger.info("Requirements to install remotely:")
        for r in reqs:
            logger.info(f"  - {r}")

    # Deploy
    logger.info("Creating Agent Engine...")
    remote_app = agent_engines.create(
        app,
        display_name=DISPLAY_NAME,
        requirements=reqs,
        extra_packages=["./treasury_agent"],
        env_vars=ENV_VARS,
    )

    # Print success info
    resource_name = remote_app.resource_name
    engine_id = resource_name.split("/")[-1]

    print()
    print("=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"Resource Name: {resource_name}")
    print(f"AGENT_ENGINE ID: {engine_id}")
    print()
    print("To connect to Gemini, update connect_gemini.sh with:")
    print(f'  REASONING_ENGINE="{engine_id}"')
    print()
