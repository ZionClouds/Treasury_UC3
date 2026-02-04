#!/usr/bin/env python3
"""Test the deployed Treasury Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

import vertexai
from vertexai import agent_engines

# Configuration - UPDATE THESE VALUES or set in .env
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = "us-central1"
REASONING_ENGINE_ID = "YOUR_REASONING_ENGINE_ID"  # From deploy.py output

# Initialize
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Get the deployed agent
RESOURCE_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{REASONING_ENGINE_ID}"
engine = agent_engines.get(RESOURCE_NAME)

# Test query
print("Testing deployed Treasury Agent...")
print("=" * 50)
response = engine.query(input="What is the current portfolio breakdown?")
print(response)
