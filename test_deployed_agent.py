#!/usr/bin/env python3
"""Test the deployed Treasury Agent."""

import vertexai
from vertexai import agent_engines

# Initialize
vertexai.init(project="project-zion-454116", location="us-central1")

# Get the deployed agent
RESOURCE_NAME = "projects/343109752014/locations/us-central1/reasoningEngines/1100077876264304640"
engine = agent_engines.get(RESOURCE_NAME)

# Test query
print("Testing deployed Treasury Agent...")
print("=" * 50)
response = engine.query(input="What is the current portfolio breakdown?")
print(response)
