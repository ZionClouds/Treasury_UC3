#!/bin/bash
#
# Register Treasury Agent with Gemini Enterprise
#
# This script registers your deployed Reasoning Engine (Agent Engine)
# as an ADK agent in Gemini Enterprise using the Discovery Engine REST API.
#
# Prerequisites:
#   1. Deploy the agent first: python3 deploy.py
#   2. Enable the Discovery Engine API
#   3. Have the Discovery Engine Admin role
#
# Usage:
#   chmod +x connect_gemini.sh
#   ./connect_gemini.sh

set -e

# Configuration
PROJECT_ID="project-zion-454116"
REASONING_ENGINE_ID="1100077876264304640"
REASONING_ENGINE_LOCATION="us-central1"

# Gemini Enterprise App ID
APP_ID="master-agent_1747926439088"

# API location (us, eu, or global)
LOCATION="global"

# Agent display settings
DISPLAY_NAME="Treasury Portfolio Intelligence Agent"
DESCRIPTION="AI agent for the Illinois State Treasurer's \$46B fixed income portfolio. Ask questions about portfolio holdings, historical performance, current market rates, Monte Carlo simulations, and rebalancing scenarios."

# Determine the API endpoint prefix
case "$LOCATION" in
    us)      ENDPOINT_PREFIX="us-" ;;
    eu)      ENDPOINT_PREFIX="eu-" ;;
    global)  ENDPOINT_PREFIX="" ;;
    *)
        echo "ERROR: LOCATION must be 'us', 'eu', or 'global'. Got: $LOCATION"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Registering Agent with Gemini Enterprise"
echo "============================================================"
echo ""
echo "  Project:            $PROJECT_ID"
echo "  Reasoning Engine:   $REASONING_ENGINE_ID"
echo "  Engine Location:    $REASONING_ENGINE_LOCATION"
echo "  App ID:             $APP_ID"
echo "  API Location:       $LOCATION"
echo "  Display Name:       $DISPLAY_NAME"
echo ""

# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Register the agent via Discovery Engine REST API
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "https://${ENDPOINT_PREFIX}discoveryengine.googleapis.com/v1alpha/projects/${PROJECT_ID}/locations/${LOCATION}/collections/default_collection/engines/${APP_ID}/assistants/default_assistant/agents" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
        \"displayName\": \"${DISPLAY_NAME}\",
        \"description\": \"${DESCRIPTION}\",
        \"adk_agent_definition\": {
            \"provisioned_reasoning_engine\": {
                \"reasoning_engine\": \"projects/${PROJECT_ID}/locations/${REASONING_ENGINE_LOCATION}/reasoningEngines/${REASONING_ENGINE_ID}\"
            }
        }
    }")

# Extract HTTP status code and body
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP Status: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
    echo "============================================================"
    echo "  AGENT REGISTERED SUCCESSFULLY"
    echo "============================================================"
    echo ""
    echo "Response:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    echo ""
    echo "Next steps:"
    echo "  1. Go to: https://console.cloud.google.com/gemini-enterprise/"
    echo "  2. Open your app"
    echo "  3. Click 'Agents' to see your registered agent"
    echo "  4. Users can now interact with it through Gemini Enterprise"
    echo ""
else
    echo "============================================================"
    echo "  REGISTRATION FAILED"
    echo "============================================================"
    echo ""
    echo "Response:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    echo ""
    echo "Common issues:"
    echo "  - 403: Missing Discovery Engine Admin role or API not enabled"
    echo "  - 404: App ID is wrong, or app doesn't exist in this project/location"
    echo "  - 400: Reasoning Engine ID is invalid or in wrong region"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Enable Discovery Engine API:"
    echo "     gcloud services enable discoveryengine.googleapis.com --project=$PROJECT_ID"
    echo ""
    echo "  2. Check your Gemini Enterprise app:"
    echo "     https://console.cloud.google.com/gemini-enterprise/"
    echo ""
    echo "  3. Verify Reasoning Engine exists:"
    echo "     gcloud ai reasoning-engines list --project=$PROJECT_ID --region=$REASONING_ENGINE_LOCATION"
    echo ""
    exit 1
fi
