#!/bin/bash

# Get port from Render (defaults to 8000 for local)
PORT=${PORT:-8000}

# Substitute env vars in agent config
envsubst < /app/agent-config.yaml > /tmp/agent-config.yaml

# Start Grafana Agent in background
grafana-agent --config.file=/tmp/agent-config.yaml &

# Start FastAPI app on Render's port
uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT
