#!/bin/bash

echo "Starting on port $PORT"

# Substitute env vars in agent config
envsubst < /root/agent-config.yaml > /tmp/agent-config.yaml

# Start Grafana Agent in background
grafana-agent --config.file=/tmp/agent-config.yaml &

echo "Strating Uvicorn server on port $PORT"
# Start FastAPI app on Render's port
uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT
