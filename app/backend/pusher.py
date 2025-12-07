"""
Grafana Cloud metrics pusher.
"""
import threading
import time
import requests
import base64
import os
from prometheus_client import generate_latest
import logging

logger = logging.getLogger(__name__)

def push_to_grafana(interval: int = 15):
    """Push metrics to Grafana Cloud continuously."""
    url = os.getenv("GRAFANA_URL")
    user = os.getenv("GRAFANA_USER")
    password = os.getenv("GRAFANA_PASSWORD")
    
    if not all([url, user, password]):
        logger.warning("Grafana credentials missing. Metrics won't be pushed.")
        return
    
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    logger.info(f"Grafana pusher started (interval: {interval}s)")
    
    while True:
        try:
            metrics = generate_latest()
            requests.post(
                url,
                data=metrics,
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/openmetrics-text"
                },
                timeout=10
            )
        except Exception as e:
            logger.error(f"Metrics push failed: {e}")
        time.sleep(interval)


def start_pusher(interval: int = 15):
    """Start pusher in background thread."""
    thread = threading.Thread(target=push_to_grafana, args=(interval,), daemon=True)
    thread.start()