FROM python:3.12-slim

# Set working directory inside container
WORKDIR /root

# Copy dependency file
COPY app/backend/requirements.txt /root/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /root/requirements.txt

# Install Grafana Agent and dependencies
RUN apt-get update && apt-get install -y wget gettext-base unzip && \
    wget -q https://github.com/grafana/agent/releases/download/v0.40.2/grafana-agent-linux-amd64.zip && \
    unzip grafana-agent-linux-amd64.zip && \
    mv grafana-agent-linux-amd64 /usr/local/bin/grafana-agent && \
    chmod +x /usr/local/bin/grafana-agent && \
    rm grafana-agent-linux-amd64.zip && \
    apt-get clean

# Copy backend source code
COPY app/backend /root/app/backend
COPY agent-config.yaml /root/agent-config.yaml
COPY start.sh /root/start.sh

RUN chmod +x /root/start.sh

# Expose FastAPI port
EXPOSE $PORT

# Start the backend
CMD ["/root/start.sh"]