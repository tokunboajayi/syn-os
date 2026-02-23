#!/bin/bash

# Syn OS Kiosk Mode Startup Script

set -e

# Hide mouse cursor initially
unclutter -idle 0.1 -root &

# 1. Start Docker Services
echo "Starting Syn OS Backend..."
cd /opt/syn-os/infra/docker
docker-compose up -d

# 2. Wait for API Health
echo "Waiting for Neural Engine..."
until curl -s http://localhost:8000/health > /dev/null; do
    sleep 2
done

# 3. Launch UI (Electron)
echo "Launching Interface..."
# Assuming compiled AppImage or unpacked binary is available
# For development/source run:
cd /opt/syn-os/syn-os-edex
npm start -- --fullscreen --kiosk

# Recovery: Restart UI if it crashes
while true; do
    npm start -- --fullscreen --kiosk
    sleep 5
done
