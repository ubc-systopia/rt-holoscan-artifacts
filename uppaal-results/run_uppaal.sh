#!/bin/bash

# UPPAAL Analysis Runner
# This script runs the UPPAAL analysis using the pre-built Docker image

set -e

# Configuration
IMAGE_NAME="ghcr.io/shubhaankar-sharma/backsolving-uppaal:latest"
RESULTS_DIR="./results"

echo "=== UPPAAL Analysis Pipeline ==="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if image exists locally, if not try to pull
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Pulling UPPAAL image from registry..."
    if ! docker pull "$IMAGE_NAME"; then
        echo "Failed to pull from registry, using local image if available..."
    fi
else
    echo "Using local UPPAAL image..."
fi

echo "Running UPPAAL analysis..."
echo "Using UPPAAL 4.1.26 Academic Edition (no license required)"

# Start container in background (volume mounting has issues on this system)
echo "Starting UPPAAL container..."
# Add memory + swap headroom to avoid OOM
CONTAINER_ID=$(docker run -d --memory=128g --memory-swap=256g "$IMAGE_NAME" sleep 3600)

# Copy input files into container
echo "Copying analysis files into container..."
docker cp uppaal/ "$CONTAINER_ID:/workspace/uppaal"

# Run the analysis per-config to reduce peak memory
CONFIGS=("255" "1000" "default")
for config in "${CONFIGS[@]}"; do
    echo "Running analysis (${config}, no plot)..."
    docker exec "$CONTAINER_ID" bash -lc "python3 /workspace/uppaal/graph_pipeline.py \
      /workspace/uppaal/UPPAALXML \
      /workspace/uppaal/UPPAALXML/query.q \
      --verifyta_path /opt/uppaal/bin-Linux/verifyta \
      --output_dir /workspace/results_${config} \
      --configs ${config} \
      --no_plot \
      --extra_args -S 2 -s"
    # Copy intermediate results back to host after each config
    mkdir -p "${RESULTS_DIR}/${config}"
    docker cp "$CONTAINER_ID:/workspace/results_${config}/." "${RESULTS_DIR}/${config}/" || echo "Failed to copy results for ${config}"
done

echo "Copying results from container..."
docker cp "$RESULTS_DIR/" "$CONTAINER_ID:/workspace/results" 


echo "Combining results from separate config runs..."
docker exec "$CONTAINER_ID" bash -lc "python3 - << 'PY'
import os, pandas as pd
base = '/workspace/results'
config_dirs = ['/workspace/results/255/', '/workspace/results/1000/', '/workspace/results/defaul/']
frames = []
for config_dir in config_dirs:
    csv_file = os.path.join(config_dir, 'uppaal_results.csv')
    if os.path.exists(csv_file):
        print(f'Reading {csv_file}')
        frames.append(pd.read_csv(csv_file))
    else:
        print(f'Warning: {csv_file} not found')
df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['Application','CPU_User_Time','Config'])
df.drop_duplicates(subset=['Application','Config'], keep='last', inplace=True)
os.makedirs(base, exist_ok=True)
df.to_csv(os.path.join(base, 'uppaal_results.csv'), index=False)
print(f'Combined CSV written to {base}/uppaal_results.csv with {len(df)} rows')
PY"

echo "Generating plots from combined data..."
docker exec "$CONTAINER_ID" bash -lc "python3 /workspace/uppaal/graph_pipeline.py \
  /workspace/uppaal/UPPAALXML \
  /workspace/uppaal/UPPAALXML/query.q \
  --verifyta_path /opt/uppaal/bin-Linux/verifyta \
  --output_dir /workspace/results \
  --configs 255,1000,default \
  --plot_only"

# Copy results back from container
echo "Copying results from container..."
docker cp "$CONTAINER_ID:/workspace/results/." "$RESULTS_DIR/"

# Stop the container
echo "Cleaning up..."
docker stop "$CONTAINER_ID" > /dev/null 2>&1 || echo "Container cleanup completed"

echo "=== Analysis completed! ==="
echo "Results are available in the '$RESULTS_DIR' directory:"
echo "- uppaal_results.csv: Raw data (combined)"
echo "- UPPAAL.png: Combined comparison plot (PNG format)"
echo "- UPPAAL.pdf: Combined comparison plot (PDF format)"
echo ""
echo "The plot compares all three configurations (255 MHz, 1000 MHz, 215-3105 MHz) for each application."
