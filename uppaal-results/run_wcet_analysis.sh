#!/usr/bin/env bash
set -euo pipefail

# Run WCET analysis on all collected benchmark data
# This should be run AFTER all three benchmark experiments are completed

echo "=============================================="
echo "Running WCET Analysis"
echo "=============================================="

# Check if all required result directories exist
missing_dirs=()
for speed in "255" "1000" "default"; do
    if [ ! -d "application_latencies_$speed" ]; then
        missing_dirs+=("application_latencies_$speed")
    fi
done

if [ ${#missing_dirs[@]} -ne 0 ]; then
    echo "Error: Missing result directories:"
    for dir in "${missing_dirs[@]}"; do
        echo "  - $dir"
    done
    echo ""
    echo "Please run the benchmark experiments first:"
    echo "  ./run_experiments_255.sh /path/to/holohub"
    echo "  ./run_experiments_1000.sh /path/to/holohub"
    echo "  ./run_experiments_default.sh /path/to/holohub"
    exit 1
fi

echo "Found all required result directories:"
for speed in "255" "1000" "default"; do
    echo "  ✓ application_latencies_$speed"
done
echo ""

# Run WCET analysis for each speed
for speed in "255" "1000" "default"; do
    echo "Analyzing results for GPU speed: $speed"
    ./change_wcet.sh "$speed"
    echo ""
done

echo "=============================================="
echo "WCET Analysis completed!"
echo "Check the uppaal/ directory for updated XML files"
echo "=============================================="
