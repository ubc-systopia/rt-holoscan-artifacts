#!/usr/bin/env bash

# Quick setup verification script
echo "=============================================="
echo "Backsolving Benchmarks - Setup Check"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "run_benchmarks.sh" ]; then
    echo "❌ Error: Please run this script from the backsolving-benchmarks directory"
    exit 1
fi

echo "✓ Directory structure looks good"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi
echo "✓ Docker found"

# Check NVIDIA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi
echo "✓ NVIDIA drivers found"

# Check if logged into GHCR
if docker pull ghcr.io/shubhaankar-sharma/backsolving-benchmarks:latest &>/dev/null; then
    echo "✓ GHCR image accessible"
else
    echo "❌ Cannot access GHCR image. Please login:"
    echo "   echo 'YOUR_PAT' | docker login ghcr.io -u YOUR_USERNAME --password-stdin"
fi

# Check script permissions
scripts=("run_benchmarks.sh" "run_experiments_255.sh" "run_experiments_1000.sh" "run_experiments_default.sh" "run_wcet_analysis.sh" "change_wcet.sh")
for script in "${scripts[@]}"; do
    if [ -x "$script" ]; then
        echo "✓ $script is executable"
    else
        echo "❌ $script is not executable. Run: chmod +x $script"
    fi
done

# Check uppaal directory
if [ -d "uppaal" ]; then
    echo "✓ uppaal directory present"
    if [ -f "uppaal/change_wcet.py" ]; then
        echo "✓ uppaal/change_wcet.py found"
    else
        echo "❌ uppaal/change_wcet.py missing"
    fi
else
    echo "❌ uppaal directory missing"
fi

echo ""
echo "=============================================="
echo "Setup Summary:"
echo "1. Manually set GPU speed: sudo nvidia-smi -lgc 255|1000 or sudo nvidia-smi -rgc"
echo "2. Run experiment: ./run_experiments_255.sh"
echo "3. Repeat for other speeds: ./run_experiments_1000.sh and ./run_experiments_default.sh"
echo "4. Generate WCET analysis: ./run_wcet_analysis.sh"
echo "=============================================="
