#!/bin/bash

SPEED=${1:-default}

# Default parameters
INSTANCES=1
REPEATS=3
MODELS=200
SCHED="eventbased"
LANGUAGE="python"

# Define app list with any special args
declare -A APP_ARGS
APP_ARGS["endoscopy_depth_estimation"]=""
APP_ARGS["colonoscopy_segmentation"]=""
APP_ARGS["body_pose_estimation"]="--source replayer"
APP_ARGS["endoscopy_out_of_body_detection"]=""
APP_ARGS["multiai_endoscopy"]=""
APP_ARGS["multiai_ultrasound"]=""

launch_container_env() {
    make devcontainer-run
}

run_benchmark() {
    local app=$1
    local run_args=$2
    echo "=========================================="
    echo "Running benchmark for: $app (GPU=$SPEED)"
    echo "=========================================="

    if [ -n "$run_args" ]; then
        make APP=$app INSTANCES=$INSTANCES REPEATS=$REPEATS MODELS=$MODELS SCHED=$SCHED LANGUAGE=$LANGUAGE RUN_ARG="$run_args" GRAPH_DIR="application_latencies_$SPEED" all-args
    else
        make APP=$app INSTANCES=$INSTANCES REPEATS=$REPEATS MODELS=$MODELS SCHED=$SCHED LANGUAGE=$LANGUAGE GRAPH_DIR="application_latencies_$SPEED" all
    fi
}

# Run all benchmarks
for app in "${!APP_ARGS[@]}"; do
    run_benchmark "$app" "${APP_ARGS[$app]}"
done

# Special CLAHE variant
echo "=========================================="
echo "Running benchmark for: endoscopy_depth_estimation (CLAHE variant, GPU=$SPEED)"
echo "=========================================="
make APP=endoscopy_depth_estimation APP_ARGS="--clahe" INSTANCES=$INSTANCES REPEATS=$REPEATS MODELS=$MODELS SCHED=$SCHED LANGUAGE=$LANGUAGE GRAPH_DIR="application_latencies_$SPEED" OUTDIR=endoscopy_depth_estimation_clahe_results GRAPH_NAME=endoscopy_depth_estimation_clahe_graph all


echo "=========================================="
echo "Updating WCET values"
echo "=========================================="
./change_wcet.sh $SPEED