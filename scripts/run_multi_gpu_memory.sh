#!/bin/bash

# Multi-GPU Image Generation Script for main_negative_memory_multi.py
# This script splits the data across multiple GPUs and runs inference in parallel

set -e  # Exit on any error

# Configuration
BENCHMARK_PATH="DrawBench_seed.txt"  # Changed to use DrawBench file
OUTPUT_BASE_DIR="results/DrawBench-fixseed"  # Updated to match benchmark name
LOG_DIR="logs"
MODEL_VERSION="vRelease"
USE_QUANTIZATION=true

# GPU Configuration - modify these arrays to specify which GPUs to use
GPU_IDS=(1)  # Add or remove GPU IDs as needed
NUM_GPUS=${#GPU_IDS[@]}

# Additional arguments - uncomment and modify as needed
# HUMAN_IN_LOOP="--human_in_the_loop"  # Uncomment to enable human-in-the-loop
# OPEN_LLM="--use_open_llm"
# OPEN_LLM_MODEL="--open_llm_model mistralai/Mistral-Small-3.1-24B-Instruct-2503"
# OPEN_LLM_HOST="--open_llm_host 0.0.0.0"
# OPEN_LLM_PORT="--open_llm_port 8000"
# CALCULATE_LATENCY="--calculate_latency"

# Create directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Multi-GPU Image Generation Setup"
echo "==================================================="
echo "Benchmark: $BENCHMARK_PATH"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "GPU IDs: ${GPU_IDS[*]}"
echo "Number of GPUs: $NUM_GPUS"
echo "==================================================="

# Calculate total data items
echo "Calculating data splits..."
TOTAL_ITEMS=$(python3 -c "
import json
import os

file_path = 'eval_benchmark/$BENCHMARK_PATH'
file_ext = os.path.splitext(file_path)[1].lower()

try:
    if file_ext == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(len(data))
    elif file_ext == '.txt':
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            print(len(lines))
    else:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(len(data))
        except json.JSONDecodeError:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                print(len(lines))
except Exception as e:
    print(f'Error: {str(e)}')
    print('0')
")

echo "Total items in dataset: $TOTAL_ITEMS"

# Function to run inference on a specific GPU
run_gpu_inference() {
    local gpu_physical_id=$1
    local gpu_logical_id=$2
    local log_file="$LOG_DIR/gpu_${gpu_physical_id}.log"

    # Calculate split indices for this GPU
    local items_per_gpu=$((TOTAL_ITEMS / NUM_GPUS))
    local remainder=$((TOTAL_ITEMS % NUM_GPUS))
    local start_idx
    local end_idx

    if [ $gpu_logical_id -lt $remainder ]; then
        start_idx=$((gpu_logical_id * (items_per_gpu + 1)))
        end_idx=$((start_idx + items_per_gpu))
    else
        start_idx=$((gpu_logical_id * items_per_gpu + remainder))
        end_idx=$((start_idx + items_per_gpu - 1))
    fi

    echo "Starting GPU $gpu_physical_id (logical ID $gpu_logical_id) with indices $start_idx to $end_idx..."

    CUDA_VISIBLE_DEVICES=$gpu_physical_id python3 main_negative_memory_multi.py \
        --benchmark_name "$BENCHMARK_PATH" \
        --model_version "$MODEL_VERSION" \
        --gpu_id $gpu_logical_id \
        --total_gpus $NUM_GPUS \
        --start_idx $start_idx \
        --end_idx $end_idx \
        $HUMAN_IN_LOOP \
        $OPEN_LLM \
        $OPEN_LLM_MODEL \
        $OPEN_LLM_HOST \
        $OPEN_LLM_PORT \
        $CALCULATE_LATENCY \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "GPU $gpu_physical_id process started (PID: $pid), logging to: $log_file"
    echo $pid > "$LOG_DIR/gpu_${gpu_physical_id}.pid"
}

# Start processes on each GPU
echo ""
echo "Starting parallel processes..."
echo "==================================================="

for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    run_gpu_inference $gpu_id $i
    sleep 2
done

echo ""
echo "All processes started!"
echo "==================================================="

# Function to check if all processes are still running
check_processes() {
    local running_count=0
    for gpu_id in "${GPU_IDS[@]}"; do
        local pid_file="$LOG_DIR/gpu_${gpu_id}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                running_count=$((running_count + 1))
            fi
        fi
    done
    echo $running_count
}

# Monitor processes
echo "Monitoring processes..."
echo "Press Ctrl+C to stop monitoring (processes will continue running)"
echo ""

start_time=$(date +%s)

while true; do
    running=$(check_processes)
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    echo -ne "\rRunning processes: $running/$NUM_GPUS | Elapsed time: ${elapsed}s"

    if [ $running -eq 0 ]; then
        echo ""
        echo "All processes completed!"
        break
    fi

    sleep 10
done

echo ""
echo "==================================================="
echo "Generation Summary"
echo "==================================================="

# Consolidate results from all GPUs
echo "Consolidating results from all GPUs..."

# Create a summary file
summary_file="$OUTPUT_BASE_DIR/AgentSys_${MODEL_VERSION}/multi_gpu_summary.json"
mkdir -p "$(dirname "$summary_file")"

python3 -c "
import json
import os
import glob

base_output_dir = '$OUTPUT_BASE_DIR/AgentSys_$MODEL_VERSION'
gpu_ids = [$(echo ${GPU_IDS[*]} | sed 's/ /,/g')]

benchmark_path = 'eval_benchmark/$BENCHMARK_PATH'
file_ext = os.path.splitext(benchmark_path)[1].lower()
is_json_format = file_ext == '.json'

print(f'Detected benchmark format: {\"JSON\" if is_json_format else \"Text (e.g., DrawBench)\"}')

total_processed = 0
total_successful = 0
total_failed = 0
gpu_stats = []

for gpu_id in gpu_ids:
    gpu_dir = os.path.join(base_output_dir, f'gpu_{gpu_id}')
    stats_file = os.path.join(gpu_dir, 'stats.json')

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        successful = stats.get('single_turn_count', 0) + stats.get('multi_turn_count', 0)
        completed_items = stats.get('completed_items', 0)
        total_items = stats.get('total_items_for_gpu', 0)

        gpu_stats.append({
            'gpu_id': gpu_id,
            'processed': completed_items,
            'total_items': total_items,
            'single_turn': stats.get('single_turn_count', 0),
            'multi_turn': stats.get('multi_turn_count', 0),
            'avg_time': sum(stats.get('end2end_times', [])) / len(stats.get('end2end_times', [1])) if stats.get('end2end_times') else 0
        })

        total_processed += completed_items
        total_successful += successful

summary = {
    'total_gpus': len(gpu_ids),
    'gpu_ids_used': gpu_ids,
    'total_processed': total_processed,
    'total_successful': total_successful,
    'per_gpu_stats': gpu_stats
}

with open('$summary_file', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\\nSummary saved to: $summary_file')
print(f'Total items processed: {total_processed}')
print(f'Total successful generations: {total_successful}')
print('\\nPer-GPU Statistics:')
for stat in gpu_stats:
    print(f'  GPU {stat[\"gpu_id\"]}: {stat[\"processed\"]}/{stat[\"total_items\"]} items processed')
"

echo "Completed multi-GPU processing!"
echo "Results are in: $OUTPUT_BASE_DIR/AgentSys_${MODEL_VERSION}/"
echo "Summary file: $summary_file"

# Optional: Clean up PID files
for gpu_id in "${GPU_IDS[@]}"; do
    pid_file="$LOG_DIR/gpu_${gpu_id}.pid"
    if [ -f "$pid_file" ]; then
        rm "$pid_file"
    fi
done

echo "Multi-GPU processing completed successfully!"