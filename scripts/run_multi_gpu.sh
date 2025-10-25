#!/bin/bash

# Multi-GPU Image Generation Script for main_negative_prompt_multi.py
# This script splits the data across 8 GPUs and runs inference in parallel

set -e  # Exit on any error

# Configuration
BENCHMARK_PATH="eval_benchmark/GenAIBenchmark/genai_image_seed.json"
OUTPUT_BASE_DIR="results/genai_benchmark"
NUM_GPUS=8
LOG_DIR="logs"
MODEL_VERSION="vRelease"
USE_QUANTIZATION=true

# Additional arguments - uncomment and modify as needed
# HUMAN_IN_LOOP=""  # Uncomment to enable human-in-the-loop
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
echo "Number of GPUs: $NUM_GPUS"
echo "==================================================="

# Check if benchmark file exists
if [ ! -f "$BENCHMARK_PATH" ]; then
    echo "Error: Benchmark file not found: $BENCHMARK_PATH"
    exit 1
fi

# Calculate data splits by reading the JSON file
echo "Calculating data splits..."
TOTAL_ITEMS=$(python3 -c "
import json
with open('$BENCHMARK_PATH', 'r') as f:
    data = json.load(f)
print(len(data))
")

echo "Total items in dataset: $TOTAL_ITEMS"

# Function to run inference on a specific GPU
run_gpu_inference() {
    local gpu_id=$1
    local log_file="$LOG_DIR/gpu_${gpu_id}.log"
    
    echo "Starting GPU $gpu_id..."
    
    # Set CUDA_VISIBLE_DEVICES to only show the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python3 main_negative_prompt_multi.py \
        --benchmark_name "$BENCHMARK_PATH" \
        --model_version "$MODEL_VERSION" \
        --gpu_id 0 \
        --total_gpus $NUM_GPUS \
        $HUMAN_IN_LOOP \
        $OPEN_LLM \
        $OPEN_LLM_MODEL \
        $OPEN_LLM_HOST \
        $OPEN_LLM_PORT \
        $CALCULATE_LATENCY \
        > "$log_file" 2>&1 &
    
    echo "GPU $gpu_id process started (PID: $!), logging to: $log_file"
    echo $! > "$LOG_DIR/gpu_${gpu_id}.pid"
}

# Start processes on each GPU
echo ""
echo "Starting parallel processes..."
echo "==================================================="

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    run_gpu_inference $gpu
    
    # Small delay to avoid race conditions
    sleep 2
done

echo ""
echo "All processes started!"
echo "==================================================="

# Function to check if all processes are still running
check_processes() {
    local running_count=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        local pid_file="$LOG_DIR/gpu_${gpu}.pid"
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
summary_file="$OUTPUT_BASE_DIR/multi_gpu_summary.json"
python3 -c "
import json
import os
import glob

output_dir = '$OUTPUT_BASE_DIR'
num_gpus = $NUM_GPUS

# Collect stats from all GPUs
total_processed = 0
total_successful = 0
total_failed = 0
gpu_stats = []

for gpu in range(num_gpus):
    gpu_dir = os.path.join(output_dir, f'gpu_{gpu}')
    stats_file = os.path.join(output_dir, f'stats_gpu_{gpu}.json')
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
        successful = stats.get('single_turn_count', 0) + stats.get('multi_turn_count', 0)
        completed_items = stats.get('completed_items', 0)
        total_items = stats.get('total_items_for_gpu', 0)
        
        gpu_stats.append({
            'gpu_id': gpu,
            'processed': completed_items,
            'total_items': total_items,
            'single_turn': stats.get('single_turn_count', 0),
            'multi_turn': stats.get('multi_turn_count', 0),
            'avg_time': sum(stats.get('end2end_times', [])) / len(stats.get('end2end_times', [1])) if stats.get('end2end_times') else 0
        })
        
        total_processed += completed_items
        total_successful += successful

# Create final summary
summary = {
    'total_gpus': num_gpus,
    'total_processed': total_processed,
    'total_successful': total_successful,
    'per_gpu_stats': gpu_stats
}

# Save summary
with open('$summary_file', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\\nSummary saved to: {summary_file}')
print(f'Total items processed: {total_processed}')
print(f'Total successful generations: {total_successful}')
"

echo "Completed multi-GPU processing!"
echo "Results are in: $OUTPUT_BASE_DIR"
echo "Summary file: $summary_file"
