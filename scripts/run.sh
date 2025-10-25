#!/bin/bash

# Multi-GPU Image Generation Script
# This script splits the data across 7 GPUs (CUDA 0-6) and runs inference in parallel

set -e  # Exit on any error

# Configuration
SCRIPT_NAME="err.py"  # Change this to your actual script name (base_run_negative.py)
BENCHMARK_PATH="/mnt/localssd/shivank/agents/agents-image-gen/eval_benchmark/DrawBench_seed.json"
OUTPUT_BASE_DIR="results/multi_gpu_qwen_draebench"
NUM_GPUS=7
LOG_DIR="logs"

# Create directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Multi-GPU Image Generation Setup"
echo "==================================================="
echo "Script: $SCRIPT_NAME"
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

# Calculate items per GPU
ITEMS_PER_GPU=$((TOTAL_ITEMS / NUM_GPUS))
REMAINDER=$((TOTAL_ITEMS % NUM_GPUS))

echo "Items per GPU: $ITEMS_PER_GPU"
echo "Remainder items: $REMAINDER"

# Function to run inference on a specific GPU
run_gpu_inference() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local output_dir="$OUTPUT_BASE_DIR/gpu_$gpu_id"
    local log_file="$LOG_DIR/gpu_${gpu_id}.log"
    
    echo "Starting GPU $gpu_id: processing items $start_idx to $((end_idx-1))"
    
    # Set CUDA_VISIBLE_DEVICES to only show the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python3 "$SCRIPT_NAME" \
        --benchmark_path "$BENCHMARK_PATH" \
        --output_dir "$output_dir" \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --gpu_id 0 \
        --use_quantization \
        > "$log_file" 2>&1 &
    
    echo "GPU $gpu_id process started (PID: $!), logging to: $log_file"
    echo $! > "$LOG_DIR/gpu_${gpu_id}.pid"
}

# Start processes on each GPU
echo ""
echo "Starting parallel processes..."
echo "==================================================="

START_IDX=0
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    # Calculate end index for this GPU
    CURRENT_ITEMS=$ITEMS_PER_GPU
    if [ $gpu -lt $REMAINDER ]; then
        CURRENT_ITEMS=$((CURRENT_ITEMS + 1))
    fi
    
    END_IDX=$((START_IDX + CURRENT_ITEMS))
    
    # Ensure we don't exceed total items
    if [ $END_IDX -gt $TOTAL_ITEMS ]; then
        END_IDX=$TOTAL_ITEMS
    fi
    
    # Run inference on this GPU
    run_gpu_inference $gpu $START_IDX $END_IDX
    
    # Update start index for next GPU
    START_IDX=$END_IDX
    
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

# Collect results from all GPUs
total_successful=0
total_failed=0
total_time=0

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    stats_file="$OUTPUT_BASE_DIR/gpu_$gpu/final_stats_gpu_$gpu.json"
    log_file="$LOG_DIR/gpu_${gpu}.log"
    
    echo "GPU $gpu:"
    if [ -f "$stats_file" ]; then
        successful=$(python3 -c "
import json
try:
    with open('$stats_file', 'r') as f:
        data = json.load(f)
    print(data.get('successful_generations', 0))
except:
    print(0)
")
        failed=$(python3 -c "
import json
try:
    with open('$stats_file', 'r') as f:
        data = json.load(f)
    print(data.get('failed_generations', 0))
except:
    print(0)
")
        gpu_time=$(python3 -c "
import json
try:
    with open('$stats_file', 'r') as f:
        data = json.load(f)
    print(data.get('total_time', 0))
except:
    print(0)
")
        
        echo "  Successful: $successful"
        echo "  Failed: $failed" 
        echo "  Time: ${gpu_time}s"
        
        total_successful=$((total_successful + successful))
        total_failed=$((total_failed + failed))
        total_time=$(python3 -c "print(max($total_time, $gpu_time))")  # Max time since parallel
    else
        echo "  No stats file found - check log: $log_file"
        # Show last few lines of log for debugging
        if [ -f "$log_file" ]; then
            echo "  Last log lines:"
            tail -3 "$log_file" | sed 's/^/    /'
        fi
    fi
    echo ""
done

echo "==================================================="
echo "FINAL RESULTS"
echo "==================================================="
echo "Total successful generations: $total_successful"
echo "Total failed generations: $total_failed"
echo "Total items processed: $((total_successful + total_failed))"
echo "Expected items: $TOTAL_ITEMS"
echo "Wall clock time: ${total_time}s"
echo ""
echo "Output directories:"
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    echo "  GPU $gpu: $OUTPUT_BASE_DIR/gpu_$gpu"
done
echo ""
echo "Log files: $LOG_DIR/"
echo "==================================================="

# Clean up PID files
rm -f "$LOG_DIR"/*.pid

# Create a combined results summary
python3 -c "
import json
import os
from glob import glob

summary = {
    'total_gpus': $NUM_GPUS,
    'total_items': $TOTAL_ITEMS,
    'total_successful': $total_successful,
    'total_failed': $total_failed,
    'wall_clock_time': $total_time,
    'gpu_results': {}
}

for gpu in range($NUM_GPUS):
    stats_file = f'$OUTPUT_BASE_DIR/gpu_{gpu}/final_stats_gpu_{gpu}.json'
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            gpu_stats = json.load(f)
        summary['gpu_results'][f'gpu_{gpu}'] = gpu_stats
    else:
        summary['gpu_results'][f'gpu_{gpu}'] = {'error': 'No stats file found'}

with open('$OUTPUT_BASE_DIR/combined_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Combined results saved to: $OUTPUT_BASE_DIR/combined_results.json')
"

echo "Multi-GPU generation complete!"
