#!/bin/bash

# Multi-GPU Image Generation Script for main_negative_memory_multi.py
# This script splits the data across multiple GPUs and runs inference in parallel

set -e  # Exit on any error

# Configuration
BENCHMARK_PATH="missing_images_ep2_oneig.csv"
OUTPUT_BASE_DIR="results/One-Ig-Ep-2-remain.txt"
LOG_DIR="logs"
MODEL_VERSION="vRelease"
USE_QUANTIZATION=true

# GPU Configuration - modify these arrays to specify which GPUs to use
GPU_IDS=(3) #1 2 3 4 5 6 7)  # Add or remove GPU IDs as needed
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

# # Check if benchmark file exists
# if [ ! -f "$BENCHMARK_PATH" ]; then
#     echo "Error: Benchmark file not found: $BENCHMARK_PATH"
#     exit 1
# fi

# Calculate total data items
echo "Calculating data splits..."
TOTAL_ITEMS=$(python3 -c "
import csv
with open('eval_benchmark/$BENCHMARK_PATH', 'r') as f:
    reader = csv.DictReader(f)
    count = sum(1 for row in reader)
print(count)
")

echo "Total items in dataset: $TOTAL_ITEMS"

# Function to run inference on a specific GPU
run_gpu_inference() {
    local gpu_physical_id=$1
    local gpu_logical_id=$2
    local log_file="$LOG_DIR/gpu_${gpu_physical_id}.log"
    
    echo "Starting GPU $gpu_physical_id (logical ID $gpu_logical_id)..."
    
    # Set CUDA_VISIBLE_DEVICES to only show the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_physical_id python main_negative_memory_multi_new.py \
        --benchmark_name "$BENCHMARK_PATH" \
        --model_version "$MODEL_VERSION" \
        --gpu_id $gpu_logical_id \
        --total_gpus $NUM_GPUS \
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
    sleep 3
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
summary_file="$OUTPUT_BASE_DIR/${MODEL_VERSION}/multi_gpu_summary.json"
mkdir -p "$(dirname "$summary_file")"

python3 -c "
import json
import os
import glob

base_output_dir = '$OUTPUT_BASE_DIR/$MODEL_VERSION'
gpu_ids = [$(echo ${GPU_IDS[*]} | sed 's/ /,/g')]

# Collect stats from all GPUs
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

# Create final summary
summary = {
    'total_gpus': len(gpu_ids),
    'gpu_ids_used': gpu_ids,
    'total_processed': total_processed,
    'total_successful': total_successful,
    'per_gpu_stats': gpu_stats
}

# Save summary
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
echo "Results are in: $OUTPUT_BASE_DIR/${MODEL_VERSION}/"
echo "Summary file: $summary_file"

# Consolidate images from all GPU folders into final structure
echo ""
echo "Consolidating images from multi-GPU processing..."
echo "==================================================="

python3 -c "
import os
import shutil
import glob
from pathlib import Path

base_output_dir = '$OUTPUT_BASE_DIR/$MODEL_VERSION'
gpu_ids = [$(echo ${GPU_IDS[*]} | sed 's/ /,/g')]

# Create final consolidated images directory
final_images_dir = os.path.join(base_output_dir, 'images')
os.makedirs(final_images_dir, exist_ok=True)

consolidated_count = 0

# Process each GPU's output
for gpu_id in gpu_ids:
    gpu_dir = os.path.join(base_output_dir, f'gpu_{gpu_id}')
    gpu_images_dir = os.path.join(gpu_dir, 'images')
    
    if os.path.exists(gpu_images_dir):
        # Walk through all category folders in this GPU's images directory
        for category_dir in os.listdir(gpu_images_dir):
            category_path = os.path.join(gpu_images_dir, category_dir)
            if os.path.isdir(category_path):
                # Create category directory in final location
                final_category_dir = os.path.join(final_images_dir, category_dir)
                final_model_dir = os.path.join(final_category_dir, 'qwen')
                os.makedirs(final_model_dir, exist_ok=True)
                
                # Copy all images from this GPU's category/qwen folder
                gpu_model_dir = os.path.join(category_path, 'qwen')
                if os.path.exists(gpu_model_dir):
                    for image_file in os.listdir(gpu_model_dir):
                        if image_file.endswith('.webp'):
                            src_path = os.path.join(gpu_model_dir, image_file)
                            dst_path = os.path.join(final_model_dir, image_file)
                            shutil.copy2(src_path, dst_path)
                            consolidated_count += 1

print(f'Consolidated {consolidated_count} images into final structure')
print(f'Final images directory: {final_images_dir}')

# List the final structure
print('\\nFinal directory structure:')
for root, dirs, files in os.walk(final_images_dir):
    level = root.replace(final_images_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files as example
        print(f'{subindent}{file}')
    if len(files) > 3:
        print(f'{subindent}... and {len(files) - 3} more files')
"

echo ""
echo "Image consolidation completed!"

# Optional: Clean up PID files
for gpu_id in "${GPU_IDS[@]}"; do
    pid_file="$LOG_DIR/gpu_${gpu_id}.pid"
    if [ -f "$pid_file" ]; then
        rm "$pid_file"
    fi
done

echo "Multi-GPU processing completed successfully!"
