import os
import torch
import json
import argparse
import time
from PIL import Image
from tqdm import tqdm
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel
)
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
import numpy as np

def load_qwen_image_model(use_quantization=True):
    """Load Qwen-Image model with optional quantization"""
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        if use_quantization:
            print("Loading Qwen Image model with quantization...")
            model_id = "Qwen/Qwen-Image"
            torch_dtype = torch.bfloat16
            
            quantization_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            )

            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            transformer = transformer.to("cpu")

            quantization_config = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                subfolder="text_encoder",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            text_encoder = text_encoder.to("cpu")

            qwen_image_pipe = QwenImagePipeline.from_pretrained(
                model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
            )
            qwen_image_pipe.to(f"cuda:{torch.cuda.current_device()}")
            
        else:
            print("Loading Qwen Image model without quantization...")
            qwen_image_pipe = QwenImagePipeline.from_pretrained(
                "Qwen/Qwen-Image", 
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{torch.cuda.current_device()}"
            )

        print("GPU Name:", torch.cuda.get_device_name(0))
        print("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 2), "GB")
        print("Memory Cached:", round(torch.cuda.memory_reserved(0)/1024**3, 2), "GB")
        print("Total Memory:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), "GB")
        print("Model loaded successfully")
        
        return qwen_image_pipe
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU out of memory. Trying to free up memory...")
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory. Try reducing batch size or image dimensions.")
        raise

def normalize_image(image):
    """Normalize image data to ensure valid pixel values before saving."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if isinstance(image, np.ndarray):
        image = np.nan_to_num(image, nan=0.5)
        image = np.clip(image, 0, 1)
        if image.dtype != np.uint8:
            image = (image * 255).round().astype(np.uint8)
        image = Image.fromarray(image)
    
    return image

def generate_image(pipe, prompt, seed, image_index, save_dir):
    """Generate image using Qwen-Image pipeline"""
    try:
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set up generator with seed
        generator = torch.Generator(f"cuda:{torch.cuda.current_device()}").manual_seed(seed)
        negative_prompt = "Gorgeous , static, blucolors, overexposedrred details, subtitles, style, artwork, painting, image, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, three legs, crowded background, walking backwards"
        
        with torch.inference_mode():
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                height=1024,
                width=1024,
                true_cfg_scale=4.0,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator
            ).images[0]
            
        # Normalize image before saving
        image = normalize_image(image)

        # Save image
        output_path = os.path.join(save_dir, f"{image_index}_Qwen-Image.png")
        image.save(output_path)
        
        print(f"Successfully generated image at: {output_path}")
        return output_path
            
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

def load_benchmark(benchmark_path):
    """Load benchmark prompts and seeds from file"""
    prompts = []
    seeds = []
    image_indices = []
    
    with open(benchmark_path, 'r') as file:
        if "DrawBench" in benchmark_path:
            if "seed" in benchmark_path:
                lines = [line.strip().split('\t') for line in file]
                prompts = [line[0] for line in lines]
                seeds = [int(line[1]) for line in lines]
            else:
                prompts = [line.strip().split('\t')[0] for line in file]
                seeds = [torch.randint(0, 1000000, (1,)).item() for _ in prompts]
            image_indices = [f"{i:03d}" for i in range(len(prompts))]
            
        elif "GenAIBenchmark" in benchmark_path:
            data = json.load(file)
            for key in data.keys():
                prompts.append(data[key]['prompt'])
                seeds.append(data[key].get('random_seed', torch.randint(0, 1000000, (1,)).item()))
                image_indices.append(data[key]['id'])
                
        else:
            # Default format: prompt\tseed
            lines = [line.strip().split('\t') for line in file]
            prompts = [line[0] for line in lines]
            seeds = [int(line[1]) for line in lines]
            image_indices = [f"{i:03d}" for i in range(len(prompts))]
    
    return prompts, seeds, image_indices

def main():
    parser = argparse.ArgumentParser(description="Direct Qwen-Image Inference")
    parser.add_argument('--benchmark_path', default='eval_benchmark/cool_sample.txt', 
                       type=str, help='Path to benchmark file')
    parser.add_argument('--output_dir', default='results/qwen_direct', 
                       type=str, help='Output directory for generated images')
    parser.add_argument('--use_quantization', action='store_true', default=True,
                       help='Use quantization for memory efficiency')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start index for resuming generation')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End index for data splitting')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use for this process')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires a GPU to run.")
    
    # Initialize CUDA
    torch.cuda.init()
    torch.cuda.set_device(args.gpu_id)
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(args.gpu_id)
    gpu_memory = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3
    print(f"\nUsing GPU {args.gpu_id}: {gpu_name}")
    print(f"Available GPU memory: {gpu_memory:.2f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark
    print(f"Loading benchmark from: {args.benchmark_path}")
    prompts, seeds, image_indices = load_benchmark(args.benchmark_path)
    print(f"Loaded {len(prompts)} prompts")
    
    # Apply data splitting
    if args.end_idx is not None:
        prompts = prompts[args.start_idx:args.end_idx]
        seeds = seeds[args.start_idx:args.end_idx]
        image_indices = image_indices[args.start_idx:args.end_idx]
        print(f"Processing slice [{args.start_idx}:{args.end_idx}] = {len(prompts)} prompts")
    else:
        prompts = prompts[args.start_idx:]
        seeds = seeds[args.start_idx:]
        image_indices = image_indices[args.start_idx:]
        print(f"Processing from index {args.start_idx} = {len(prompts)} prompts")
    
    # Load model
    print("Loading Qwen-Image model...")
    pipe = load_qwen_image_model(args.use_quantization)
    
    # Track timing
    inference_times = []
    successful_generations = 0
    failed_generations = 0
    
    # Generate images
    print(f"Starting generation...")
    for i, (prompt, seed, img_idx) in enumerate(tqdm(
        zip(prompts, seeds, image_indices),
        desc=f"GPU {args.gpu_id} generating images",
        total=len(prompts)
    )):
        start_time = time.time()
        
        result_path = generate_image(pipe, prompt, seed, img_idx, args.output_dir)
        
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        if result_path:
            successful_generations += 1
        else:
            failed_generations += 1
            
        print(f"Image {img_idx}: {inference_time:.2f}s")
        
        # Save progress periodically
        if (i + 1) % 10 == 0:
            progress = {
                "gpu_id": args.gpu_id,
                "completed": i + 1,
                "total": len(prompts),
                "successful": successful_generations,
                "failed": failed_generations,
                "avg_time": sum(inference_times) / len(inference_times)
            }
            progress_file = os.path.join(args.output_dir, f"progress_gpu_{args.gpu_id}.json")
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)
    
    # Final statistics
    total_time = sum(inference_times)
    avg_time = total_time / len(inference_times) if inference_times else 0
    
    print("\n" + "="*50)
    print(f"GENERATION COMPLETE ON GPU {args.gpu_id}")
    print("="*50)
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Average time per image: {avg_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    
    # Save final statistics
    final_stats = {
        "gpu_id": args.gpu_id,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "total_prompts": len(prompts),
        "successful_generations": successful_generations,
        "failed_generations": failed_generations,
        "avg_time_per_image": avg_time,
        "total_time": total_time,
        "inference_times": inference_times
    }
    
    stats_file = os.path.join(args.output_dir, f"final_stats_gpu_{args.gpu_id}.json")
    with open(stats_file, "w") as f:
        json.dump(final_stats, f, indent=2)

if __name__ == "__main__":
    main()