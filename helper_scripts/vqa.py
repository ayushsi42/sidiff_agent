import t2v_metrics
import json
import os
import glob

def load_prompts_from_txt(txt_path):
    prompts = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                prompt = parts[0]
                index = parts[1]
                prompts[index] = prompt
    return prompts

def evaluate_dataset():
    # Initialize the VQA scoring model
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')
    
    # Load prompts from TXT file
    txt_path = '/mnt/home2/home/pranav_s/agents-image-gen/eval_benchmark/DrawBench_seed.txt'
    prompts_data = load_prompts_from_txt(txt_path)
    
    # Define the image folder path
    image_folder = "/mnt/home2/home/pranav_s/agents-image-gen/final_images_drawbench_ep2"
    # image_folder = "/mnt/home2/home/pranav_s/agents-image-gen/results_drawbench_memory_genai/DrawBench-fixseed_memory_ep1/AgentSys_vRelease/final_memory_images"
    
    # Get all image files in the folder
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.jpeg")))
    
    results = {}
    scores = []
    
    print(f"Found {len(image_files)} images to evaluate...")
    
    for image_path in image_files:
        # Extract the index from the filename (e.g., "123.png" -> "123")
        filename = os.path.basename(image_path)
        index = os.path.splitext(filename)[0].lstrip('0')
        
        if index in prompts_data:
            prompt_text = prompts_data[index]
            
            print(f"Evaluating image {index}: {filename}")
            print(f"Prompt: {prompt_text}")
            
            # Calculate VQA score
            try:
                score = clip_flant5_score(images=[image_path], texts=[prompt_text])
                score_value = float(score[0]) if isinstance(score, (list, tuple)) else float(score)
                
                results[index] = {
                    "image_file": filename,
                    "prompt": prompt_text,
                    "vqa_score": score_value
                }
                
                scores.append(score_value)
                print(f"VQA Score: {score_value:.4f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error evaluating {filename}: {str(e)}")
                results[index] = {
                    "image_file": filename,
                    "prompt": prompt_text,
                    "vqa_score": None,
                    "error": str(e)
                }
        else:
            print(f"Warning: No prompt found for image index {index}")
    
    # Calculate average score
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"\nAverage VQA Score: {average_score:.4f}")
        print(f"Total images evaluated: {len(scores)}")
    else:
        average_score = None
        print("\nNo scores calculated.")
    
    # Add summary to results
    results["_summary"] = {
        "total_images_evaluated": len(scores),
        "average_vqa_score": average_score,
        "total_images_found": len(image_files),
        "total_prompts_in_dataset": len(prompts_data)
    }
    
    # Save results to JSON file
    with open('results_drawbench.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to 'results_drawbench.json'")
    return results

if __name__ == "__main__":
    results = evaluate_dataset()