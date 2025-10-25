import os
import json
import shutil
import re

base_dir = ''
final_dir = ''
os.makedirs(final_dir, exist_ok=True)

def resolve_path(img_path):
    # If absolute and exists, return as is
    if os.path.isabs(img_path) and os.path.isfile(img_path):
        return img_path
    # If starts with 'results/OneIG-Bench/vRelease/', strip and join with base_dir
    prefix = ''
    if img_path.startswith(prefix):
        rel_path = img_path[len(prefix):]
        candidate = os.path.join(base_dir, rel_path)
        if os.path.isfile(candidate):
            return candidate
    # Try relative to base_dir
    candidate = os.path.join(base_dir, img_path)
    if os.path.isfile(candidate):
        return candidate
    # Try as is (relative to CWD)
    if os.path.isfile(img_path):
        return img_path
    return None

gpu_folders = sorted([f for f in os.listdir(base_dir) if f.startswith('gpu_')], key=lambda x: int(x.split('_')[1]))

for gpu_folder in gpu_folders:
    gpu_path = os.path.join(base_dir, gpu_folder)
    config_files = sorted([f for f in os.listdir(gpu_path) if f.endswith('_config.json')])
    for config_file in config_files:
        match = re.match(r"(\d+)_config\.json", config_file)
        if not match:
            print(f"WARNING: Could not parse index from {config_file}")
            continue
        index = match.group(1)
        config_path = os.path.join(gpu_path, config_file)
        with open(config_path, 'r') as f:
            config = json.load(f)
        regen_configs = config.get('regeneration_configs', {})
        best_score = float('-inf')
        best_img_path = None
        for block in regen_configs.values():
            score = block.get('evaluation_score', float('-inf'))
            img_path = block.get('gen_image_path')
            if img_path and score > best_score:
                best_score = score
                best_img_path = img_path
        if best_img_path:
            src_img = resolve_path(best_img_path)
            if src_img is None:
                print(f"WARNING: Image not found for {config_file}: {best_img_path}")
                continue
            # Use original extension
            ext = os.path.splitext(src_img)[1]
            dst_img = os.path.join(final_dir, f"{index}{ext}")
            shutil.copyfile(src_img, dst_img)