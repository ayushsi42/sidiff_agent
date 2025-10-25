import os
import shutil

src_dir = "OneIG-ep2-best-images"
dst_dir = "OneIG-ep2-best-images-rewired"

for filename in os.listdir(src_dir):
    if not filename.endswith(".webp"):
        continue
    try:
        # Split on last underscore
        category, index = filename.rsplit("_", 1)
        index = index.replace(".webp", "")
    except ValueError:
        print(f"Skipping {filename}: unexpected format")
        continue

    new_dir = os.path.join(dst_dir, category, "Qwen-Image")
    os.makedirs(new_dir, exist_ok=True)
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(new_dir, f"{index}.webp")
    shutil.copy2(src_path, dst_path)
    print(f"Moved {filename} -> {dst_path}")

print("Rewiring complete.")