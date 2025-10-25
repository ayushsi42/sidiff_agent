import json

input_path = "/mnt/localssd/shivank/agents/agents-image-gen/eval_benchmark/DrawBench_seed.txt"
output_path = "/mnt/localssd/shivank/agents/agents-image-gen/eval_benchmark/DrawBench_seed.json"

result = {}
with open(input_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) < 3:
            print("Skipping")
            continue  # skip malformed lines
        prompt, random_seed, _ = parts
        id_str = f"{idx:05d}"
        result[id_str] = {
            "id": id_str,
            "prompt": prompt,
            "random_seed": int(random_seed)
        }

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"Converted {input_path} to {output_path}")