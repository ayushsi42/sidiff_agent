import os
import csv

csv_path = ""
img_dir = ""
output_csv = ""

# 1. Collect expected image filenames from CSV
expected = []
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        category = row['category']
        idx = row['id']
        fname = f"{category}_{idx}.webp"
        expected.append((fname, row))

# 2. List all images in the folder
present = set(os.listdir(img_dir))

# 3. Find missing images
missing_rows = [row for fname, row in expected if fname not in present]

# 4. Write missing entries to a new CSV
if missing_rows:
    with open(output_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=missing_rows[0].keys())
        writer.writeheader()
        writer.writerows(missing_rows)
    print(f"Missing entries written to {output_csv}")
else:
    print("No missing images found.")