import os
from PIL import Image

# Configuration
input_dir = "DATASET_v1/Test"  # Directory with original images
output_dir = "DATASET_v1/TestR"  # Directory to save renamed images
start_number = 1  # Starting number for filenames

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image files sorted alphabetically
image_files = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Rename and save files
for idx, filename in enumerate(image_files, start=start_number):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{idx}.png")

    with Image.open(input_path) as img:
        # Convert to RGB if needed (for JPEG compatibility)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(output_path)

print(f"Renamed {len(image_files)} files to {output_dir}")
