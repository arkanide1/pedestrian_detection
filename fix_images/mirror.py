import os
from PIL import Image

# Set paths
input_dir = 'DATASET_v1/Test'
output_dir = 'DATASET_v1/TestM'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    input_path = os.path.join(input_dir, img_name)

    # Open the image
    with Image.open(input_path) as img:
        # Create mirrored version
        mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Save with modified filename
        base_name = os.path.splitext(img_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_mirror.png")
        mirrored_img.save(output_path)

print("Mirror images created successfully!")
