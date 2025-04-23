import os
import xml.etree.ElementTree as ET
from PIL import Image

# Set your paths here
annotations_dir = 'DATASET_v1/Train/Annotations'  # Path to XML annotations
images_dir = 'DATASET_v1/Train/JPEGImages'        # Path to source images
output_dir = 'DATASET_v1/train'            # Path to save cropped pedestrians

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each XML annotation file
for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith('.xml'):
        continue

    # Parse XML file
    xml_path = os.path.join(annotations_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get corresponding image file
    image_filename = os.path.splitext(xml_file)[0] + '.png'
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Image {image_filename} not found, skipping...")
        continue

    # Open the image
    with Image.open(image_path) as img:
        # Extract all pedestrians
        for i, obj in enumerate(root.findall('object')):
            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Crop and save pedestrian
            pedestrian_img = img.crop((xmin, ymin, xmax, ymax))
            output_filename = f"{os.path.splitext(image_filename)[0]}_{i}.png"
            output_path = os.path.join(output_dir, output_filename)
            pedestrian_img.save(output_path)

print("Pedestrian extraction completed!")
