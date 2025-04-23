import os
import cv2
import xml.etree.ElementTree as ET

def process_dataset(base_input_dir, base_output_dir):
    # Process both Train and Test splits
    for split in ['Train', 'Test']:
        # Set target dimensions based on split
        if split == 'Train':
            target_width, target_height = 96, 160
        else:
            target_width, target_height = 70, 137

        # Create output directories
        output_dir = os.path.join(base_output_dir, split)
        os.makedirs(output_dir, exist_ok=True)

        # Paths to input directories
        annotations_dir = os.path.join(base_input_dir, split, 'Annotations')
        images_dir = os.path.join(base_input_dir, split, 'JPEGImages')

        # Process each XML annotation file
        for xml_file in os.listdir(annotations_dir):
            if not xml_file.endswith('.xml'):
                continue

            # Parse XML file
            xml_path = os.path.join(annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get corresponding image path
            img_name = os.path.splitext(xml_file)[0] + '.png'
            img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Image {img_path} not found, skipping")
                continue

            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Extract all pedestrian bounding boxes
            for i, obj in enumerate(root.findall('object')):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Crop and resize the pedestrian region
                pedestrian = img[ymin:ymax, xmin:xmax]
                resized = cv2.resize(pedestrian, (target_width, target_height))

                # Save the processed image
                output_name = f"{os.path.splitext(xml_file)[0]}_{i}.png"
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, resized)

if __name__ == '__main__':
    input_dataset = 'DATASET_v1'
    output_dataset = 'DATASET_v1_Cropped'
    process_dataset(input_dataset, output_dataset)