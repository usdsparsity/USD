
import os
import glob

# Define the base directory
train_dir = "../tiny-imagenet-200/test_img"
output_file = "../tiny-imagenet-200/val.txt"

# Open the output file for writing
with open(output_file, 'w') as f:
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Process each class directory
    for class_dir in class_dirs:
        class_path = os.path.join(train_dir, class_dir)
        
        # Get all image files in the class directory
        image_files = glob.glob(os.path.join(class_path, "*.*"))
        
        # Write each image path in the required format
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            # Format: <class>/<image_name>
            formatted_path = f"{class_dir}/{img_name} {class_dir}"
            f.write(f"{formatted_path}\n")

print(f"Created {output_file} with image paths in the format <class>/<image_name>")
exit(0)

import os
import shutil

import os
import shutil
import glob

# Define the base directory
train_dir = "../tiny-imagenet-200/train"

# Get all class directories
class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

# Process each class directory
for class_dir in class_dirs:
    class_path = os.path.join(train_dir, class_dir)
    images_dir = os.path.join(class_path, "images")
    
    # Check if the images subdirectory exists
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        # Get all image files
        image_files = glob.glob(os.path.join(images_dir, "*.*"))
        
        # Move each image to the class root directory
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(class_path, img_name)
            
            # Move the file
            shutil.move(img_path, dst_path)
            print(f"Moved {img_path} to {dst_path}")
        
        # Remove the now empty images directory
        if not os.listdir(images_dir):
            os.rmdir(images_dir)
            print(f"Removed empty directory: {images_dir}")

print("All images have been moved to their class root directories.")

# Define the paths
annotations_file = "../tiny-imagenet-200/val/val_annotations.txt"
images_dir = "../tiny-imagenet-200/val/images"
output_base_dir = "../tiny-imagenet-200/val/organized"  # Base directory for organized images

# Create the output base directory if it doesn't exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Read the annotations file
with open(annotations_file, 'r') as f:
    annotations = f.readlines()

# Process each annotation
for line in annotations:
    parts = line.strip().split()
    if len(parts) >= 2:  # Ensure we have at least image_name and class_id
        image_name = parts[0]
        class_id = parts[1]
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_base_dir, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # Define source and destination paths
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(class_dir, image_name)
        
        # Copy the image to its class directory
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {image_name} to {class_dir}")
        else:
            print(f"Warning: Could not find {src_path}")

print("Organization complete!")