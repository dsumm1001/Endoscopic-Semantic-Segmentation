import os
import cv2
import numpy as np
from tqdm import tqdm

# Mapping part intensities (input) to class IDs (1–4)
PART_LABELS = {
    30: 1,  # Shaft
    100: 2,  # Wrist
    255: 3,  # Claspers
    200: 4   # Probe
}

def remap_mask(mask):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for original_val, class_id in PART_LABELS.items():
        remapped[mask == original_val] = class_id
    return remapped

def process_partseg_masks(root, ignore_dirs=None):
    # Default to an empty list if no directories to ignore
    if ignore_dirs is None:
        ignore_dirs = ["binary_composite"]

    for dataset_name in sorted(os.listdir(root)):
        dataset_path = os.path.join(root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"Processing {dataset_name}...")

        gt_dir = os.path.join(dataset_path, 'ground_truth')
        partseg_dir = os.path.join(gt_dir, 'PartsSegmentation')
        output_dir = os.path.join(gt_dir, 'part_seg_composite')
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(partseg_dir):
            print(f"PartsSegmentation folder not found in {dataset_name}, skipping.")
            continue

        for frame_file in sorted(os.listdir(partseg_dir)):
            if not frame_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            part_mask_path = os.path.join(partseg_dir, frame_file)
            mask = cv2.imread(part_mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Failed to load {part_mask_path}, skipping.")
                continue

            remapped = remap_mask(mask)
            out_path = os.path.join(output_dir, frame_file)
            cv2.imwrite(out_path, remapped)
            print(f"Saved remapped mask to {out_path}")


# Load the image in grayscale
image_path = "C:/Users/dsumm/OneDrive/Documents/UMD ENPM Robotics Files/BIOE658B (Intro to Medical Image Analysis)/Project/dataset/test/instrument_dataset_1/ground_truth/PartsSegmentation/frame225.png"  # Replace with your image path
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded properly
if gray_image is None:
    print("Failed to load image.")
else:
    # Get unique pixel intensities in the image
    unique_vals = np.unique(gray_image)
    print(f"Unique pixel intensities in the image: {unique_vals}")

#input("Hello")

if __name__ == "__main__":
    root_dir = "C:/Users/dsumm/OneDrive/Documents/UMD ENPM Robotics Files/BIOE658B (Intro to Medical Image Analysis)/Project/dataset/test/"
    process_partseg_masks(root_dir)
    print("All remapped PartSegmentation masks saved.")