import os
import cv2
import numpy as np
from tqdm import tqdm

# Converting instrument labels to simple integers for conversion to megamask
instrument_map = {
    "Bipolar_Forceps": 1,
    "Prograsp_Forceps": 2,
    "Large_Needle_Driver": 3,
    "Vessel_Sealer": 4,
    "Grasping_Retractor": 5,
    "Monopolar_Curved_Scissors": 6,
    "Other": 7
}

import os 
import cv2
import numpy as np
from tqdm import tqdm

def process_typeseg_masks(root, ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = ["binary_composite"]

    total_unique_vals = set()

    for dataset_name in sorted(os.listdir(root)):
        dataset_path = os.path.join(root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\nProcessing {dataset_name}...")

        gt_dir = os.path.join(dataset_path, 'ground_truth')
        typeseg_dir = os.path.join(gt_dir, 'TypeSegmentation')
        #output_dir = os.path.join(gt_dir, 'instrument_seg_composite')
        #os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(typeseg_dir):
            print(f"TypeSegmentation folder not found in {dataset_name}, skipping.")
            continue

        for frame_file in sorted(os.listdir(typeseg_dir)):
            if not frame_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            type_mask_path = os.path.join(typeseg_dir, frame_file)
            mask = cv2.imread(type_mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Failed to load {type_mask_path}, skipping.")
                continue

            # Track and print unique pixel values in the current image
            unique_vals = np.unique(mask)
            total_unique_vals.update(unique_vals)
            print(f"{frame_file}: unique pixel values = {unique_vals}")

            # Save the mask to the output directory
            #out_path = os.path.join(output_dir, frame_file)
            #cv2.imwrite(out_path, mask)

    # Print summary of all unique pixel values across the dataset
    print(f"\nTotal unique pixel intensities found in all images: {sorted(total_unique_vals)}")

if __name__ == "__main__":
    root_dir = "C:/Users/dsumm/OneDrive/Documents/UMD ENPM Robotics Files/BIOE658B (Intro to Medical Image Analysis)/Project/dataset/test/"
    process_typeseg_masks(root_dir)
    print("All instrument segmentation masks processed.")