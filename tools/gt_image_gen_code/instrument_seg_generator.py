import os
import cv2
import numpy as np
from tqdm import tqdm

# Mapping part intensities (input) to class IDs (1–4)
PART_LABELS = {
    10: 1,  # Shaft
    20: 2,  # Wrist
    30: 3,  # Claspers
    40: 4   # Probe
}

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

def remap_mask(mask):
    # Create a new empty mask with same shape
    remapped = np.zeros_like(mask, dtype=np.uint8)

    for original_val, class_id in PART_LABELS.items():
        remapped[mask == original_val] = class_id

    return remapped

def process_dataset(root, ignore_dirs=None):
    # Default to an empty list if no directories to ignore
    if ignore_dirs is None:
        ignore_dirs = ["binary_composite, part_seg_composite"]

    # Iterate thru availanle instrument datasets (1-8)
    for dataset_name in sorted(os.listdir(root)):

        # Makes new subdir path for current instrument dataset (1-8)
        dataset_path = os.path.join(root, dataset_name)

        # Check to see if dir exists
        if not os.path.isdir(dataset_path):
            continue

        # Processing current instrument dataset
        print(f"Processing {dataset_name}...")

        # Find left_frames path
        left_frames_dir = os.path.join(dataset_path, 'left_frames')

        # Find ground_truth path
        gt_dir = os.path.join(dataset_path, 'ground_truth')

        # Make folder for binary composite images based on current path
        part_seg_out = os.path.join(gt_dir, 'instrument_seg_composite')
        os.makedirs(part_seg_out, exist_ok=True)

        # Get the list of instruments in the ground truth directory (filter ignored)
        instrument_dirs = sorted([
            d for d in os.listdir(gt_dir)
            if os.path.isdir(os.path.join(gt_dir, d)) and d not in ignore_dirs
        ])

        # Iterate through all the frames
        for i, frame_file in enumerate(sorted(os.listdir(left_frames_dir))):
            if not frame_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Initialize a blank composite mask (all zeros initially)
            composite_mask = np.zeros((1080,1920), dtype=np.uint8)

            for instrument_name in instrument_dirs:
                # If the instrument directory is in the ignore list, skip it
                if instrument_name in ignore_dirs:
                    print(f"Skipping {instrument_name}...")
                    continue

                # Match folder name to known instrument key
                matched_key = None
                for key in instrument_map:
                    if key in instrument_name:
                        matched_key = key
                        break

                if matched_key is None:
                    print(f"Warning: {instrument_name} not in instrument_map. Skipping.")
                    continue

                class_id = instrument_map[matched_key]

                # Get the path to the current instrument mask
                instrument_path = os.path.join(gt_dir, instrument_name)
                instrument_mask_path = os.path.join(instrument_path, frame_file)

                if os.path.exists(instrument_mask_path):
                    # Read the current instrument mask (grayscale)
                    instrument_mask = cv2.imread(instrument_mask_path, cv2.IMREAD_GRAYSCALE)

                    #cv2.imshow('Instrument mask', instrument_mask)
                    #cv2.waitKey(0)

                    # Combine the mask into the composite mask (add values)
                    # Apply instrument class ID to non-zero regions
                    class_id = instrument_map[matched_key]
                    binary_mask = (instrument_mask > 0).astype(np.uint8)
                    labeled_mask = binary_mask * class_id

                    # Merge into composite mask
                    composite_mask = np.maximum(composite_mask, labeled_mask)

            # If valid composite mask, save it
            if composite_mask is not None:
                cv2.imwrite(os.path.join(part_seg_out, frame_file), composite_mask)
                print(f"Saved semantic composite mask for {frame_file}")

if __name__ == "__main__":
    root_dir = "C:/Users/dsumm/OneDrive/Documents/UMD ENPM Robotics Files/BIOE658B (Intro to Medical Image Analysis)/Project/dataset/train"
    process_dataset(root_dir)
    print("All composite masks generated.")