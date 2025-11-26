#!/usr/bin/env python

import os
import shutil

def main():
    # Define dataset paths
    original_neg_dir = "/Original_dataset/IDC_regular_ps50_idx5/negative_IDC/"
    original_pos_dir = "/Original_dataset/IDC_regular_ps50_idx5/positive_IDC/"

    train_neg_dir = "train/negative_IDC/"
    train_pos_dir = "train/positive_IDC/"
    test_neg_dir = "test/negative_IDC/"
    test_pos_dir = "test/positive_IDC/"
    val_neg_dir = "validate/negative_IDC/"
    val_pos_dir = "validate/positive_IDC/"

    # Create directories if they don't exist
    for folder in [train_neg_dir, train_pos_dir, test_neg_dir, test_pos_dir, val_neg_dir, val_pos_dir]:
        os.makedirs(folder, exist_ok=True)

    # Helper function to copy files to target folder
    def copy_files(file_list, source_dir, target_dir, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            shutil.copy(os.path.join(source_dir, file_list[i]), target_dir)

    # List original files
    nontumor_files = os.listdir(original_neg_dir)
    tumor_files = os.listdir(original_pos_dir)

    # Split negative images
    copy_files(nontumor_files, original_neg_dir, train_neg_dir, 0, 300)
    copy_files(nontumor_files, original_neg_dir, test_neg_dir, 300, 600)
    copy_files(nontumor_files, original_neg_dir, val_neg_dir, 600, len(nontumor_files))

    # Split positive images
    copy_files(tumor_files, original_pos_dir, train_pos_dir, 0, 300)
    copy_files(tumor_files, original_pos_dir, test_pos_dir, 300, 600)
    copy_files(tumor_files, original_pos_dir, val_pos_dir, 600, len(tumor_files))

    # Move some validation images back to training to increase training data
    neg_val_data = os.listdir(val_neg_dir)
    pos_val_data = os.listdir(val_pos_dir)

    for z in range(min(250, len(neg_val_data))):
        src = os.path.join(val_neg_dir, neg_val_data[z])
        dst = os.path.join(train_neg_dir, neg_val_data[z])
        if not os.path.exists(dst):
            shutil.move(src, dst)

    for y in range(min(250, len(pos_val_data))):
        src = os.path.join(val_pos_dir, pos_val_data[y])
        dst = os.path.join(train_pos_dir, pos_val_data[y])
        if not os.path.exists(dst):
            shutil.move(src, dst)

    print("Preprocessing completed! Dataset is ready for training, testing, and validation.")


# Ensure the script runs main() when executed
if __name__ == "__main__":
    main()

