import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define label mapping based on activity types
LABEL_MAP = {
    'empty': 0,
    'ideal': 1,  # 'ideal' corresponds to 'idle'
    'walk': 2,
    'run': 3,
    'jump': 4
}

# Function to parse a single .cleaned file
def parse_cleaned_file(filepath):
    csi_list = []
    with open(filepath, 'r') as f:
        for line in f:
            # Parse CSI data from the line (based on your earlier format)
            array_match = re.search(r'"\[(.*?)\]"', line)
            if array_match:
                array_str = array_match.group(1)
                # Convert to integers, skipping invalid entries
                csi_values = []
                for x in array_str.split(','):
                    try:
                        csi_values.append(int(x))
                    except ValueError:
                        print(f"Skipping invalid value '{x}' in {filepath}")
                        continue
                # Ensure consistent length (e.g., 256 values as per your earlier data)
                if len(csi_values) == 256:  # Adjust this based on your expected CSI length
                    csi_list.append(csi_values)
                else:
                    print(f"Skipping malformed CSI sample in {filepath}, length={len(csi_values)}")
    # Convert to NumPy array
    if csi_list:
        return np.array(csi_list)
    else:
        return None

# Function to segment data into fixed-size windows with overlap
def segment_data(data, window_size=64, overlap=0.5):
    if data is None or len(data) < window_size:
        return None
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

# Function to load and label the dataset
def load_dataset(data_dir, window_size=64, overlap=0.5):
    all_data = []
    all_labels = []
    
    # Walk through all files in the directory
    for filename in os.listdir(data_dir):
        if not filename.endswith('.cleaned'):
            continue
        filepath = os.path.join(data_dir, filename)
        
        # Extract activity type from filename
        activity = None
        for key in LABEL_MAP.keys():
            if key in filename:
                activity = key
                break
        if activity is None:
            print(f"Skipping file with unknown activity: {filename}")
            continue
        
        # Parse the CSI data
        csi_data = parse_cleaned_file(filepath)
        if csi_data is None:
            print(f"No valid data in {filename}")
            continue
        
        # Segment the data into fixed-size windows
        segments = segment_data(csi_data, window_size=window_size, overlap=overlap)
        if segments is None:
            print(f"Insufficient data for windowing in {filename}")
            continue
        
        # Assign labels
        label = LABEL_MAP[activity]
        labels = np.full(segments.shape[0], label)
        
        all_data.append(segments)
        all_labels.append(labels)
    
    # Concatenate all data and labels
    if all_data:
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels
    else:
        return None, None

# Function to split dataset into train/val/test sets
def create_data_splits(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if data is None or labels is None:
        return None, None, None, None, None, None
    
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_ratio, stratify=labels, random_state=42
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Main function to execute the data loading pipeline
def main():
    data_dir = r"C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned"
    window_size = 64
    overlap = 0.5
    
    # Load and label the dataset
    print("Loading and labeling dataset...")
    data, labels = load_dataset(data_dir, window_size=window_size, overlap=overlap)
    
    if data is None:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {data.shape[0]} samples with shape {data.shape[1:]}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split into train/val/test
    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(data, labels)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Optionally save the splits to disk (e.g., as NumPy arrays)
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    print("Dataset splits saved to disk.")

if __name__ == "__main__":
    main()