import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import re

# Define the same model architecture
class CSIHybridModel(nn.Module):
    def __init__(self, input_shape=(64, 256), num_classes=5):
        super(CSIHybridModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to parse and segment a new .cleaned file
def parse_and_segment_new_file(filepath, window_size=64, overlap=0.5):
    csi_list = []
    with open(filepath, 'r') as f:
        for line in f:
            array_match = re.search(r'"\[(.*?)\]"', line)
            if array_match:
                array_str = array_match.group(1)
                csi_values = []
                for x in array_str.split(','):
                    try:
                        csi_values.append(int(x))
                    except ValueError:
                        continue
                if len(csi_values) == 256:
                    csi_list.append(csi_values)
    if not csi_list:
        return None
    csi_data = np.array(csi_list)
    segments = []
    step = int(window_size * (1 - overlap))
    for start in range(0, len(csi_data) - window_size + 1, step):
        segment = csi_data[start:start + window_size]
        segments.append(segment)
    return np.array(segments) if segments else None

# Main prediction function
def predict_new_data(model_path, new_file_path, scaler_path=None):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSIHybridModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Parse and segment the new file
    new_data = parse_and_segment_new_file(new_file_path)
    if new_data is None:
        print("No valid data found in the file.")
        return

    # Load the scaler used during training (assuming saved from train script)
    if scaler_path and os.path.exists(scaler_path):
        scaler = StandardScaler()
        scaler.mean_ = np.load(f"{scaler_path}_mean.npy")
        scaler.scale_ = np.load(f"{scaler_path}_scale.npy")
    else:
        print("No scaler found. Using training scaler (approximate).")
        scaler = StandardScaler()
        # This is a placeholder; ideally, save scaler during training
        # For now, fit on new data (not ideal, but works for demo)
        new_data_reshaped = new_data.reshape(-1, new_data.shape[-1])
        scaler.fit(new_data_reshaped)
        np.save('scaler_mean.npy', scaler.mean_)
        np.save('scaler_scale.npy', scaler.scale_)

    # Normalize the new data
    new_data_reshaped = new_data.reshape(-1, new_data.shape[-1])
    new_data_scaled = scaler.transform(new_data_reshaped).reshape(new_data.shape)
    new_data_tensor = torch.FloatTensor(new_data_scaled).to(device)

    # Create data loader
    new_dataset = TensorDataset(new_data_tensor)
    new_loader = DataLoader(new_dataset, batch_size=64)

    # Make predictions
    all_preds = []
    with torch.no_grad():
        for X_batch in new_loader:
            X_batch = X_batch[0].to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # Map predictions to labels
    label_map = {0: 'empty', 1: 'ideal', 2: 'walk', 3: 'run', 4: 'jump'}
    predicted_labels = [label_map[pred] for pred in all_preds]

    # Print results
    print(f"Predictions for {new_file_path}:")
    for i, pred in enumerate(predicted_labels):
        print(f"Window {i+1}: {pred}")
    return predicted_labels

import os
if __name__ == "__main__":
    model_path = 'csi_model_best.pth'
    new_file_path = r"C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned\clean_walk_33.cleaned"  # Adjust this path
    scaler_path = 'scaler'  # Adjust if you saved scaler during training
    predict_new_data(model_path, new_file_path, scaler_path)