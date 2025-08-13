import asyncio
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import websockets
import re

# Define the model architecture
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

# Function to parse and segment CSI data
def parse_and_segment_csi(csi_string, window_size=64, overlap=0.5):
    csi_values = []
    array_match = re.search(r'\[(.*?)\]', csi_string)
    if array_match:
        array_str = array_match.group(1)
        for x in array_str.split(','):
            try:
                csi_values.append(int(x))
            except ValueError:
                continue
    if len(csi_values) != 256:
        return None
    csi_data = np.array([csi_values])  # Single sample for now
    segments = []
    step = int(window_size * (1 - overlap))
    for start in range(0, len(csi_data) - window_size + 1, step):
        segment = csi_data[start:start + window_size]
        segments.append(segment)
    return np.array(segments) if segments else None

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSIHybridModel().to(device)
model.load_state_dict(torch.load(r'C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned\csi_model_best.pth', weights_only=True))
model.eval()

# Load or initialize scaler (approximate for now)
scaler = StandardScaler()
# Ideally, load saved scaler mean and scale from training
# For now, this is a placeholder; adjust if you saved scaler
scaler.mean_ = np.load(r'C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned\scaler_mean.npy') if os.path.exists(r'C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned\scaler_mean.npy') else np.zeros(256)
scaler.scale_ = np.load(r'C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned\scaler_scale.npy') if os.path.exists(r'C:\Users\Admin\Desktop\csi_copy\dataset\mat\cleaned\scaler_scale.npy') else np.ones(256)

# WebSocket handler
async def handler(websocket, path):
    while True:
        try:
            # Receive CSI data as JSON
            data = await websocket.recv()
            csi_input = json.loads(data)
            csi_string = csi_input.get("csi", "")
            
            # Parse and segment the CSI data
            segments = parse_and_segment_csi(csi_string)
            if segments is None or len(segments) == 0:
                await websocket.send(json.dumps({"error": "Invalid or insufficient CSI data"}))
                continue

            # Normalize the data
            segments_reshaped = segments.reshape(-1, segments.shape[-1])
            segments_scaled = scaler.transform(segments_reshaped).reshape(segments.shape)
            segments_tensor = torch.FloatTensor(segments_scaled).to(device)

            # Create data loader
            dataset = TensorDataset(segments_tensor)
            loader = DataLoader(dataset, batch_size=64)

            # Make predictions
            all_preds = []
            with torch.no_grad():
                for X_batch in loader:
                    X_batch = X_batch[0].to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())

            # Map predictions to labels
            label_map = {0: 'empty', 1: 'ideal', 2: 'walk', 3: 'run', 4: 'jump'}
            predicted_labels = [label_map[pred] for pred in all_preds]

            # Send predictions back
            response = {"predictions": predicted_labels}
            await websocket.send(json.dumps(response))

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

# Start the WebSocket server
start_server = websockets.serve(handler, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()