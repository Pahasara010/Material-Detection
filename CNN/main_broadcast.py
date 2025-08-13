import asyncio
import argparse
import logging
import numpy as np
import torch
import websockets

from cnn_model import CSICNN  # Make sure this matches your actual CNN model class
from io import StringIO
from datetime import datetime
import json

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_cnn")

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--load', required=True, help='Path to saved CNN model')
parser.add_argument('--host', default='localhost', help='WebSocket host')
parser.add_argument('--port', type=int, default=9999, help='WebSocket port')
args = parser.parse_args()

# Load model
model = CSICNN()
model.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
model.eval()
logger.info("[✓] CNN WebSocket HAR running on %s:%d", args.host, args.port)

# Human-readable activity labels
LABELS = ['empty', 'idle', 'jump', 'run', 'walk']

# CSI parsing function
def parse_csi_message(msg):
    try:
        if '"[' not in msg:
            return None
        csi_str = msg.split('[', 1)[1].split(']')[0]
        csi_raw = np.fromstring(csi_str, sep=',', dtype=int)
        if len(csi_raw) % 2 != 0:
            return None
        csi_complex = np.array([complex(csi_raw[i], csi_raw[i + 1]) for i in range(0, len(csi_raw), 2)])
        if len(csi_complex) != 123 * 256:
            return None
        csi_tensor = torch.tensor(np.abs(csi_complex).reshape(123, 256), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return csi_tensor
    except Exception as e:
        logger.error("Sample error: %s", e)
        return None

# WebSocket handler
async def handle_connection(websocket):
    logger.info("Client connected.")
    await websocket.send(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "activity": "Waiting for CSI data..."
    }))

    async for message in websocket:
        sample = parse_csi_message(message)
        if sample is None:
            continue
        with torch.no_grad():
            logits = model(sample)
            pred = torch.argmax(logits, dim=1).item()
            activity = LABELS[pred]
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "activity": activity
            }
            logger.info(f"Predicted activity: {activity}")
            await websocket.send(json.dumps(result))

# Start WebSocket server
async def main():
    async with websockets.serve(handle_connection, args.host, args.port):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    asyncio.run(main())
