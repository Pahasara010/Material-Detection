#!/usr/bin/env python3

import asyncio
import argparse
import logging
import numpy as np
import torch
import websockets
import json

from cnn_model import CSICNN

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_cnn")

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--load', required=True, help='Path to saved CNN model (.pth)')
parser.add_argument('--host', default='localhost', help='Host for WebSocket server')
parser.add_argument('--port', type=int, default=9999, help='WebSocket server port')
args = parser.parse_args()

# --- Load CNN Model ---
model = CSICNN()
model.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
model.eval()
logger.info("[✓] CNN model loaded and ready for real-time HAR!")

# --- Activity Labels ---
LABELS = ['empty', 'idle', 'jump', 'run', 'walk']

# --- CSI Parsing Function ---
def parse_csi_message(msg):
    try:
        data = json.loads(msg)
        csi_raw = np.array(data.get("csi", []), dtype=int)

        if len(csi_raw) % 2 != 0:
            logger.warning("CSI data length is odd; must be even for complex pairs.")
            return None

        csi_complex = np.array([complex(csi_raw[i], csi_raw[i + 1]) for i in range(0, len(csi_raw), 2)])

        if len(csi_complex) != 123 * 256:
            logger.warning("CSI data size is %d, expected %d", len(csi_complex), 123 * 256)
            return None

        csi_tensor = torch.tensor(np.abs(csi_complex).reshape(123, 256), dtype=torch.float32)
        csi_tensor = csi_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 123, 256)
        return csi_tensor

    except Exception as e:
        logger.error("Failed to parse CSI message: %s", e)
        return None

# --- WebSocket Handler ---
async def handle_connection(websocket):
    logger.info("Client connected. Awaiting CSI data for Real-Time HAR.")
    await websocket.send("🧍 Activity: Waiting for CSI data...\n🔌 Connection: Connected")

    async for message in websocket:
        logger.debug(f"Raw incoming message: {message}")

        sample = parse_csi_message(message)
        if sample is None:
            logger.warning("Invalid CSI sample. Skipped.")
            continue

        with torch.no_grad():
            logits = model(sample)
            pred = torch.argmax(logits, dim=1).item()
            activity = LABELS[pred]

            # ✅ Print clearly to terminal
            print(f"[🧠 Prediction] Activity: {activity}")

            # Logging and sending
            logger.info(f"Predicted activity: {activity}")
            await websocket.send(f"Activity: {activity}")

# --- Server Entrypoint ---
async def main():
    async with websockets.serve(handle_connection, args.host, args.port):
        logger.info(f"🌐 WebSocket Server started at ws://{args.host}:{args.port}")
        await asyncio.Future()  # Keep server alive

# --- Main Call ---
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("🛑 Server shutdown by user.")
