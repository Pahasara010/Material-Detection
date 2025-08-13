#!/usr/bin/env python3

import sys
import pathlib
import warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from io import StringIO
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import websockets
import pyrebase
import smtplib
from email.mime.text import MIMEText
from scipy.signal import butter, sosfilt

from MID.transformers import CSIAmplitudeMinMaxScaler
from MID.constants import CSI_COL_NAMES, NULL_SUBCARRIERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class CSIMaterialID:
    def __init__(self, config):
        self.config = config
        self.fifo_path = config["fifo_path"]
        self.expected_samples = config["expected_samples"]
        self.input_shape = tuple(config["input_shape"])
        self.class_labels = config["class_labels"]
        self.num_classes = len(self.class_labels)

        self.scaler = CSIAmplitudeMinMaxScaler()
        self.model = CNNLSTMModel(self.num_classes, self.input_shape)
        self.model.load_state_dict(torch.load(config["model_path"] / "cnn_lstm_model.pth"))
        self.model.eval()

        self.output_payload = {
            "timestamp": None,
            "hypothesis": None,
            "classnames": self.class_labels,
            "rx_sources": None
        }

        self.fifo = None
        self.ensure_fifo_open()

        self.window_predictions = []
        self.last_output_time = datetime.utcnow()
        self.window_duration = timedelta(seconds=3)

        self.firebase = pyrebase.initialize_app(config["firebase_config"])
        self.db = self.firebase.database()
        self.db.child("system").set({"armed": True})
        logger.info("[Firebase] Initialized and set system/armed: true")

        self.email_address = config["email_address"]
        self.email_password = config["email_password"]

    def ensure_fifo_open(self):
        while self.fifo is None:
            try:
                self.fifo = open(self.fifo_path, 'r')
                logger.info(f"[FIFO] Connected to {self.fifo_path}")
            except FileNotFoundError:
                logger.warning(f"[FIFO] {self.fifo_path} not found. Retrying in 1 second...")
                time.sleep(1)

    def send_alert(self, material_info):
        msg = MIMEText(f"Alert: {material_info} detected at {datetime.utcnow()}")
        msg['Subject'] = 'Material Detection Alert'
        msg['From'] = self.email_address
        msg['To'] = self.config["alert_email"]
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            logger.info(f"[Alert] Sent email for {material_info}")
        except Exception as e:
            logger.error(f"[Alert Error] {e}")

    def get_container_from_line(self) -> str:
        try:
            self.fifo.seek(0)
            line = self.fifo.readline()
            if line.startswith(("cardboard", "plastic")):
                return line.split(":")[0].strip()
            return "unknown"
        except Exception:
            return "unknown"

    def make_prediction(self) -> Optional[str]:
        try:
            csi_sample = self.get_next_sample()
            if csi_sample is None:
                logger.warning("[Prediction] No sample retrieved.")
                return None

            target_h, target_w = self.input_shape
            h, w = csi_sample.shape

            if h < target_h:
                csi_sample = np.pad(csi_sample, ((0, target_h - h), (0, 0)))
            elif h > target_h:
                csi_sample = csi_sample[:target_h, :]

            if w < target_w:
                csi_sample = np.pad(csi_sample, ((0, 0), (0, target_w - w)))
            elif w > target_w:
                csi_sample = csi_sample[:, :target_w]

            if not np.isfinite(csi_sample).all():
                logger.warning("[Guard] Non-finite values in CSI sample. Skipping.")
                return None

            csi_sample = np.clip(csi_sample, -1e3, 1e3)
            csi_sample = csi_sample.reshape(1, *csi_sample.shape)
            csi_sample = self.scaler.fit_transform(csi_sample)

            input_tensor = torch.tensor(csi_sample, dtype=torch.float32)
            with torch.no_grad():
                pred = torch.argmax(self.model(input_tensor), dim=1).item()
            container = self.get_container_from_line()
            self.window_predictions.append((pred, container))

            now = datetime.utcnow()
            if now - self.last_output_time >= self.window_duration:
                if self.window_predictions:
                    pred_counts = {}
                    for pred, container in self.window_predictions:
                        key = f"{container}_{self.class_labels[pred]}"
                        pred_counts[key] = pred_counts.get(key, 0) + 1
                    if pred_counts:
                        final_pred_key = max(pred_counts, key=pred_counts.get)
                        final_container, final_material = final_pred_key.split("_", 1)
                        material_info = final_pred_key

                        contributing_containers = [container for pred, container in self.window_predictions
                                              if f"{container}_{self.class_labels[pred]}" == final_pred_key]

                        self.output_payload["timestamp"] = now.isoformat()
                        self.output_payload["hypothesis"] = material_info
                        self.output_payload["rx_sources"] = contributing_containers

                        self.db.child("material_logs").child(final_container).push({
                            "timestamp": now.isoformat(),
                            "material": final_material,
                            "rx_sources": contributing_containers
                        })
                        logger.info(f"[Firebase] Pushed to material_logs/{final_container}: {final_material}")

                        armed = self.db.child("system").get().val().get("armed", True)
                        if armed and final_material in ['iron', 'water']:
                            self.send_alert(material_info)

                        self.last_output_time = now
                        self.window_predictions = []

                        logger.info(f"[AGGREGATED] Prediction: {material_info} (Containers: {contributing_containers})")
                        return json.dumps(self.output_payload)

            return None

        except Exceptionಸ
System: **Error: Your request exceeded the maximum time limit of 120 seconds.**