import sys
import serial
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLineEdit, QLabel, QFileDialog, QTextEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import time
import os
import ast
import csv
import io
from sklearn.utils.class_weight import compute_class_weight

print("Starting CSI Gesture Identification script...")

class CollectionThread(QThread):
    data_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(int, float)
    countdown_signal = pyqtSignal(int)

    def __init__(self, port, activity, save_path, log_path):
        super().__init__()
        self.port = port
        self.activity = activity
        self.save_path = save_path
        self.log_path = log_path
        self.running = False
        self.segment_duration_map = {'sit': 30, 'empty': 120, 'idle': 30, 'run': 60, 'walk': 60}
        self.segment_duration = self.segment_duration_map.get(activity, 30)
        self.break_duration = 10
        self.target_packets = 15000

    def run(self):
        self.running = True
        try:
            ser = serial.Serial(self.port, 921600, timeout=5)
            packets = []
            start_time = time.time()
            packet_count = 0
            segment_count = 0
            max_segments = 6

            with open(self.save_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write('')
            with open(self.log_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write('')

            if self.activity == 'empty':
                self.data_signal.emit("Starting 15-second delay for 'empty' activity...")
                delay_start = time.time()
                delay_duration = 15
                while self.running and (time.time() - delay_start) < delay_duration:
                    remaining = int(delay_duration - (time.time() - delay_start))
                    self.data_signal.emit(f"Delay countdown: {remaining} seconds")
                    time.sleep(1)
                if not self.running:
                    ser.close()
                    self.data_signal.emit("Collection stopped during initial delay")
                    return
                self.data_signal.emit("Delay complete. Starting data collection...")

            while self.running and packet_count < self.target_packets and segment_count < max_segments:
                segment_start = time.time()
                while self.running and time.time() - segment_start < self.segment_duration and packet_count < self.target_packets:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='replace').strip()
                        with open(self.log_path, 'a', encoding='utf-8', errors='replace') as f:
                            f.write(line + '\n')
                        if line.startswith('CSI_DATA'):
                            csv_reader = csv.reader(io.StringIO(line), skipinitialspace=True)
                            parts = next(csv_reader)
                            if len(parts) != 25:
                                self.data_signal.emit(f"Invalid CSI data format: {len(parts)} fields, expected 25")
                                continue
                            if not (parts[1].isdigit() and parts[2].startswith('fc:') and parts[3].lstrip('-').isdigit()):
                                self.data_signal.emit(f"Invalid CSI data fields: type={parts[1]}, mac={parts[2]}, rssi={parts[3]}")
                                continue
                            try:
                                csi_data = ast.literal_eval(parts[-1])
                                if not isinstance(csi_data, list) or len(csi_data) != 256 or not all(isinstance(x, (int, float)) for x in csi_data):
                                    self.data_signal.emit(f"Invalid CSI buffer: not a list of 256 numbers")
                                    continue
                                csi_data = np.array(csi_data[:128])
                                packet_count += 1
                                packets.append(parts[:-1] + [str(csi_data.tolist()), self.activity])
                                valid_csi = csi_data[csi_data != 0]
                                csi_std = np.std(np.abs(valid_csi)) if len(valid_csi) > 0 else 0
                                if len(valid_csi) > 10:
                                    window_size = min(20, len(valid_csi))
                                    csi_std = np.std(np.abs(valid_csi[-window_size:]))
                                self.stats_signal.emit(packet_count, csi_std)
                                packet_rate = packet_count / (time.time() - start_time) if time.time() > start_time else 0
                                self.data_signal.emit(f"Packets: {packet_count}, CSI std: {csi_std:.2f}, Rate: {packet_rate:.2f} pkt/s")
                            except (SyntaxError, ValueError) as e:
                                self.data_signal.emit(f"Error parsing CSI data: {str(e)}")
                                continue
                segment_count += 1
                if packet_count < self.target_packets:
                    self.data_signal.emit(f"Segment {segment_count} complete. Taking 10s break...")
                    time.sleep(self.break_duration)
            ser.close()
            if packets:
                df = pd.DataFrame(packets, columns=['prefix', 'type', 'mac', 'rssi', 'rate', 'sig_mode',
                                                    'mcs', 'bandwidth', 'smoothing', 'not_sounding', 'aggregation',
                                                    'stbc', 'fec_coding', 'sgi', 'noise_floor', 'ampdu_cnt', 'channel',
                                                    'secondary_channel', 'timestamp', 'len', 'dummy1', 'dummy2',
                                                    'extra_field1', 'extra_field2', 'csi_buf', 'activity'])
                df.to_csv(self.save_path, index=False, encoding='utf-8', errors='replace')
                self.data_signal.emit(f"Saved {len(packets)} packets to {self.save_path}")
            else:
                self.data_signal.emit("No valid CSI packets collected")
            self.data_signal.emit(f"Final packet rate: {packet_count / (time.time() - start_time):.2f} pkt/s")
        except Exception as e:
            self.data_signal.emit(f"Error: {str(e)}")

    def stop(self):
        self.running = False

class TrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(list, list, list, list)

    def __init__(self, model_type, data_path, model_path, epochs, batch_size):
        super().__init__()
        self.model_type = model_type
        self.data_path = data_path
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            # Load data
            X = np.load(os.path.join(os.path.dirname(self.data_path), 'features.npy'))
            y = np.load(os.path.join(os.path.dirname(self.data_path), 'labels.npy'), allow_pickle=True)

            # Normalize the features
            scaler = StandardScaler()
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
            joblib.dump(scaler, os.path.join(os.path.dirname(self.model_path), 'scaler.pkl'))

            # Shuffle the dataset
            shuffle_idx = np.random.permutation(len(y))
            X_scaled = X_scaled[shuffle_idx]
            y = y[shuffle_idx]

            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            joblib.dump(le, os.path.join(os.path.dirname(self.model_path), 'label_encoder.pkl'))

            # Compute class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weight_dict = dict(enumerate(class_weights))

            if self.model_type == 'Random Forest':
                # Flatten for Random Forest
                X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
                model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                model.fit(X_flat, y_encoded)
                joblib.dump(model, self.model_path)
                self.progress_signal.emit("Random Forest training complete")
            else:
                if self.model_type == 'LSTM':
                    # Ensure input shape (samples, time_steps, features)
                    if X_scaled.shape[1:] != (50, 257):
                        self.progress_signal.emit(f"Error: Expected shape (samples, 50, 257), got {X_scaled.shape}")
                        return
                    model = tf.keras.Sequential([
                        tf.keras.layers.LSTM(64, input_shape=(50, 257), return_sequences=True),
                        tf.keras.layers.LSTM(32),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(32, activation='relu'),
                        tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
                    ])
                else:  # CNN
                    # Reshape for CNN (samples, height, width, channels)
                    if X_scaled.shape[1:] != (50, 257):
                        self.progress_signal.emit(f"Error: Expected shape (samples, 50, 257), got {X_scaled.shape}")
                        return
                    X_cnn = X_scaled.reshape(X_scaled.shape[0], 50, 257, 1)
                    model = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 257, 1)),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
                    ])

                # Compile and train the model
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(X_scaled if self.model_type == 'LSTM' else X_cnn, y_encoded,
                                  epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2,
                                  verbose=1, class_weight=class_weight_dict)
                model.save(self.model_path)
                self.plot_signal.emit(history.history['loss'], history.history['val_loss'],
                                     history.history['accuracy'], history.history['val_accuracy'])
                self.progress_signal.emit(f"{self.model_type} training complete")
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")

class CSIGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing CSIGUI...")
        self.setWindowTitle("CSI Gesture Identification")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        collection_layout = QHBoxLayout()
        self.activity_combo = QComboBox()
        self.activity_combo.addItems(['empty', 'idle', 'sit', 'run', 'walk'])
        collection_layout.addWidget(QLabel("Activity:"))
        collection_layout.addWidget(self.activity_combo)
        self.port_input = QLineEdit("COM9")
        collection_layout.addWidget(QLabel("Port:"))
        collection_layout.addWidget(self.port_input)
        self.save_button = QPushButton("Select Save Path")
        self.save_button.clicked.connect(self.select_save_path)
        collection_layout.addWidget(self.save_button)
        self.start_collect_button = QPushButton("Start Collection")
        self.start_collect_button.clicked.connect(self.start_collection)
        collection_layout.addWidget(self.start_collect_button)
        self.stop_collect_button = QPushButton("Stop Collection")
        self.stop_collect_button.clicked.connect(self.stop_collection)
        self.stop_collect_button.setEnabled(False)
        collection_layout.addWidget(self.stop_collect_button)
        self.countdown_label = QLabel("Countdown: N/A")
        collection_layout.addWidget(self.countdown_label)
        main_layout.addLayout(collection_layout)

        training_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Random Forest', 'LSTM', 'CNN'])
        training_layout.addWidget(QLabel("Model:"))
        training_layout.addWidget(self.model_combo)
        self.data_button = QPushButton("Select Data Path")
        self.data_button.clicked.connect(self.select_data_path)
        training_layout.addWidget(self.data_button)
        self.model_button = QPushButton("Select Model Save Path")
        self.model_button.clicked.connect(self.select_model_path)
        training_layout.addWidget(self.model_button)
        self.epochs_input = QLineEdit("10")
        training_layout.addWidget(QLabel("Epochs:"))
        training_layout.addWidget(self.epochs_input)
        self.batch_input = QLineEdit("32")
        training_layout.addWidget(QLabel("Batch Size:"))
        training_layout.addWidget(self.batch_input)
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.start_training)
        training_layout.addWidget(self.train_button)
        main_layout.addLayout(training_layout)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        main_layout.addWidget(self.console)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Metrics")
        self.plot_widget.addLegend()
        main_layout.addWidget(self.plot_widget)

        self.save_path = ""
        self.log_path = ""
        self.data_path = ""
        self.model_path = ""
        self.collection_thread = None
        self.training_thread = None
        self.packet_counts = []
        self.amplitudes = []
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(200)
        print("CSIGUI initialization complete.")

    def select_save_path(self):
        self.save_path, _ = QFileDialog.getSaveFileName(self, "Select CSV Save Path", "", "CSV Files (*.csv)")
        if self.save_path:
            self.log_path = self.save_path.replace('.csv', '_log.txt')
            self.console.append(f"Save path: {self.save_path}, Log path: {self.log_path}")

    def select_data_path(self):
        self.data_path, _ = QFileDialog.getOpenFileName(self, "Select Processed Data", "", "Pickle Files (*.pkl)")
        if self.data_path:
            self.console.append(f"Data path: {self.data_path}")

    def select_model_path(self):
        self.model_path, _ = QFileDialog.getSaveFileName(self, "Select Model Save Path", "", "Model Files (*.pkl *.h5)")
        if self.model_path:
            self.console.append(f"Model path: {self.model_path}")

    def start_collection(self):
        if not self.save_path:
            self.console.append("Error: Select save path")
            return
        self.start_collect_button.setEnabled(False)
        self.stop_collect_button.setEnabled(True)
        self.packet_counts = []
        self.amplitudes = []
        self.collection_thread = CollectionThread(self.port_input.text(), self.activity_combo.currentText(),
                                                 self.save_path, self.log_path)
        self.collection_thread.data_signal.connect(self.console.append)
        self.collection_thread.stats_signal.connect(self.update_stats)
        self.collection_thread.start()

    def stop_collection(self):
        if self.collection_thread:
            self.collection_thread.stop()
            self.collection_thread = None
        self.start_collect_button.setEnabled(True)
        self.stop_collect_button.setEnabled(False)
        self.countdown_label.setText("Countdown: N/A")

    def start_training(self):
        if not self.data_path or not self.model_path:
            self.console.append("Error: Select data and model paths")
            return
        self.train_button.setEnabled(False)
        self.training_thread = TrainingThread(self.model_combo.currentText(), self.data_path, self.model_path,
                                             int(self.epochs_input.text()), int(self.batch_input.text()))
        self.training_thread.progress_signal.connect(self.console.append)
        self.training_thread.plot_signal.connect(self.plot_training)
        self.training_thread.finished.connect(lambda: self.train_button.setEnabled(True))
        self.training_thread.start()

    def update_stats(self, count, amplitude):
        self.packet_counts.append(count)
        self.amplitudes.append(amplitude)
        if len(self.packet_counts) > 100:
            self.packet_counts.pop(0)
            self.amplitudes.pop(0)

    def update_plot(self):
        self.plot_widget.clear()
        if self.packet_counts:
            recent_counts = self.packet_counts[-100:]
            self.plot_widget.plot(range(len(recent_counts)), recent_counts, pen='b', name='Packet Count')
            self.plot_widget.setLabel('left', 'Packet Count')
            self.plot_widget.setYRange(max(0, min(recent_counts) - 10), max(recent_counts) * 1.1)
        if self.amplitudes:
            recent_amplitudes = self.amplitudes[-100:]
            self.plot_widget.showAxis('right')
            view_box = self.plot_widget.getViewBox()
            view_box2 = pg.ViewBox()
            self.plot_widget.scene().addItem(view_box2)
            view_box2.setXLink(view_box)
            view_box2.setYRange(0, 100)
            axis = pg.AxisItem('right')
            self.plot_widget.getPlotItem().layout.addItem(axis, 2, 3)
            axis.linkToView(view_box2)
            axis.setLabel('CSI Std')
            self.plot_widget.plot(range(len(recent_amplitudes)), recent_amplitudes, pen='r', name='CSI Std', viewBox=view_box2)

    def plot_training(self, loss, val_loss, acc, val_acc):
        self.plot_widget.clear()
        epochs = list(range(1, len(loss) + 1))
        self.plot_widget.plot(epochs, loss, pen='b', name='Training Loss')
        self.plot_widget.plot(epochs, val_loss, pen='r', name='Validation Loss')
        self.plot_widget.plot(epochs, acc, pen='g', name='Training Accuracy')
        self.plot_widget.plot(epochs, val_acc, pen='y', name='Validation Accuracy')

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        print("QApplication initialized.")
        window = CSIGUI()
        print("CSIGUI window created.")
        window.show()
        print("Window shown. Starting event loop...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)