#!/usr/bin/env python3

import sys
import pathlib
import logging
import contextlib
import io
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, sosfilt

# Assuming MID.transformers and MID.io are custom modules for data loading
from MID.transformers import CSIAmplitudeMinMaxScaler
from MID.io import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class CSIModelTrainer:
    MATERIAL_LABELS = ["iron", "wood", "plastic", "water", "fabric"]
    NUM_CLASSES = len(MATERIAL_LABELS)

    def __init__(self, main_dataset_path, holdout_dataset_path, train_fraction, output_dir):
        self.main_dataset_path = main_dataset_path
        self.holdout_dataset_path = holdout_dataset_path
        self.train_fraction = train_fraction
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = CSIAmplitudeMinMaxScaler()

    def run(self):
        stages = [
            ("Loading and splitting main dataset", 1),
            ("Training the model and accuracy curve", 6),
            ("Evaluating on validation set", 2),
            ("Loading holdout dataset", 1),
            ("Evaluating on holdout set", 2),
            ("Saving model and plots", 2),
        ]
        total_units = sum(weight for _, weight in stages)

        with tqdm(total=total_units, desc="Training Session", unit="step") as pbar:
            # Stage 1: Load and prepare main dataset
            pbar.set_description("Loading and splitting main dataset")
            X, y, _, _, input_shape = load_dataset(self.main_dataset_path)
            X = self.preprocess_data(X)
            X = X.reshape(X.shape[0], *input_shape)
            logger.info(f"Dataset loaded: {X.shape[0]} samples with shape {X.shape[1:]}")
            pbar.update(stages[0][1])

            # Stage 2: Train the model
            pbar.set_description("Training the model and accuracy curve")
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_fraction, stratify=y)
            model = CNNLSTMModel(self.NUM_CLASSES, input_shape)
            accuracies = self._train_model_with_curve(X_train, y_train, model)
            self._plot_accuracy_curve(accuracies)
            pbar.update(stages[1][1])

            # Stage 3: Evaluate on validation set
            pbar.set_description("Evaluating on validation set")
            val_predictions = self._predict(model, X_test)
            self._plot_confusion_matrix(y_test, val_predictions, "validation_confusion_matrix.png")
            logger.info(f"Validation Accuracy: {accuracy_score(y_test, val_predictions)*100:.2f}%")
            logger.info("Validation Confusion Matrix:")
            logger.info(confusion_matrix(y_test, val_predictions))
            logger.info("Validation Classification Report:")
            logger.info(classification_report(y_test, val_predictions, target_names=self.MATERIAL_LABELS))
            pbar.update(stages[2][1])

            # Stage 4: Load holdout dataset
            pbar.set_description("Loading holdout dataset")
            X_holdout, _, y_holdout, _ = self._prepare_data_splits(self.holdout_dataset_path, None)
            X_holdout = self.preprocess_data(X_holdout)
            pbar.update(stages[3][1])

            # Stage 5: Evaluate on holdout set
            pbar.set_description("Evaluating on holdout set")
            holding_predictions = self._predict(model, X_holdout)
            self._plot_confusion_matrix(y_holdout, holdout_predictions, "holdout_confusion_matrix.png")
            logger.info(f"Holdout Accuracy: {accuracy_score(y_holdout, holdout_predictions)*100:.2f}%")
            logger.info("Holdout Confusion Matrix:")
            logger.info(confusion_matrix(y_holdout, holdout_predictions))
            logger.info("Holdout Classification Report:")
            logger.info(classification_report(y_holdout, holdout_predictions, target_names=self.MATERIAL_LABELS))
            pbar.update(stages[4][1])

            # Stage 6: Save the model and plots
            pbar.set_description("Saving model and plots")
            torch.save(model.state_dict(), self.output_dir / "cnn_lstm_model.pth")
            plt.savefig(self.output_dir / "accuracy_curve.png")
            logger.info(f"Trained model and plots saved to: {self.output_dir}")
            pbar.update(stages[5][1])

    def preprocess_data(self, X):
        fs = 20
        nyq = 0.5 * fs
        cutoff = 5 / nyq
        sos = butter(4, cutoff, 'low', output='sos')
        X = sosfilt(sos, X, axis=2)
        X = self.scaler.fit_transform(X)
        return X

    def _prepare_data_splits(self, dataset_path, train_fraction):
        X, y, _, _, input_shape = load_dataset(dataset_path)
        X = self.preprocess_data(X)
        X = X.reshape(X.shape[0], *input_shape)
        if train_fraction is None:
            return X, None, y, None
        return train_test_split(X, y, train_size=train_fraction, stratify=y)

    def _train_model_with_curve(self, X, y, model):
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            train_dataset = CSIDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(10):
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_predictions = self._predict(model, X_val)
                accuracy = accuracy_score(y_val, val_predictions)
                accuracies.append(accuracy)
                logger.info(f"Cross-validation fold accuracy: {accuracy*100:.2f}%")
        
        return accuracies

    def _predict(self, model, X):
        model.eval()
        dataset = CSIDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32)
        predictions = []
        with torch.no_grad():
            for data, _ in loader:
                output = model(data)
                predictions.extend(torch.argmax(output, dim=1).numpy())
        return np.array(predictions)

    def _plot_accuracy_curve(self, accuracies):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b')
        plt.title('Accuracy Curve Across Cross-Validation Folds')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.savefig(self.output_dir / "accuracy_curve.png")
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, filename):
        try:
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"Generating confusion matrix with data: {cm}")
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {filename.split(".")[0]}')
            plt.colorbar()
            tick_marks = np.arange(len(self.MATERIAL_LABELS))
            plt.xticks(tick_marks, self.MATERIAL_LABELS, rotation=45)
            plt.yticks(tick_marks, self.MATERIAL_LABELS)
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=8)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(self.output_dir / filename)
            logger.info(f"Confusion matrix saved to {self.output_dir / filename}")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")

def main(args):
    trainer = CSIModelTrainer(args.main_set, args.hold_set, args.train_size, args.dump)
    trainer.run()

if __name__ == "__main__":
    import argparse

    def valid_fraction(value):
        float_value = float(value)
        if float_value < 0 or float_value > 1:
            raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
        return float_value

    parser = argparse.ArgumentParser(description="Train material identification model using CSI data")
    parser.add_argument("--main-set", help="Path to main dataset for training and initial testing", type=str, required=True)
    parser.add_argument("--hold-set", help="Path to holdout dataset for final testing", type=str, required=True)
    parser.add_argument("--train-size", help="Proportion of main set to use for training", type=valid_fraction, default=0.8)
    parser.add_argument("--dump", help="Directory to save trained model parameters", type=str, required=True)

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logger.error("Training interrupted by user.")