#!/usr/bin/env python3

import sys
import pathlib
import logging
import contextlib
import io
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from HAR.transformers import ActivityRecognitionPipeline
from HAR.io import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSIModelTrainer:
    ACTIVITY_LABELS = ["walking", "running", "idle", "empty"]
    NUM_CLASSES = len(ACTIVITY_LABELS)
    NUM_KERNELS = 500

    def __init__(self, main_dataset_path, holdout_dataset_path, train_fraction, output_dir) -> None:
        self.main_dataset_path = main_dataset_path
        self.holdout_dataset_path = holdout_dataset_path
        self.train_fraction = train_fraction
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model pipeline with progress bars disabled
        self.pipeline = ActivityRecognitionPipeline(
            num_classes=self.NUM_CLASSES,
            num_kernels=self.NUM_KERNELS,
            batch_size=64,
            normalize=True,
            show_progress=False,  # Disable Rocket's internal progress bars
        )

    def run(self):
        # Define stages and their relative weights for the progress bar
        stages = [
            ("Loading and splitting main dataset", 1),
            ("Training the model", 5),  # Training is the heaviest step
            ("Evaluating on validation set", 2),
            ("Loading holdout dataset", 1),
            ("Evaluating on holdout set", 2),
            ("Saving model", 1),
        ]
        total_units = sum(weight for _, weight in stages)

        # Create a single progress bar for the entire training session
        with tqdm(total=total_units, desc="Training Session", unit="step") as pbar:
            # Stage 1: Load and split main dataset
            pbar.set_description("Loading and splitting main dataset")
            X_train, X_test, y_train, y_test = self._prepare_data_splits(
                self.main_dataset_path, self.train_fraction
            )
            pbar.update(stages[0][1])

            # Stage 2: Train the model
            pbar.set_description(f"Training the model with {X_train.shape[0]} samples")
            with contextlib.redirect_stdout(io.StringIO()):  # Suppress Rocket's progress bars
                self._train_model(X_train, y_train)
            pbar.update(stages[1][1])

            # Stage 3: Evaluate on validation set
            pbar.set_description("Evaluating on validation set")
            with contextlib.redirect_stdout(io.StringIO()):  # Suppress Rocket's progress bars
                self._evaluate_model(X_test, y_test)
            pbar.update(stages[2][1])

            # Stage 4: Load holdout dataset
            pbar.set_description("Loading holdout dataset")
            X_holdout, _, y_holdout, _ = self._prepare_data_splits(self.holdout_dataset_path, None)
            pbar.update(stages[3][1])

            # Stage 5: Evaluate on holdout set
            pbar.set_description("Evaluating on holdout set")
            with contextlib.redirect_stdout(io.StringIO()):  # Suppress Rocket's progress bars
                self._evaluate_model(X_holdout, y_holdout)
            pbar.update(stages[4][1])

            # Stage 6: Save the model
            pbar.set_description("Saving model")
            self.pipeline.save(self.output_dir)
            logger.info(f"Trained model saved to: {self.output_dir}")
            pbar.update(stages[5][1])

    def _prepare_data_splits(self, dataset_path, train_fraction):
        X, y, _, _, input_shape = load_dataset(dataset_path)
        X = X.reshape(X.shape[0], *input_shape)  # Ensure correct shape for input
        logger.info(f"Dataset loaded: {X.shape[0]} samples with shape {X.shape[1:]}")
        if train_fraction is None:
            return X, None, y, None
        return train_test_split(X, y, train_size=train_fraction, stratify=y)

    def _train_model(self, X, y):
        self.pipeline.fit(X, y)

    def _evaluate_model(self, X, y):
        predictions = self.pipeline.predict(X)
        logger.info(f"Accuracy: {accuracy_score(y, predictions)*100:.2f}%")
        logger.info("Confusion Matrix:")
        logger.info(confusion_matrix(y, predictions))
        logger.info("Classification Report:")
        logger.info(classification_report(y, predictions, target_names=self.ACTIVITY_LABELS))

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

    parser = argparse.ArgumentParser(description="Train HAR model using CSI data")
    parser.add_argument("--main-set", help="Path to main dataset for training and initial testing", type=str, required=True)
    parser.add_argument("--hold-set", help="Path to holdout dataset for final testing", type=str, required=True)
    parser.add_argument("--train-size", help="Proportion of main set to use for training", type=valid_fraction, default=0.8)
    parser.add_argument("--dump", help="Directory to save trained model parameters", type=str, required=True)

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        logger.error("Training interrupted by user.")