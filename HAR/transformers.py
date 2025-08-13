import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import joblib
from .rocket_functions import generate_kernels, apply_kernels

class CSIAmplitudeMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, X, y=None):
        self.min_vals = np.min(X, axis=(0, 2), keepdims=True)
        self.max_vals = np.max(X, axis=(0, 2), keepdims=True)
        return self

    def transform(self, X):
        X = X - self.min_vals
        with np.errstate(divide='ignore', invalid='ignore'):
            X = np.divide(X, self.max_vals, out=np.zeros_like(X), where=self.max_vals != 0)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class Rocket(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=10000):
        self.num_kernels = num_kernels
        self.kernel_bank = None

    def fit(self, X, y=None):
        # X shape: (n_samples, n_channels, signal_length)
        signal_length = X.shape[2]
        self.kernel_bank = generate_kernels(signal_length, self.num_kernels)
        return self

    def transform(self, X):
        # X shape: (n_samples, n_channels, signal_length)
        n_samples, n_channels, signal_length = X.shape
        transformed = []

        for i in tqdm(range(n_samples), desc="🚀 Rocket Transforming"):
            # Process each sample's channels
            sample = X[i]  # shape: (n_channels, signal_length)
            sample_transformed = apply_kernels(sample, self.kernel_bank)  # shape: (n_channels, num_kernels * 2)
            transformed.append(sample_transformed.mean(axis=0))  # Average over channels

        return np.vstack(transformed)  # shape: (n_samples, num_kernels * 2)

class RocketFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_kernels=10000, show_progress=True):
        self.n_kernels = n_kernels
        self.rocket = Rocket(num_kernels=self.n_kernels)
        self.show_progress = show_progress

    def fit(self, X, y=None):
        self.rocket.fit(X)
        return self

    def transform(self, X):
        batch_size = 1  # Controls RAM usage
        transformed = []

        for i in tqdm(range(0, X.shape[0], batch_size), desc="🚀 Rocket Transforming", disable=not self.show_progress):
            batch = X[i:i + batch_size]
            transformed_batch = self.rocket.transform(batch)
            transformed.append(transformed_batch)

        return np.vstack(transformed)

class ActivityRecognitionPipeline:
    def __init__(self, num_classes=5, num_kernels=10000, batch_size=64, normalize=True, show_progress=True):
        self.num_classes = num_classes
        self.num_kernels = num_kernels
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self.pipeline = None

    def fit(self, X, y):
        steps = []
        if self.normalize:
            steps.append(("scaler", CSIAmplitudeMinMaxScaler()))
        steps.append(("rocket", RocketFeatureExtractor(n_kernels=self.num_kernels, show_progress=self.show_progress)))
        steps.append(("classifier", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))))

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, path):
        joblib.dump(self.pipeline.named_steps["scaler"], path / "scaler.pkl")
        joblib.dump(self.pipeline.named_steps["rocket"], path / "kernels.pkl")
        joblib.dump(self.pipeline.named_steps["classifier"], path / "models.pkl")

    def load(self, path):
        self.pipeline = Pipeline([
            ("scaler", joblib.load(path / "scaler.pkl")),
            ("rocket", joblib.load(path / "kernels.pkl")),
            ("classifier", joblib.load(path / "models.pkl")),
        ])
        return self