from sklearn.base import ClassifierMixin, BaseEstimator
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import pickle
import logging

logger = logging.getLogger(__name__)


class PrincipalComponentVarianceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, variance_threshold) -> None:
        super().__init__()
        self.variance_threshold = variance_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        U, _, _ = np.linalg.svd(X)
        pc2_projection = np.einsum("ijk,ij->ik", X, U[:, :, 1])
        sample_variances = np.var(pc2_projection, axis=1)
        logger.debug(f"[PC2Classifier] variances = {sample_variances}")
        return sample_variances > self.variance_threshold


class RidgeEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, num_classes, show_progress=True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.show_progress = show_progress
        self._models = None

    def fit(self, X, y):
        _, num_subcarriers, _ = X.shape
        if self._models is None:
            self._models = Parallel(n_jobs=-2, backend="threading")(
                delayed(self._train_single_model)(X[:, sc_idx, :], y)
                for sc_idx in tqdm(range(num_subcarriers), disable=not self.show_progress)
            )
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        num_samples, num_subcarriers, _ = X.shape
        final_predictions = np.zeros((num_samples,))

        for sample_idx in range(num_samples):
            sample_votes = Parallel(n_jobs=1, backend="threading")(
                delayed(self._predict_single_model)(
                    self._models[sc_idx], X[sample_idx, sc_idx, :][None, :]
                ) for sc_idx in range(num_subcarriers)
            )
            labels, counts = np.unique(sample_votes, return_counts=True)
            final_predictions[sample_idx] = labels[np.argmax(counts)]

        logger.debug(f"[RidgeEnsembleClassifier] predictions = {final_predictions}")
        return final_predictions

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self._models, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Models saved to {filepath}")

    def load(self, filepath):
        logger.info(f"Loading models from {filepath}")
        with open(filepath, "rb") as f:
            self._models = pickle.load(f)

    def _train_single_model(self, X, y):
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        model.fit(X, y)
        return model

    def _predict_single_model(self, model, X):
        return model.predict(X)
