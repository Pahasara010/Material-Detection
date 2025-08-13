import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from HAR.transformers import ActivityRecognitionPipeline
from HAR.io import load_dataset
from pathlib import Path

# === Load dataset ===
X, y, _, _, input_shape = load_dataset("dataset/mat/HAR_complete.mat")
X = X.reshape(X.shape[0], *input_shape)

# === Load the trained model ===
pipeline = ActivityRecognitionPipeline(
    num_classes=5,
    num_kernels=500,
    batch_size=64,
    normalize=True,
    show_progress=False
)
pipeline.load(Path("saved_models/"))  # ✅ FIXED

# === Predict ===
y_pred = pipeline.predict(X)

# === Labels ===
labels = ["walking", "running", "jumping", "idle", "empty"]

# === Confusion Matrix ===
cm = confusion_matrix(y, y_pred)

# === Plot Heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Saved real confusion matrix heatmap as 'confusion_matrix.png'")
