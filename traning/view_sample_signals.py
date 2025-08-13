import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# === Load your .mat file ===
data = scipy.io.loadmat("dataset/mat/HAR_complete.mat")  # adjust path if needed

# === Transpose CSI matrix to (samples, features) ===
X = data["csi"].T
nsamples = data["nsamples"][0]
classnames = [str(name).strip() for name in data["classnames"].flatten()]

# === Build labels ===
y = []
for i, count in enumerate(nsamples):
    y += [i] * count
y = np.array(y)

# === Plot one CSI window per class and save the image ===
plt.figure(figsize=(12, 10))
for i, label in enumerate(classnames):
    idx = np.where(y == i)[0][0]  # first sample of each class
    plt.subplot(len(classnames), 1, i + 1)
    plt.plot(X[idx])
    plt.title(f"CSI Sample - Class: {label.strip()}")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.xlabel("Feature Index")
plt.tight_layout()
plt.savefig("csi_sample_signals.png")
print("✅ Plot saved as csi_sample_signals.png")
