import scipy.io
import numpy as np

data = scipy.io.loadmat("dataset/mat/HAR_complete.mat")

X = data["csi"].T
nsamples = data["nsamples"][0]

# Properly load classnames
label_names = [str(s) for s in data["classnames"].flatten()]

# Build y
y = []
for idx, count in enumerate(nsamples):
    y.extend([idx] * count)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", label_names)
