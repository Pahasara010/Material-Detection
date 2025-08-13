import numpy as np
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Sample labels: {y_train[:10]}")