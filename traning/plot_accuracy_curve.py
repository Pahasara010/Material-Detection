# plot_accuracy_curve.py

import matplotlib.pyplot as plt

# Replace with your real values
train_acc = [0.60, 0.68, 0.71, 0.73, 0.74]  # example
val_acc = [0.58, 0.65, 0.69, 0.72, 0.74]    # example

epochs = list(range(1, len(train_acc) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png")
print("✅ Saved accuracy curve as 'accuracy_curve.png'")
