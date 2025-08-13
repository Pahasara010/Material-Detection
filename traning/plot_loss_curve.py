import matplotlib.pyplot as plt

# 🔢 Replace with your real loss values per epoch if available
train_loss = [0.95, 0.72, 0.60, 0.55, 0.52]
val_loss   = [1.00, 0.78, 0.66, 0.60, 0.58]
epochs     = list(range(1, len(train_loss) + 1))

# 📉 Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
print("✅ Saved loss curve as 'loss_curve.png'")
