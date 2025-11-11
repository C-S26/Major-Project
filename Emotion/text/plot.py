import pickle
import matplotlib.pyplot as plt

# Load metrics
with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

train_losses = metrics['train_losses']
train_accuracies = metrics['train_accuracies']
precision_list = metrics['precision_list']
recall_list = metrics['recall_list']
f1_list = metrics['f1_list']

epochs = range(1, len(train_losses) + 1)

# Plot Loss
plt.figure()
plt.plot(epochs, train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epochs, train_accuracies, marker='o')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Plot Precision, Recall, F1
plt.figure()
plt.plot(epochs, precision_list, label="Precision", marker='o')
plt.plot(epochs, recall_list, label="Recall", marker='o')
plt.plot(epochs, f1_list, label="F1-Score", marker='o')
plt.title("Precision, Recall, F1 per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
