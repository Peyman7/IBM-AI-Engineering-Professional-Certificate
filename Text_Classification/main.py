# main.py
from models.model import TextClassificationModel
from utils.data_utils import get_dataloaders, vocab
from utils.train_eval import train, evaluate
from config import *
import json 
import matplotlib.pyplot as plt

# Prepare data loaders
train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, vocab, device)

# Initialize model, optimizer, criterion
model = TextClassificationModel(len(vocab), EMBED_DIM, NUM_CLASSES).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# Train and evaluate
results = train(model, criterion, optimizer, train_loader, val_loader, EPOCHS, model_path="model.pth")


# Save results to JSON
with open("training_results.json", "w") as f:
    json.dump(results, f)
print("Saved training results to training_results.json")

# Plot training progress
epochs = list(range(1, EPOCHS + 1))
plt.figure(figsize=(10, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, results['train_loss'], label="Train Loss")
plt.plot(epochs, results['validation_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, results['accuracy'], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
print("Saved training plot to training_plot.png")
