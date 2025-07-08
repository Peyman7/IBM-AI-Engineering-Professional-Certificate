import torch
from models.model import TextClassificationModel
from utils.data_utils import get_dataloaders, vocab
from config import *
import torch.nn as nn

def test(model, dataloader, criterion, device):
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            outputs = model(text, offsets)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == label).sum().item()
            total_count += label.size(0)
    avg_acc = total_acc / total_count
    avg_loss = total_loss / total_count
    return avg_acc, avg_loss


if __name__ == "__main__":
    # Load test data
    _, _, test_loader = get_dataloaders(BATCH_SIZE, vocab, device)

    # Model setup (make sure weights are loaded if saved)
    model = TextClassificationModel(len(vocab), EMBED_DIM, NUM_CLASSES).to(device)
    
    # Optionally load trained model weights
    model.load_state_dict(torch.load("model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()

    # Evaluate
    acc, loss = test(model, test_loader, criterion, device)

    print(f"\n Test Accuracy: {acc*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
