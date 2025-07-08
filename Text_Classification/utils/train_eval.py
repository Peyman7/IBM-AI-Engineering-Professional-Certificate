# utils/train_eval.py
import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_acc, total_count, total_loss = 0, 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_loss += loss.item()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count, total_loss

def train(model, criterion, optimizer, train_loader, val_loader, epochs, model_path="model.pth"):
    best_val_acc = 0.0
    results = {'train_loss': [], 'validation_loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for label, text, offsets in train_loader:
            label, text, offsets = label.to(model.device), text.to(model.device), offsets.to(model.device)

            optimizer.zero_grad()
            output = model(text, offsets)
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()

        acc, val_loss = evaluate(model, val_loader, criterion, model.device)
        results['train_loss'].append(total_loss)
        results['validation_loss'].append(val_loss)
        results['accuracy'].append(acc)

        print(f"Epoch {epoch+1}: Train Loss={total_loss:.2f}, Val Loss={val_loss:.2f}, Accuracy={acc:.4f}")

        # Save best model
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved (accuracy: {best_val_acc:.4f})")

    return results