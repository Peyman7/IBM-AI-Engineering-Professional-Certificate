# main.py
from models.model import TextClassificationModel
from utils.data_utils import get_dataloaders, vocab
from utils.train_eval import train, evaluate
from config import *

# Prepare data loaders
train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, vocab, device)

# Initialize model, optimizer, criterion
model = TextClassificationModel(len(vocab), EMBED_DIM, NUM_CLASSES).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# Train and evaluate
train(model, criterion, optimizer, train_loader, val_loader, EPOCHS)
