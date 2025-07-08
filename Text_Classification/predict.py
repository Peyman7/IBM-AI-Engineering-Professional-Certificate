import torch
from models.model import TextClassificationModel
from utils.data_utils import tokenizer, vocab
from config import *

# Class labels (AG_NEWS dataset)
LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

def predict(text, model, vocab, device):
    model.eval()
    with torch.no_grad():
        # Preprocess
        tokens = tokenizer(text.lower())
        token_ids = torch.tensor([vocab[token] for token in tokens], dtype=torch.int64).to(device)
        offsets = torch.tensor([0]).to(device)

        # Predict
        output = model(token_ids, offsets)
        predicted_class = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted_class].item()
        return LABELS[predicted_class], confidence

if __name__ == "__main__":
    # Load model
    model = TextClassificationModel(len(vocab), EMBED_DIM, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))

    # Sample input
    sample_text = input("Enter a news headline: ")

    # Run prediction
    label, conf = predict(sample_text, model, vocab, device)

    print(f"\n Predicted category: **{label}**")
    print(f" Confidence: {conf*100:.2f}%")
