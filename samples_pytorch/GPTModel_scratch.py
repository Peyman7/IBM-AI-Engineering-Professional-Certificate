import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import shutil


# ----------- Tokenizer Setup ------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
block_size = 128

# ----------- Data Processing ------------
def tokenize(example):
    tokens = tokenizer(example["text"], add_special_tokens=False)
    return {"input_ids": tokens["input_ids"]}

def group_texts(examples):
    joined = sum(examples["input_ids"], [])
    total_len = (len(joined) // block_size) * block_size
    joined = joined[:total_len]

    input_ids = [joined[i:i + block_size] for i in range(0, total_len, block_size)]
    labels = [x[1:] + [tokenizer.eos_token_id] for x in input_ids]

    return {"input_ids": input_ids, "labels": labels}

dataset_raw = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    cache_dir="/content/hf_cache_clean"
)
tokenized_dataset = dataset_raw.map(tokenize, batched=True, remove_columns=["text"])
lm_dataset = tokenized_dataset.map(group_texts, batched=True)
lm_dataset.set_format(type="torch", columns=["input_ids", "labels"])

train_dataset = lm_dataset["train"]
val_dataset = lm_dataset["validation"]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
dataset = {"train_loader": train_loader, "val_loader": val_loader}

# ----------- Model Definition ------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, n_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, n_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_dim, 2).float() * (-math.log(10000.0) / n_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:seq_len].unsqueeze(0))

class GPTModel(nn.Module):
    def __init__(self, n_dim, vocab_size, num_heads, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, n_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(n_dim, vocab_size)

        self.n_dim = n_dim

    def forward(self, x):
        B, T = x.size()
        x = self.embedding(x) * math.sqrt(self.n_dim)
        x = self.positional_encoding(x)

        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        output = self.transformer_encoder(x, mask)
        logits = self.lm_head(output)
        return logits

# ----------- Training Function ------------
def train(model, criterion, optimizer, scheduler, dataset, epochs, print_every=10, max_grad_norm=1.0, device='cpu'):
    results = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    save_path = "best_gpt_model.pth"

    train_loader = dataset['train_loader']
    val_loader = dataset['val_loader']

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            y = y.view(B * T)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)

                logits = model(x)
                B, T, V = logits.shape
                logits = logits.view(B * T, V)
                y = y.view(B * T)

                val_loss = criterion(logits, y)
                total_val_loss += val_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        results['train_loss'].append(avg_train_loss)
        results['val_loss'].append(avg_val_loss)

    return results

# ----------- Text Generation Function ------------
def generate_text(model, tokenizer, start_text, length=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    context = tokenizer.encode(start_text, return_tensors="pt").to(device)
    generated = context

    for _ in range(length):
        input_ids = generated[:, -block_size:]
        with torch.no_grad():
            logits = model(input_ids)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0])

# ----------- Hyperparameters + Training ------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTModel(
    n_dim=128,
    vocab_size=vocab_size,
    num_heads=4,
    num_layers=2,
    max_seq_len=block_size,
    dropout=0.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
criterion = nn.CrossEntropyLoss()

results = train(
    model,
    criterion,
    optimizer,
    scheduler,
    dataset,
    epochs=5,
    print_every=1,
    max_grad_norm=1.0,
    device=device
)

# ----------- Generate Sample Text ------------
print(generate_text(model, tokenizer, "In the future", length=50))