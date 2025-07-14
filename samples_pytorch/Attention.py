import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, n_dim, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, n_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_dim]
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)  # shape: [1, seq_len, n_dim]
        
        
        
class AttentionHead(nn.Module):
    def __init__(self, vocab_size, n_dim, max_len=100):
        super(AttentionHead, self).__init__()
        self.embdx = nn.Embedding(vocab_size, n_dim)
        self.pos_encoder = PositionalEncoding(max_len, n_dim)
        self.query = nn.Linear(n_dim, n_dim, bias=False)
        self.key = nn.Linear(n_dim, n_dim, bias=True)
        self.value = nn.Linear(n_dim, n_dim, bias=False)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embed_layer = self.embdx(x)  # [batch_size, seq_len, n_dim]
        x = self.pos_encoder(embed_layer)  # add positional encoding

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = attn_weights @ v  # [batch_size, seq_len, n_dim]
        return out, attn_weights
        
        
vocab_size = 1000
n_dim = 16
model = AttentionHead(vocab_size, n_dim)

# Dummy input: batch of 1 sentence with 6 tokens
input_ids = torch.tensor([[2, 45, 67, 89, 23, 9]])
output, weights = model(input_ids)

print("Output shape:", output.shape) 
print("Attention weights shape:", weights.shape)  