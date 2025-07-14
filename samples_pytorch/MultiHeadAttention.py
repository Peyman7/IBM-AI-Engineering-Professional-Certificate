import torch
import torch.nn as nn
import torch.nn.functional as F

""" Built Transformer with MultiHead Attention """
# Positional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, n_dim, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, n_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        ### using the original positional encoding formula in Vasani et al, 2017
        div_term = torch.exp(torch.arange(0, n_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_dim]
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)  # shape: [1, seq_len, n_dim]

# Multh head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, vocab_size, n_dim):
        super(MultiHeadAttention, self).__init__()
        assert (n_dim % num_heads ==0)
        self.n_heads = num_heads
        self.head_dim = n_dim // n_heads
        
        self.q_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, n_dim, bias=False)
        self.out_proj = nn.Linear(n_dim, n_dim)
        
    def forward(self, x):
        batch_size, seq_len, n_dim = x.size() 
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        weights = F.softmax(scores)
        attn_out = weights @ v #[B, h, T, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_dim)
        
        output = self.out_proj(attn_out)
        
        return output, weights

# Transformer Block        
class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, n_dim, n_heads, max_len=100):
        super(TransformerBlock, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.pos_encoder = PositionalEncoding(max_len, n_dim)
        self.mha = MultiHeadAttention(n_heads, vocab_size, n_dim)
        self.norm = nn.LayerNorm(n_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        attn_out, weights = self.mha(x)
        x = self.norm(attn_out + x)
        return x, weights
        
        
        
###################################
## initial testing ##
vocab_size = 1000
n_dim = 64
n_heads = 8

model = TransformerBlock(vocab_size, n_dim, n_heads)

input_ids = torch.tensor([[2, 45, 67, 89, 23, 9]])
output, weights = model(input_ids)

print("Output shape:", output.shape)   # [1, 6, 64]
print("Attention shape:", weights.shape)  # [1, 8, 6, 6] 