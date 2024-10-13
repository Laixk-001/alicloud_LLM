import torch.nn as nn
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.combination = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, mask=None):
        batch,time,dimention = query.size()
        query = self.w_q(query).view(batch, time, self.num_heads, self.head_dim).transpose(1,2)
        key = self.w_k(key).view(batch, time, self.num_heads, self.head_dim).transpose(1,2)
        value = self.w_v(value).view(batch, time, self.num_heads, self.head_dim).transpose(1,2)
        attn_weights = torch.matmul(query, key.transpose(-2,-1))
        attn_weights = attn_weights / self.head_dim**0.5
        # if mask is not None:
        #     attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, value).transpose(1,2).contiguous().view(batch, time, -1)
        attn_output = self.combination(attn_output)
        return attn_output

batch = 12
d_model = 512
num_heads = 8
mask = torch.ones(batch, 10, 10)
query = torch.randn(batch, 10, d_model)
key = torch.randn(batch, 10, d_model)
value = torch.randn(batch, 10, d_model)
attn_output = MultiheadAttention(d_model, num_heads)(query, key, value, mask)
print(attn_output)

