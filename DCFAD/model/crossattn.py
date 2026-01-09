import torch
import torch.nn as nn
import math


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv, attn_mask=None):

        B, T_q, _ = x_q.shape
        B, T_kv, _ = x_kv.shape

        # Linear projections
        Q = self.q_proj(x_q)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        # Split into heads
        Q = Q.view(B, T_q, self.n_heads, self.d_k).transpose(1,2)
        K = K.view(B, T_kv, self.n_heads, self.d_k).transpose(1,2)
        V = V.view(B, T_kv, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1,2).contiguous().view(B, T_q, self.d_model)  # [B, T_q, D]
        out = self.out_proj(context)
        out = self.norm(out + x_q)

        return out, attn_weights

