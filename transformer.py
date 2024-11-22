import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
  
        self.query_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.key_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.value_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, emb_dim)


    def _split_heads(self, hidden_states):
       
       batch_size, seq_len, emb_dim = hidden_states.shape
       hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
       
       return hidden_states.permute(0, 2, 1, 3)

    def _merge_heads(self, hidden_states):

      batch_size, seq_len, num_heads, head_dim = hidden_states.shape
      hidden_states = hidden_states.reshape(batch_size, seq_len, num_heads * head_dim)

      return hidden_states
	
    def forward(self, query, key, value, mask=None):

      query = self.query_proj(query)
      key = self.key_proj(key)
      value = self.value_proj(value)
      
      query = self._split_heads(query)
      key = self._split_heads(key)
      value = self._split_heads(value)
      
      key_out = query @ key.transpose(-2, -1)

      if mask is not None:
          key_out = key_out.masked_fill(mask == 0, -1e20)
	
      key_out = torch.softmax(key_out / self.head_dim**0.5, dim=-1)
      
      attn = key_out @ value
      
      attn = self._merge_heads(attn)
      
      attn_output = self.out_proj(attn)

      return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

	self.emb_dim = emb_dim
	self.num_heads = num_heads
	self.dropout = nn.Dropout(dropout)
	self.forward_dim = forward_dim

	self.norm = nn.LayerNorm(eps=1e-6)
	self.forward_norm = nn.LayerNorm(eps=1e-6)

	self.FNN = nn.Sequential(
		nn.Linear(self.emb_dim, self.forward_dim),
		nn.ReLU(),
		nn.Linear(self.forward_dim, self.emb_dim)
	)

    def forward(self, query, key, value, mask):
	# Attention
	attn = MultiHeadAttention(self.emb_dim, self.num_heads)
	attn = attn(query, key, value, mask)
	attn = attn + query # Skip con
	attn = self.dropout(attn)
	attn = self.norm(attn)

	# Feed Forward
	output = self.FFN(attn)
	output = output + attn # Skip con
	output = self.dropout(output)
	output = self.forward_norm(output)
	    
        return output
