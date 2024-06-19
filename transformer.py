import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import config
INIT_STD = 0.02

temp_mask = torch.ones((512, 512), dtype=torch.bool).tril(diagonal=0).cuda()
# %%
# temp_mask
# # %%
# temp_mask = torch.logical_xor(temp_mask, torch.ones((512, 512), dtype=torch.bool).tril(diagonal=-256)).cuda()
# print(temp_mask.size())

# attn_val = [[] for _ in range(6)]
class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        theta = 1. / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer('theta', theta)

    def forward(self, qk):
        # qk: batch, num_head*2, sequence, head_dim

        s = torch.arange(qk.size(2), device=qk.device)
        
        freqs = torch.outer(s, self.theta) # seq_len, dim // 2
        freqs = torch.cat((freqs, freqs), dim=-1)

        qk1, qk2 = qk.chunk(2, dim=-1)
        qk2 = torch.cat((-qk2, qk1), dim=-1)

        return qk * freqs.cos() + qk2 * freqs.sin()

# class PE(nn.Module):
#     def __init__(self, head_dim, base=10000):
#         super().__init__()
#         theta = torch.arange(512) - 256
#         self.register_buffer('theta', theta)

#     def forward(self, qk):
#         # qk: batch, num_head*2, sequence, head_dim
#         b, h, s, d = qk.size()
#         qk = qk.view(b, h, s, d//4, 4)
#         decay = (torch.arange(16, device=qk.device, dtype=qk.dtype) + 1) / 500

#         freqs = torch.outer(self.theta, decay) # seq_len, 8
        
#         freqs = freqs.exp().unsqueeze(-1)

#         q, k = qk.chunk(2, 1)

#         q = q /freqs

#         k = k * freqs

#         return q.view(b, h//2, s, d), k.view(b, h//2, s, d)

class RMSNorm(nn.Module):

    def __init__(self, dim_size, eps=1e-6):
        super().__init__()

        self.root_dim = math.sqrt(dim_size)
        self.weight = nn.Parameter(torch.ones(dim_size))
        self.eps = eps
    
    def forward(self, x):

        x = F.normalize(x, dim=-1, eps=self.eps) * self.root_dim * self.weight

        return x

class GropuRMSNorm(nn.Module):

    def __init__(self, dim_size, num_groups, eps=1e-6):
        super().__init__()

        group_dim = dim_size // num_groups
        self.root_dim = math.sqrt(group_dim)
        self.weight = nn.Parameter(torch.ones(num_groups, 1))
        self.eps = eps
    
    def forward(self, x):

        x = F.normalize(x, dim=-1, eps=self.eps) * self.root_dim * self.weight

        return x

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.hidden = input_dim*8//3
        self.in_proj = nn.Linear(input_dim, self.hidden*2)
        self.out_proj = nn.Linear(self.hidden, input_dim)

        torch.nn.init.normal_(self.in_proj.weight, std=INIT_STD)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):

        x = self.in_proj(x)
        x1, x2 = x.split([self.hidden, self.hidden], 2)
        # attn = torch.einsum('b h x d, b h d y -> b h x y', q, k.transpose(2, 3)) / math.sqrt(self.input_dim//self.num_heads)
        # attn = torch.masked_fill(attn, torch.logical_not(temp_mask), float('-inf'))
        # attn = torch.softmax(attn, 3)
        # mask = torch.abs(F.silu(x1)) > 5e-2
        # print(mask.float().mean())
        # attn_val.append(mask.detach().clone().cpu().numpy())
        x = F.silu(x1) * x2

        x = self.out_proj(x)

        return x


class Attention(nn.Module):

    def __init__(self, input_dim, num_heads, layer_idx):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        
        self.qkv = nn.Linear(input_dim, input_dim*3, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.rope = RoPE(input_dim//num_heads)

        torch.nn.init.normal_(self.qkv.weight, std=INIT_STD)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):

        qkv = self.qkv(x)

        qkv = rearrange(qkv, 'b s (h d) -> b h s d', h=self.num_heads*3)
        qk, v = qkv.split([self.num_heads*2, self.num_heads], 1)
        qk = self.rope(qk)
        q, k = qk.chunk(2, 1)

        # with torch.no_grad():
        #     attn = torch.einsum('b h x d, b h d y -> b h x y', q, k.transpose(2, 3)) / math.sqrt(self.input_dim//self.num_heads)
        #     attn = torch.masked_fill(attn, torch.logical_not(temp_mask), float('-inf'))
        #     attn = torch.softmax(attn, 3)
        #     mask = attn > 1e-4
        #     val = torch.masked_select(mask, temp_mask)
        # attn_val[self.layer_idx].append(val.detach().clone().cpu())

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.out_proj(x)

        return x

    
    


class Block(nn.Module):

    def __init__(self, input_dim, num_heads, layer_idx):
        super().__init__()

        self.attn = Attention(input_dim, num_heads, layer_idx)
        self.mlp = MLP(input_dim)

        self.norm1 = RMSNorm(input_dim)
        self.norm2 = RMSNorm(input_dim)



    def forward(self, x):

        x_out = self.attn(self.norm1(x))
        x = x_out + x

        x_out = self.mlp(self.norm2(x))
        x = x_out + x

        return x


class Model(nn.Module):

    def __init__(self, num_embeddings, input_dim, num_heads, num_layers, tie=False):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings, input_dim)
        self.layers = nn.ModuleList([Block(input_dim, num_heads, i) for i in range(num_layers)])
        self.head = nn.Linear(input_dim, num_embeddings, bias=False)

        self.norm = RMSNorm(input_dim) 


        nn.init.normal_(self.embedding.weight, std=INIT_STD)
        nn.init.normal_(self.head.weight, std=INIT_STD)

        for n, p in self.named_parameters():
            if 'out_proj' in n and 'weight' in n:
                torch.nn.init.normal_(p, std=INIT_STD/math.sqrt(num_layers*2)) 

        if tie:
            self.head.weight = self.embedding.weight

    def forward(self, x):
        
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.head(x)

        return x
