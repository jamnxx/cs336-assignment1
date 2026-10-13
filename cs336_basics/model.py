import torch
import torch.nn as nn
from einops import rearrange
from .layers import Linear, RMSNorm, silu, Embedding
from .attention import MultiHeadSelfAttention

class SwiGlu(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)

        return self.w2(gate * value)


class TransformerBlock(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float = 10000.0,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            max_seq_len = max_seq_len,
            theta = theta,
            device = device,
            dtype = dtype,
        )
        self.ffn = SwiGlu(
            d_model = d_model,
            d_ff = d_ff,
            device = device,
            dtype = dtype,
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = x + self.attn(self.ln1(x), token_positions)
        y = z + self.ffn(self.ln2(z))
        return y

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            theta: float = 10000.0,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(in_features = d_model, out_features = vocab_size, device=device, dtype=dtype)

    def forward(
            self,
            token_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_embeddings(token_ids)

        batch, seq_len = token_ids.shape
        device = token_ids.device
        token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits



