import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import Linear  # 你刚写的 Linear


# ============================================================
# 1. Scaled Dot-Product Attention（无状态函数）
# ============================================================
def scaled_dot_product_attention(
        Q: torch.Tensor,  # (..., seq_len_q, d_k)
        K: torch.Tensor,  # (..., seq_len_k, d_k)
        V: torch.Tensor,  # (..., seq_len_k, d_v)
        mask: torch.Tensor | None = None,  # (..., seq_len_q, seq_len_k) 或 broadcastable
) -> torch.Tensor:  # (..., seq_len_q, d_v)
    """
        实现公式: softmax(Q @ K.T / sqrt(d_k)) @ V
        """
    # TODO 1.1: 获取 d_k（Q 的最后一维）
    d_k = Q.shape[-1]
    # TODO 1.2: 计算 Q @ K^T，得到 scores，形状 (..., seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # TODO 1.3: 除以 sqrt(d_k) 进行缩放
    scores = scores / (d_k ** 0.5)
    # TODO 1.4: 如果 mask 不为 None，将 mask 中为 False 的位置设为 -inf
    #           提示：用 masked_fill 或 torch.where
    #           注意：mask 的维度需要和 scores 对齐（可能要用 unsqueeze）
    if mask is not None:
            #if mask.dim() == 2:
            #        mask = mask.unsqueeze(0).unsqueeze(0)
            #elif mask.dim() == 3:
            #    mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
    # TODO 1.5: 对最后一维做 softmax，得到注意力权重
    atten_weights = torch.softmax(scores, dim=-1)
    # TODO 1.6: 注意力权重 @ V，返回结果
    output = torch.matmul(atten_weights, V)
    return output



# ============================================================
# 2. Rotary Positional Embedding（RoPE）
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,  # 默认 10000.0
            d_k: int,  # 每个头的维度
            max_seq_len: int,  # 最大序列长度（预计算缓存用）
            device: torch.device | None = None,
    ):
        super().__init__()
        # TODO 2.1: 计算 inv_freq（逆频率）
        #           公式: inv_freq[i] = 1 / (theta ** (2*i / d_k))
        #           其中 i 从 0 到 d_k//2 - 1
        #           形状: (d_k // 2,)
        i = torch.arange(d_k // 2, dtype=torch.float, device=device)
        inv_freq = 1 / (theta ** (2*i / d_k))
        # TODO 2.2: 计算所有位置的频率
        #           位置序列: [0, 1, 2, ..., max_seq_len-1]
        #           freqs = positions.outer(inv_freq)   # (max_seq_len, d_k//2)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        freqs_interleaved = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        # TODO 2.3: 把 freqs 复制成完整的 cos 和 sin 表
        #           因为每个维度要成对旋转，所以要把 freqs 扩展成 (max_seq_len, d_k)
        #           即: cos_table 和 sin_table，形状 (max_seq_len, d_k)
        #           cos_table 和 sin_table 应该是成对出现的：前半维和后半维同一频率
        #cos_table = torch.cos(torch.cat([freqs, freqs], dim=-1))
        #sin_table = torch.sin(torch.cat([freqs, freqs], dim=-1))
        cos_table = torch.cos(freqs_interleaved)
        sin_table = torch.sin(freqs_interleaved)

        # TODO 2.4: 用 self.register_buffer 注册 cos_table 和 sin_table
        #           注意: persistent=False（不是模型参数，不需要保存到 state_dict）
        self.register_buffer('cos_table', cos_table, persistent=False)
        self.register_buffer('sin_table', sin_table, persistent=False)


    def forward(
            self,
            x: torch.Tensor,  # (..., seq_len, d_k)
            token_positions: torch.Tensor | None = None,  # (..., seq_len) 或 (seq_len,)
    ) -> torch.Tensor:  # (..., seq_len, d_k)
        """
                对 x 应用 RoPE 旋转
                """
        # TODO 2.5: 如果 token_positions 为 None，使用默认位置 0, 1, 2, ..., seq_len-1
        #           提示: torch.arange(seq_len, device=x.device)
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        # TODO 2.6: 根据 token_positions 从 cos_table 和 sin_table 中查表
        #           提示: cos_table[token_positions]  # 注意形状对齐
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        ########################################
        if x.ndim == 4 and cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # TODO 2.7: 实现 rotate_half 操作
        #           把 x 分成前半和后半: x1 = x[..., :d_k//2], x2 = x[..., d_k//2:]
        #           rotate_half = [-x2, x1] (在最后一维拼接)
        #           即: torch.cat([-x2, x1], dim=-1)
        #x1, x2 = x.chunk(2, dim=-1)
        #x_rotated_half = torch.cat([-x2, x1], dim=-1)
        shape = x.shape
        x_reshaped = x.view(*shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        x_rotated_half = torch.stack([-x2, x1], dim=-1).view(shape)

        # TODO 2.8: 应用旋转公式
        #           x_rot = x * cos + rotate_half(x) * sin
        #           返回 x_rot
        return x*cos + x_rotated_half*sin


# ============================================================
# 3. Multi-Head Self-Attention（完整层）
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int,
            theta: float = 10000.0,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads

        # TODO 3.1: 计算 d_k = d_model // num_heads
        #           注意: d_model 必须能被 num_heads 整除
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        # TODO 3.2: 初始化 Q, K, V 投影（用你写的 Linear）
        #           q_proj: d_model -> d_model
        #           k_proj: d_model -> d_model
        #           v_proj: d_model -> d_model
        #           传入 device 和 dtype
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # TODO 3.3: 初始化输出投影（用你写的 Linear）
        #           o_proj: d_model -> d_model
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # TODO 3.4: 初始化 RoPE
        #           传入 theta, d_k, max_seq_len, device

        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)


        # TODO 3.5: 注册 causal mask（下三角掩码）
        #           形状: (max_seq_len, max_seq_len)
        #           用 torch.tril 生成，dtype=torch.bool
        #           用 register_buffer 注册，persistent=False
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(
            self,
            x: torch.Tensor,  # (batch, seq_len, d_model)
            token_positions: torch.Tensor | None = None,  # (batch, seq_len)
    ) -> torch.Tensor:  # (batch, seq_len, d_model)
        """
                因果多头自注意力 + RoPE
                """
        # TODO 3.6: 获取 batch, seq_len
        batch, seq_len, _ = x.shape

        # TODO 3.7: Q, K, V 投影
        #           Q, K, V 形状: (batch, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # TODO 3.8: 拆分成多头
        #           用 rearrange: 'b seq (heads d_k) -> b heads seq d_k'
        #           传入 heads=num_heads
        #           现在 Q, K, V 都是 (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, 'b seq (heads d_k) -> b heads seq d_k', heads=self.num_heads)
        K = rearrange(K, 'b seq (heads d_k) -> b heads seq d_k', heads=self.num_heads)
        V = rearrange(V, 'b seq (heads d_k) -> b heads seq d_k', heads=self.num_heads)

        if token_positions is not None:
            # TODO 3.9: 对 Q 和 K 应用 RoPE（V 不应用）
            #           注意: token_positions 的形状是 (batch, seq_len)
            #           RoPE 支持 batch 维作为前缀
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)


        # TODO 3.10: 准备 causal mask
        #           从 self.causal_mask 中取前 seq_len 个
        #           形状: (seq_len, seq_len)
        #           需要扩展到 (batch, num_heads, seq_len, seq_len)
        #           提示: 用 unsqueeze(0).unsqueeze(0) 或 expand
        #           注意: mask=True 表示"保留"，False 表示"掩码掉"
        #                在 SDPA 中，False 的位置会被设为 -inf
        mask = self.causal_mask[:seq_len, :seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, -1, -1)

        # TODO 3.11: 调用 scaled_dot_product_attention
        #           传入 Q, K, V, mask
        #           输出: (batch, num_heads, seq_len, d_k)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # TODO 3.12: 合并多头
        #           用 rearrange: 'b heads seq d_k -> b seq (heads d_k)'
        #           形状: (batch, seq_len, d_model)
        attn_output = rearrange(attn_output, 'b heads seq d_k -> b seq (heads d_k)')
        # TODO 3.13: 输出投影
        #           返回 o_proj(merged)
        return self.output_proj(attn_output)
