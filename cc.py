import numpy as np
import torch
from einops import rearrange
from cs336_basics.model import TransformerLM
from cs336_basics.layers import cross_entropy
from cs336_basics.optim import AdamW

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 小模型 ----
model = TransformerLM(
    vocab_size=10000, context_length=64, d_model=128,
    num_layers=2, num_heads=4, d_ff=256, device=device,
).to(device)

# ---- 固定 batch（真数据）----
train_data = np.load("data/train.npy", mmap_mode="r")
seq = np.array(train_data[:64], dtype=np.int64)
x = torch.from_numpy(seq).long().unsqueeze(0).to(device)
y = torch.from_numpy(np.array(train_data[1:65], dtype=np.int64)).long().unsqueeze(0).to(device)

# ---- 手动前向，打印每层统计 ----
def stats(t, name):
    print(f"  {name:20s} shape={tuple(t.shape)} mean={t.mean().item():+.4e} "
          f"std={t.std().item():.4e} absmax={t.abs().max().item():.4e}")

print("=" * 70)
print("STEP 1: 前向传播逐层统计")
print("=" * 70)

with torch.no_grad():
    h = model.token_embeddings(x)
    stats(h, "token_emb out")

    for i, layer in enumerate(model.layers):
        # attention 前的 norm
        n1 = layer.ln1(h)
        stats(n1, f"layer{i} ln1")

        # 手算 attention 的中间量
        Q = layer.attn.q_proj(n1)
        K = layer.attn.k_proj(n1)
        V = layer.attn.v_proj(n1)
        stats(Q, f"layer{i} Q (pre-rope)")
        stats(K, f"layer{i} K (pre-rope)")

        # 多头
        B, S, D = Q.shape
        H = layer.attn.num_heads
        dk = D // H
        Qh = rearrange(Q, 'b s (h d) -> b h s d', h=H)
        Kh = rearrange(K, 'b s (h d) -> b h s d', h=H)
        pos = torch.arange(S, device=device).unsqueeze(0)
        Qr = layer.attn.rope(Qh, pos)
        Kr = layer.attn.rope(Kh, pos)
        stats(Qr, f"layer{i} Q (post-rope)")
        stats(Kr, f"layer{i} K (post-rope)")

        # attention scores
        scores = Qr @ Kr.transpose(-2, -1) / (dk ** 0.5)
        stats(scores, f"layer{i} scores")
        mask = layer.attn.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        stats(attn, f"layer{i} attn_w")

        # attention 输出
        Vh = rearrange(V, 'b s (h d) -> b h s d', h=H)
        attn_out = attn @ Vh
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        stats(attn_out, f"layer{i} attn_out")
        attn_out = layer.attn.output_proj(attn_out)
        stats(attn_out, f"layer{i} attn_proj")

        h = h + attn_out
        stats(h, f"layer{i} after_attn_res")

        # FFN
        n2 = layer.ln2(h)
        stats(n2, f"layer{i} ln2")
        ffn_out = layer.ffn(n2)
        stats(ffn_out, f"layer{i} ffn_out")
        h = h + ffn_out
        stats(h, f"layer{i} after_ffn_res")

    h = model.ln_final(h)
    stats(h, "final_norm")
    logits = model.lm_head(h)
    stats(logits, "logits")

# ---- 反向传播，打印每层梯度 ----
print()
print("=" * 70)
print("STEP 2: 反向传播逐层梯度范数")
print("=" * 70)

model.zero_grad()
logits = model(x)
loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
loss.backward()

for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"  {name:45s} grad_norm={p.grad.norm().item():.4e}")

# ---- 优化器 step，打印参数变化 ----
print()
print("=" * 70)
print("STEP 3: 参数更新量（optimizer.step 前后差值）")
print("=" * 70)

opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

# 记录关键参数
keys = ['lm_head.weight', 'token_embeddings.weight',
        'layers.0.attn.q_proj.weight', 'layers.0.attn.output_proj.weight',
        'layers.0.ffn.w1.weight', 'layers.0.ln1.weight']
snap_before = {k: dict(model.named_parameters())[k].detach().clone() for k in keys}

opt.step()

for k in keys:
    p = dict(model.named_parameters())[k].detach()
    delta = (p - snap_before[k]).abs()
    print(f"  {k:45s} Δmax={delta.max().item():.4e} Δmean={delta.mean().item():.4e}")

# ---- 连续 10 步，看 loss 是否变化 ----
print()
print("=" * 70)
print("STEP 4: 连续 10 步 loss 变化")
print("=" * 70)

for step in range(10):
    logits = model(x)
    loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"  step {step}: loss={loss.item():.6f}")