import numpy as np
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.layers import cross_entropy
from cs336_basics.optim import AdamW

torch.manual_seed(0)
device = "cuda"
train_data = np.load("data/train.npy", mmap_mode="r")

model = TransformerLM(
    vocab_size=10000, context_length=64, d_model=128,
    num_layers=2, num_heads=4, d_ff=256, device=device,
).to(device)

opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

# 固定一个 batch，反复过拟合
x = torch.from_numpy(np.array(train_data[:64])).long().unsqueeze(0).to(device)
y = torch.from_numpy(np.array(train_data[1:65])).long().unsqueeze(0).to(device)

print("x dtype:", x.dtype, "shape:", x.shape)
print("x[0, :10]:", x[0, :10].tolist())
print("y[0, :10]:", y[0, :10].tolist())
print("x unique:", torch.unique(x).tolist())
print("y unique:", torch.unique(y).tolist())
print("train_data[:10]:", np.array(train_data[:10]))
print("train_data.dtype:", train_data.dtype)

for it in range(200):
    logits = model(x)
    loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    # 打印第一个 Linear 和 lm_head 的梯度
    if it % 20 == 0:
        g_lm = model.lm_head.weight.grad.norm().item()
        g_emb = model.token_embeddings.weight.grad.norm().item()
        g_q0 = model.layers[0].attn.q_proj.weight.grad.norm().item()
        print(f"it={it} loss={loss.item():.4f} | g_lm={g_lm:.3e} g_emb={g_emb:.3e} g_q0={g_q0:.3e}")
    opt.step()