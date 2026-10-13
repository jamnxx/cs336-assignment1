import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features+out_features)) ** 0.5
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        orid_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True)+self.eps)
        output = (x_float/rms) * self.weight.float()

        return output.to(orid_dtype)

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x-x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(logits, targets):
    max_logits = logits.max(dim=-1, keepdim=True)[0].detach()
    logits_shifted = logits-max_logits

    exp_logits = torch.exp(logits_shifted)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    log_softmax = logits_shifted-torch.log(sum_exp)

    target_log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = -target_log_probs.mean()
    return loss


