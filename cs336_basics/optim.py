import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta params at index 0:{betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta params at index 0:{betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay val: {weight_decay}")

        defaults = {
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                step = state["step"]
                step += 1
                #state["step"] = step

                m.mul_(beta1).add_(grad, alpha=1.0-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0-beta2)

                b1 = 1.0-beta1**step
                b2 = 1.0-beta2**step
                step_size = lr * math.sqrt(b2)/b1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters:int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if it <= cosine_cycle_iters:
        ratio = (it-warmup_iters) / (cosine_cycle_iters-warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * ratio))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def clip_grad_norm(parameters, max_norm:float, eps:float=1e-6) -> float:
    parameters = list(parameters)
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.norm(2).item()**2
    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return  total_norm










