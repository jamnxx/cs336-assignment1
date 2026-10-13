import argparse
import os
import json
import time
import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, clip_grad_norm
from cs336_basics.train_utils import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.layers import cross_entropy

def parse_args():
    p = argparse.ArgumentParser()
    #data
    p.add_argument("--train_data", type=str, default="data/train.py")
    p.add_argument("--val_data", type=str, default="data/val.npy")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    #model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    #train
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_iters", type=int, default=10000)
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=500)
    p.add_argument("--cosine_cycle_iters", type=int, default=10000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    #logs
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--run_name", type=str, default="default")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()

def get_optimizer_param_groups(model: torch.nn.Module, weight_decay:float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, eval_iters=50):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

def train():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log = {
        "args": vars(args),
        "iter": [], "wall_time": [], "lr": [], "train_loss": [],
        "val_iter": [], "val_loss": [], "val_wall_time": [],
    }
    t_start = time.time()

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
    ).to(args.device)

    param_groups = get_optimizer_param_groups(model, args.weight_decay)
    optimizer = AdamW(
        param_groups,
        lr=args.lr_max,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at iter {start_iter}")

    model.train()
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            it,
            args.lr_max,
            args.lr_min,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        log["iter"].append(it)
        log["wall_time"].append(time.time() - t_start)
        log["lr"].append(lr)
        log["train_loss"].append(loss.item())
        # ================================================
        if it % 50 == 0:
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            logits_std = logits.std().item()
            print(f"  [debug] grad_norm={grad_norm:.4f}, logits_std={logits_std:.4f}, loss={loss.item():.4f}")

        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        if it > 0 and it % args.eval_interval == 0:
            #val_data = np.load(args.val_data, mmap_mode="r")
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device)
            print(f" >>val loss {val_loss:.4f}")
            log["val_iter"].append(it)
            log["val_loss"].append(val_loss)
            log["val_wall_time"].append(time.time() - t_start)

        if it > 0 and it % args.save_interval == 0:
            path = os.path.join(args.out_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, path)
            print(f" >> saved {path}")

    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    log_path = os.path.join(args.log_dir, f"{args.run_name}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {log_path}")
    print("Training finished!")

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    train()

    






