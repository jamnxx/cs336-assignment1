import torch
import numpy as np
import os
from typing import BinaryIO, IO

def get_batch(
        dataset: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=(batch_size,))

    inputs = np.array([dataset[start:start + context_length] for start in starts])
    targets = np.array([dataset[start+1:start + context_length + 1] for start in starts])

    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)

    return inputs, targets

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]


