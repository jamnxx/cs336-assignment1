#### IMPLEMENTATION DETAILS

---

```markdown
# CS336 Assignment 1: Building a Transformer Language Model from Scratch

My implementation of Assignment 1 from
[Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/).

Everything is built **from scratch** in PyTorch, without using `torch.nn.Linear`,
`torch.nn.Embedding`, `torch.nn.functional`, or `torch.optim` (except for the
allowed base classes `nn.Module`, `nn.Parameter`, `nn.ModuleList`, and
`torch.optim.Optimizer`).

## Setup

Tested with Python 3.12 and PyTorch 2.7.1+cu118 on an NVIDIA RTX 3080.

```bash
# 1. Create and activate the environment
conda create -n cs336 python=3.12 -y
conda activate cs336

# 2. Install PyTorch with CUDA support (pick the one matching your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Or for CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121

# 3. Install the project and its dependencies from pyproject.toml
pip install -e .
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.7.1+cu118 True
```

> **Note**: `pyproject.toml` pins `torch~=2.11.0`, but cu118 wheels for that
> version aren't published yet. Installing torch *before* `pip install -e .`
> keeps the cu118 build in place — pip will only print a harmless version
> warning. The code works fine with 2.7.1+cu118.

---

## Implementation Order

I built the assignment bottom-up. Here's the order and rationale for each file.

### 1. `cs336_basics/bpe.py` — BPE Tokenizer Training

The foundation. Before any modeling, text needs to become integers.

- Byte-level BPE with GPT-2 regex pre-tokenization (`regex` library, `finditer`
  to avoid loading all matches at once).
- Special tokens (`<|endoftext|>`) are treated as **hard boundaries**: they
  never contribute to merge statistics, and no merge crosses their span.
- **Incremental pair-count updates**: after each merge, only the pairs
  overlapping the merge site are updated, rather than rescanning the corpus.
  This was the difference between 141 s and <1.5 s on the speed test.
- Multi-process pre-tokenization across document chunks for further speedup.

### 2. `cs336_basics/tokenizer.py` — Tokenizer Class

Uses the trained vocab + merges to encode/decode text.

- `encode`: pre-tokenize → apply merges **in order of creation** → map bytes
  to IDs via a reverse lookup dict.
- `decode`: look up IDs → concatenate bytes → `.decode("utf-8", errors="replace")`.
- `encode_iterable`: lazy generator for streaming large files without loading
  them into memory (constant memory footprint).

### 3. `cs336_basics/layers.py` — Basic Building Blocks

The primitives every other file depends on.

- `Linear` (no bias, truncated-normal init with `std = sqrt(2/(d_in+d_out))`).
- `Embedding` (truncated-normal init, `std = 1.0`).
- `RMSNorm` (with **float32 upcast** to prevent fp16 overflow when squaring).
- `silu`, `softmax`, `cross_entropy` (all numerically stable, max-subtraction).

### 4. `cs336_basics/attention.py` — Attention Mechanisms

- `scaled_dot_product_attention`: `softmax(QK^T/√d_k + mask) V`. Supports
  arbitrary batch dimensions; mask is `True` = keep, `False` = −∞.
- `RotaryPositionalEmbedding`: precomputes `cos`/`sin` tables as non-persistent
  buffers, then applies `x*cos + rotate_half(x)*sin`. The dot product of a
  rotated query and key depends only on their *relative* position.
- `MultiHeadSelfAttention`: single batched QKV projection → split heads with
  `einops.rearrange` → RoPE on Q,K only → causal mask → SDPA → merge heads →
  output projection.

### 5. `cs336_basics/model.py` — Full Transformer

- `SwiGLU`: `w2(silu(w1(x)) * w3(x))`, `d_ff ≈ 8/3 * d_model` rounded to 64.
- `TransformerBlock`: pre-norm residual, `z = x + MHA(RMSNorm(x))`,
  `y = z + FFN(RMSNorm(z))`.
- `TransformerLM`: token embedding → N blocks → final RMSNorm → LM head.
  Weights are named to match the reference state dict (`layers.*`,
  `output_proj`, `gain` → `weight`) so `load_state_dict` works without
  manual remapping.

### 6. `cs336_basics/optim.py` — Optimizer & Schedulers

- `AdamW`: decoupled weight decay (`θ ← θ − lr·λ·θ` before the gradient step),
  bias-corrected moments following the algorithm in Loshchilov & Hutter 2019.
- `get_lr_cosine_schedule`: linear warmup → cosine annealing → floor at `lr_min`.
- `clip_grad_norm_`: scales gradients in-place so the global L2 norm ≤ `max_norm`.

### 7. `cs336_basics/train_utils.py` — Training Utilities

- `get_batch`: random start indices → slice `context_length` tokens for inputs,
  offset-by-one for targets.
- `save_checkpoint` / `load_checkpoint`: persist model + optimizer state +
  iteration in a single `torch.save` dict.

### 8. `train.py` — End-to-End Training

Ties everything together: memmap'd dataset, model init, AdamW with parameter
groups (no weight decay on norms/biases), cosine LR schedule, gradient
clipping, periodic evaluation and checkpointing.

```bash
python train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000 \
    --max_iters 5000 \
    --batch_size 32 \
    --lr_max 3e-4 \
    --lr_min 3e-5 \
    --warmup_iters 500 \
    --cosine_cycle_iters 5000 \
    --eval_interval 100 \
    --save_interval 500 \
    --grad_clip 1.0
```

### 9. `generate.py` — Text Generation (Inference)

Loads a trained checkpoint and samples from the model autoregressively.

- **Prompt encoding**: the prompt is tokenized with the same BPE tokenizer used
  during training (vocab + merges must be loaded from disk).
- **Autoregressive loop**: feed the current sequence into the model, take the
  logits at the last position, apply temperature scaling, optionally apply
  top-p (nucleus) sampling, then sample the next token and append it.
- **Temperature scaling**: `softmax(logits / τ)`. Lower `τ` → sharper, more
  deterministic; higher `τ` → more diverse but noisier. `τ → 0` is argmax.
- **Top-p (nucleus) sampling**: sort probabilities descending, keep the
  smallest set whose cumulative mass ≥ `p`, renormalize, sample from that set.
  This cuts off the long tail of unlikely tokens.
- **Stopping criteria**: stop when `<|endoftext|>` is generated or when
  `max_new_tokens` is reached.

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_final.pt \
    --vocab_path data/tinystories_vocab.json \
    --merges_path data/tinystories_merges.txt \
    --prompt "Once upon a time" \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --device cuda
```

> **Tuning tips**:
> - If output is repetitive → lower `temperature` (e.g. 0.7) or lower `top_p` (e.g. 0.9).
> - If output is incoherent → raise `temperature` (e.g. 1.0) or raise `top_p` (e.g. 0.98).
> - For TinyStories (small model), `τ ∈ [0.7, 1.0]` and `p ∈ [0.9, 0.95]`
>   usually give the most fluent results.

---

## Project Structure

```
assignment1-basics/
├── cs336_basics/
│   ├── bpe.py               # (1) BPE training
│   ├── tokenizer.py         # (2) Tokenizer: encode / decode / encode_iterable
│   ├── layers.py            # (3) Linear, Embedding, RMSNorm, SiLU, softmax, cross_entropy
│   ├── attention.py         # (4) SDPA, RoPE, MultiHeadSelfAttention
│   ├── model.py             # (5) SwiGLU, TransformerBlock, TransformerLM
│   ├── optim.py             # (6) AdamW, LR schedule, gradient clipping
│   └── train_utils.py       # (7) get_batch, checkpoint save/load
├── tests/                   # Provided tests (unmodified)
├── adapters.py              # Glue code: tests → my implementation
├── train.py                 # (8) End-to-end training script
├── generate.py              # (9) Text generation / inference script
├── data/                    # Tokenized datasets (.npy, gitignored)
└── checkpoints/             # Saved checkpoints (gitignored)
```

## Tests

```bash
pytest tests/test_train_bpe.py       -v   # BPE
pytest tests/test_tokenizer.py       -v   # Tokenizer
pytest tests/test_model.py           -v   # All model components
pytest tests/test_nn_utils.py        -v   # softmax, cross_entropy, grad clip
pytest tests/test_optimizer.py       -v   # AdamW, LR schedule
pytest tests/test_data.py            -v   # get_batch
pytest tests/test_serialization.py   -v   # checkpointing
```

## Generation

See `generate.py` for a self-contained inference script. It loads a trained
checkpoint plus the tokenizer files, and samples text with temperature and
top-p controls.

## Notes & Lessons

- **RoPE is relative**: rotating Q and K by `R(pos)` makes their dot product
  depend only on `pos_q − pos_k`, which generalizes far better than learned
  absolute position embeddings.
- **Incremental merge updates** dominate BPE runtime. Caching pair counts and
  updating only the neighborhoods of each merge is the single biggest win.
- **RMSNorm upcast**: fp16/bf16 squares overflow fast; computing the norm in
  fp32 and downcasting back is required for stability.
- **AdamW `beta2`**: `(0.9, 0.95)` (LLaMA/GPT-3 convention) trains more stably
  than the default `(0.9, 0.999)` on small models.

## References

- [Stanford CS336](https://stanford-cs336.github.io/)
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Su et al., *RoFormer*, 2021. (RoPE)
- Zhang & Sennrich, *RMSNorm*, NeurIPS 2019.
- Shazeer, *GLU Variants Improve Transformer*, 2020.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016.

## License

MIT
