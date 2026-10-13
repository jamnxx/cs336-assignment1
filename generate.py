import argparse
import torch
import pickle
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_id: int | None = None,
    device: str = "cuda",
):
    model.eval()
    ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        logits = model(ids)
        next_logits = logits[:, -1, :]

        temp = max(temperature, 1e-8)
        next_logits = next_logits / temp

        if top_p is not None and top_p < 1.0:
            probs = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)

            cum = sorted_probs.cumsum(dim=-1)
            mask = (cum-sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)

            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdims=True)

            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_id = sorted_idx.gather(-1, sampled)
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

        ids = torch.cat([ids, next_id], dim=-1)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--merges", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=1344)
    ap.add_argument("--rope_theta", type=float, default=10000.0)

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #tok = Tokenizer.from_files(
    #    args.vocab, args.merges,
    #    special_tokens=["<|endoftext|>"],
    #)
    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    prompt_ids = torch.tensor(
        [tok.encode(args.prompt)], dtype=torch.long
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eos_id = tok.encode("<|endoftext|>")[0]

    out_ids = generate(
        model, prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
        device=device,
    )

    text = tok.decode(out_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()








