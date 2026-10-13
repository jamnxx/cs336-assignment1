# save_tokenizer.py
import pickle
import time
from cs336_basics.bpe import train_bpe

if __name__ == "__main__":
    train_txt = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print("Re-training BPE (deterministic, same as before)...")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=train_txt,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    print(f"Done in {time.time()-t0:.1f}s | vocab={len(vocab)} | merges={len(merges)}")

    with open("data/tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print("Saved data/tinystories_vocab.pkl and data/tinystories_merges.pkl")