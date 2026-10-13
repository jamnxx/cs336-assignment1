from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
import os
from tqdm import tqdm
import numpy as np

def encode_file_to_npy(txt_path: str, npy_path: str, tokenizer: Tokenizer):
    print(f"开始分词: {txt_path} -> {npy_path}")
    all_ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Encoding"):
            if not line.strip():
                continue
            ids = tokenizer.encode(line)
            all_ids.extend(ids)

    ids_arr = np.array(all_ids, dtype=np.uint16)
    np.save(npy_path, ids_arr)
    print(f"save {len(ids_arr)} tokens to {npy_path}\n")

if __name__ == "__main__":
    train_txt = "data/TinyStoriesV2-GPT4-train.txt"
    val_txt = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print("training BPE...")
    vocab, merges = train_bpe(input_path=train_txt, vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    print("BPE finished!")

    encode_file_to_npy(train_txt, "data/train.npy", tokenizer)
    if os.path.exists(val_txt):
        encode_file_to_npy(val_txt, "data/val.npy", tokenizer)