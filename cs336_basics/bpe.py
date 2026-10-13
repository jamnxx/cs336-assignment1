import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import multiprocessing as mp
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_and_split(filepath:str, special_tokens:List[str]) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    pattern = '|'.join([re.escape(t) for t in special_tokens])
    parts = re.split(pattern, text)
    return [p for p in parts if p != '']

def compute_word_counts(documents: List[str]) -> Dict[Tuple[int, ...],int]:
    word_counts = defaultdict(int)

    for doc in documents:
        for match in re.finditer(PAT, doc):
            word_bytes = match.group().encode('utf-8')
            word_counts[tuple(word_bytes)] += 1

    return dict(word_counts)

def compute_pair_counts(word_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int,int], int]:
    pair_counts = defaultdict(int)

    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple)-1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] += count

    return dict(pair_counts)

def apply_merge(word_counts: Dict[Tuple[int, ...],int],
                pair_counts: Dict[Tuple[int, int], int],
                pair_to_merge: Tuple[int, int],
                new_byte_id: int) -> Tuple[Dict[Tuple[int, ...], int],
                                     Dict[Tuple[int, int], int]]:
    new_word_counts = defaultdict(int)
    a, b = pair_to_merge

    for word_tuple, count in word_counts.items():
        if a not in word_tuple:
            new_word_counts[word_tuple] += count
            continue

        word_list = list(word_tuple)
        new_w = []
        i = 0
        while i < len(word_list):
            if i+1 < len(word_list) and word_list[i] == a and word_list[i+1] == b:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) - count
                if new_w:
                    left = new_w[-1]
                    pair_counts[(left, a)] =  pair_counts.get((left, a), 0)- count
                    #if(pair_counts[(left, a)] == 0):
                    #    del pair_counts[(left, a)]
                    pair_counts[(left, new_byte_id)] = pair_counts.get((left, new_byte_id),0) + count

                if i+2 < len(word_list):
                    right = word_list[i+2]
                    pair_counts[(b, right)] =  pair_counts.get((b, right), 0) - count
                    #if (pair_counts[(b, right)] == 0):
                    #    del pair_counts[(b, right)]
                    pair_counts[(new_byte_id, right)] = pair_counts.get((new_byte_id, right),0) + count

                new_w.append(new_byte_id)
                i+=2

            else:
                new_w.append(word_list[i])
                i+=1

        new_word_counts[tuple(new_w)] += count

    keys_to_del = [k for k, v in pair_counts.items() if v <= 0]
    for k in keys_to_del:
        del pair_counts[k]

    return new_word_counts, pair_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str])->Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_size_so_far = 256

    for token in special_tokens:
        vocab[vocab_size_so_far] = token.encode('utf-8')
        vocab_size_so_far += 1

    documents = read_and_split(input_path, special_tokens)
    num_workers = min(mp.cpu_count(), 8)

    chunk_size = max(1, len(documents) // num_workers)
    chunks = [documents[i:i+chunk_size] for i in range(0, len(documents), chunk_size)]

    # print(f"正在使用{num_workers}个进程进行分词...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(compute_word_counts, chunks)

    word_counts = defaultdict(int)
    for res in results:
        for word, count in res.items():
            word_counts[word] += count

    word_counts = dict(word_counts)
    pair_counts = compute_pair_counts(word_counts)

    merges = []
    while vocab_size_so_far < vocab_size and pair_counts:
        max_pair = max(pair_counts.keys(), key=lambda p:(pair_counts[p], vocab[p[0]],vocab[p[1]]))
        a, b = max_pair
        merges.append((vocab[a], vocab[b]))
        new_token = vocab[a]+vocab[b]
        vocab[vocab_size_so_far] = new_token
        vocab_size_so_far += 1

        word_counts, pair_counts = apply_merge(word_counts, pair_counts, max_pair, vocab_size_so_far-1)

    return vocab, merges

if __name__  == "__main__":
    mp.set_start_method("spawn", force=True)

    with open("text.txt", "w") as f:
        f.write("low low low lower widest newest")

    vocab, merges = train_bpe("text.txt",vocab_size=270, special_tokens=["<|endoftext|>"])
    print(f"vocab_size:{len(vocab)}")
    print(f"#merges:{len(merges)}")
    print(f"first 5 merges:{merges[:5]}")
