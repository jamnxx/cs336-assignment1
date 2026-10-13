from __future__ import annotations
import regex as re
from typing import Dict, List, Tuple, Iterable, Iterator, Optional

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self,
                 vocab: Dict[int, bytes],
                 merges: List[Tuple[bytes, bytes]],
                 special_tokens: Optional[List[str]] = None,) -> None:
        self.vocab: Dict[int, bytes] = vocab
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.special_tokens: List[str] = special_tokens or []

        self.bytes_to_id: Dict[bytes, int] = {b: i for i, b in vocab.items()}
        self.merge_rank: Dict[Tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}

        self.special_to_id: Dict[str, int] = {}
        for s in self.special_tokens:
            b = s.encode('utf-8')
            if b in self.bytes_to_id:
                self.special_to_id[s] = self.bytes_to_id[b]
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

    def decode(self, ids: List[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in ids)
        return text_bytes.decode('utf-8', errors="replace")

    def encode(self, text:str) -> List[int]:
        ids: List[int] = [ ]
        if self.special_tokens_sorted:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens_sorted) + ")"
            result = re.split(pattern, text)
        else:
            result = [text]

        for res in result:
            if not res:
                continue
            if res in self.special_to_id:
                ids.append(self.special_to_id[res])
            else:
                for match in re.finditer(PAT, res):
                    token_str = match.group()
                    token_bytes = token_str.encode('utf-8')
                    ids.extend(self.bpe_merge_to_token(token_bytes))

        return ids


    def bpe_merge_to_token(self, token_bytes: bytes) -> List[int]:
        parts = [bytes([b]) for b in token_bytes]
        if not parts:
            return []

        while len(parts) >= 2:
            best_rank = None
            best_i = None
            for i in range(len(parts)-1):
                pair = (parts[i], parts[i+1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_i = i
                    best_rank = rank

            if best_i is None:
                break

            merged = parts[best_i] + parts[best_i + 1]
            parts = parts[:best_i] + [merged] + parts[best_i+2:]

        return [self.bytes_to_id[p] for p in parts]

    def encode_iterable(self, iterable:Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid




