from collections import defaultdict
import json
import regex as re
import tqdm
from concurrent.futures import ProcessPoolExecutor



import os
from typing import BinaryIO, Iterable, Iterator


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def vocabulary_from_file(file_path) -> dict[int, bytes]:
    with open(file_path, "r", encoding="utf-8") as f:
        vocab_serializable = json.load(f)

    # Convert {'hello': '42'} → {42: b'hello':}
    return { int(v) : k.encode("utf-8") for k, v in vocab_serializable.items()}


def merges_from_file(file_path="merges.json") -> list[tuple[bytes, bytes]]:
    merges = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip blank lines and comments
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid merge line: {line}")
            left, right = parts
            merges.append((left.encode("utf-8"), right.encode("utf-8")))
    return merges

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        
        # Hack: flip vocab
        self.vocabulary = { v: k for k, v in vocab.items() }

        # print(list(self.vocabulary.items())[0])
        
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = vocabulary_from_file(vocab_filepath)
        merges = merges_from_file(merges_filepath)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:

        if self.special_tokens:
            split_pattern = f"({'|'.join(map(re.escape, self.special_tokens))})"
            chunks = re.split(split_pattern, text)
        else:
            chunks = [text]

        
        merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        output_ids = []

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                output_ids.append(self.vocabulary[chunk.encode()])
            else:

                for token in re.finditer(PAT, text):
                    token = token.group(0)

                    # Start with individual bytes
                    symbols = [bytes([b]) for b in token.encode("utf-8")]

                    # Build initial pairs
                    while True:
                        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)]
                        ranked = [(merge_ranks[p], i) for i, p in enumerate(pairs) if p in merge_ranks]

                        if not ranked:
                            break

                        # Find highest priority merge
                        _, idx = min(ranked)
                        a, b = symbols[idx], symbols[idx+1]
                        symbols = symbols[:idx] + [a + b] + symbols[idx+2:]

                    # Map merged symbols to vocab IDs
                    for sym in symbols:
                        if sym not in self.vocabulary:
                            print(self.vocabulary)
                            raise ValueError(f"Symbol {sym} not in vocabulary")
                            
                        output_ids.append(self.vocabulary[sym])

        return output_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        
        # flip vocab
        vocab_flipped = { v: k for k, v in self.vocabulary.items() }

        return b"".join([vocab_flipped[idx] for idx in ids]).decode("utf-8", errors="replace")