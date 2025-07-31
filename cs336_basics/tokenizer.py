from collections import defaultdict
import json
import regex as re
from concurrent.futures import ProcessPoolExecutor


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class Tokenizer:

    def __init__(self, vocab_length=512, special_tokens=["<|endoftext|>"]):
        self.vocabulary = {i : bytes([i]) for i in range(256)}

        for token in special_tokens:
            self.add_to_vocabulary(token.encode('utf-8'))

        self.vocab_length = vocab_length
        self.special_tokens = special_tokens

    def add_to_vocabulary(self, token: bytes):
        try:
            assert(token not in self.vocabulary.keys())
        except AssertionError:
            print(token)
            raise SystemExit
        self.vocabulary[len(self.vocabulary)] = token

    def process_chunk(data: bytes, split_pattern: str) -> dict[tuple[bytes], int]:
        tokens = defaultdict(int)
        for chunk in re.split(split_pattern, data):
            for token in re.finditer(PAT, chunk):
                byte_tuple = tuple([bytes([t]) for t in token.group(0).encode("utf-8")])
                tokens[byte_tuple] += 1
        return tokens

    def tokenize(self, file: BinaryIO, parallelize=True, verbose=False):
        tokens: dict[tuple[bytes], int] = defaultdict(int)

        split_pattern = '|'.join(map(re.escape, self.special_tokens))
        
        print("Pretokenizing...")

        if parallelize:
            # Assume file is a BinaryIO
            file.seek(0)
            boundaries = find_chunk_boundaries(file, desired_num_chunks=os.cpu_count(), split_special_token=self.special_tokens[0].encode("utf-8"))

            with ProcessPoolExecutor() as executor:
                futures = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    file.seek(start)
                    chunk_data = file.read(end - start).decode("utf-8", errors="ignore")

                    futures.append(executor.submit(Tokenizer.process_chunk, chunk_data, split_pattern))

                for fut in futures:
                    chunk_tokens = fut.result()
                    for k, v in chunk_tokens.items():
                        tokens[k] += v
        else:
            file.seek(0)
            for chunk in re.split(split_pattern, file.read().decode("utf-8", errors="ignore")):
                for token in re.finditer(PAT, chunk):
                    tokens[tuple([bytes([t]) for t in token.group(0).encode("utf-8")])] += 1

            # for chunk in re.split(split_pattern, text):
            #     # Pretokenize
            #     for token in re.finditer(PAT, chunk):
            #         tokens[tuple([bytes([t]) for t in token.group(0).encode('utf-8')]) ] += 1

        print("Pretokenization done")
        
        merges = []

        # Initialize pair counts
        counts: dict[tuple[bytes], int] = defaultdict(int)

        for t, c in tokens.items():
            for i in range(len(t) - 1):
                counts[(t[i], t[i + 1])] += c


        print("Initialization done")
        while len(self.vocabulary) < self.vocab_length:
            if verbose:
                print(f"Vocab size: {len(self.vocabulary)} (token_pairs: {len(tokens)}, counts: {len(counts)})")
            
            
            # Get maximum count pair
            max_val = max(counts.values())
            max_pair = max(k for k, v in counts.items() if v == max_val )

            # print([(bytes(k).decode('utf8'), counts[k]) for k in sorted(counts, key=counts.get)])
        
            merges.append(max_pair)

            counts.pop(max_pair)

            max_pair_fused = b''.join(max_pair)
            self.add_to_vocabulary(max_pair_fused)
            # print(max_pair_fused)

            # Update token pair and count dicts
            tokens_new = {}
            for k, v in tokens.items():
                new = []
                i = 0
                while i < len(k):
                    # Try to match a pair
                    if i < len(k) - 1 and (k[i], k[i+1]) == max_pair:
                        fused = b''.join((k[i], k[i+1]))
                        new.append(fused)

                        # Update prev pair count
                        if i > 0:
                            prev = (k[i-1], k[i])
                            counts[prev] -= v
                            new_prev = (k[i-1], fused)
                            counts[new_prev] += v

                        # Update next pair count (will be created in next iteration if i+2 exists)
                        if i < len(k) - 2:
                            next_ = (k[i+1], k[i+2])
                            counts[next_] -= v
                            new_next = (fused, k[i+2])
                            counts[new_next] += v

                        i += 2  # skip the merged pair
                    else:
                        new.append(k[i])
                        i += 1

                if new:
                    tokens_new[tuple(new)] = v
                
                # if max_pair_fused == b'ning' and k != tuple(new):
                #     print(max_pair_fused, k, tuple(new))
            
            tokens = tokens_new

        return self.vocabulary, merges
    
    def serialize_vocabulary(self, file_path="vocab.json"):
        # Convert {b'hello': 42} → {'hello': 42}
        vocab_serializable = {int(k): v.decode("utf-8", errors="replace") for k, v in self.vocabulary.items()}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)


def tokenize_tiny_stories():
    tokenizer = Tokenizer(vocab_length=10000)
    with open("data/TinyStoriesV2-GPT4-train.txt", "rb") as data:
        vocab, merges = tokenizer.tokenize(data.read(), verbose=True)

    tokenizer.serialize_vocabulary("tiny_stories_vocab.json")

def tokenize_owt():
    tokenizer = Tokenizer(vocab_length=32000)
    with open("data/owt_train.txt", "rb") as data:
        vocab, merges = tokenizer.tokenize(data.read(), verbose=True)

    tokenizer.serialize_vocabulary("tiny_stories_vocab.json")

if __name__ == "__main__":

    tokenize_tiny_stories()

    # tokenizer = Tokenizer(vocab_length=1000)
    # with open("data/TinyStoriesV2-GPT4-valid.txt") as data:
    #     vocab, merges = tokenizer.tokenize(data.read(), verbose=True)

    # tokenizer.serialize_vocabulary()
    # print(merges)