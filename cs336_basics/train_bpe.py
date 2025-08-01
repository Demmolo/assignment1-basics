from concurrent.futures import ProcessPoolExecutor
import json
from typing import BinaryIO
import regex as re
from collections import defaultdict
import os
import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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



def process_chunk(file_path, start, end, split_pattern: str) -> dict[tuple[bytes], int]:
    with open(file_path, "rb") as file:
        file.seek(start)
        data = file.read(end - start).decode("utf-8", errors="ignore")

    tokens = defaultdict(int)
    for chunk in re.split(split_pattern, data):
        for token in re.finditer(PAT, chunk):
            byte_tuple = tuple([bytes([t]) for t in token.group(0).encode("utf-8")])
            tokens[byte_tuple] += 1
    return tokens


def pretokenize(file_path, special_tokens, parallelize):
    tokens: dict[tuple[bytes], int] = defaultdict(int)
    split_pattern = '|'.join(map(re.escape, special_tokens))
    
    if parallelize:
        # Assume file is a BinaryIO
        with open(file_path, "rb") as file:
            file.seek(0)
            boundaries = find_chunk_boundaries(file, desired_num_chunks=os.cpu_count() * 8, split_special_token=special_tokens[0].encode("utf-8"))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                # file.seek(start)
                # chunk_data = file.read(end - start).decode("utf-8", errors="ignore")

                futures.append(executor.submit(process_chunk, file_path, start, end, split_pattern))

            for fut in tqdm.tqdm(futures):
                chunk_tokens = fut.result()
                for k, v in chunk_tokens.items():
                    tokens[k] += v
    else:
        with open(file_path, "rb") as file:
            for chunk in re.split(split_pattern, file.read().decode("utf-8", errors="ignore")):
                for token in re.finditer(PAT, chunk):
                    tokens[tuple([bytes([t]) for t in token.group(0).encode("utf-8")])] += 1
    
    return tokens

def train_bpe(file_path: str, vocab_length=512, special_tokens=["<|endoftext|>"], parallelize=True, verbose=False):

    vocabulary = {i : bytes([i]) for i in range(256)}

    def add_to_vocabulary(token: bytes):
        try:
            assert(token not in vocabulary.keys())
        except AssertionError:
            print(token)
            raise SystemExit
        vocabulary[len(vocabulary)] = token

    for token in special_tokens:
        add_to_vocabulary(token.encode('utf-8'))

    print("Pretokenizing...")
    tokens = pretokenize(file_path, special_tokens, parallelize)
    print("Pretokenization done")
    
    merges = []

    # Initialize pair counts
    counts: dict[tuple[bytes], int] = defaultdict(int)

    for t, c in tokens.items():
        for i in range(len(t) - 1):
            counts[(t[i], t[i + 1])] += c


    print("Initialization done")
    while len(vocabulary) < vocab_length:
        if verbose:
            print(f"Vocab size: {len(vocabulary)} (token_pairs: {len(tokens)}, counts: {len(counts)})")
        
        
        # Get maximum count pair
        max_val = max(counts.values())
        max_pair = max(k for k, v in counts.items() if v == max_val )

        # print([(bytes(k).decode('utf8'), counts[k]) for k in sorted(counts, key=counts.get)])
    
        merges.append(max_pair)

        counts.pop(max_pair)

        max_pair_fused = b''.join(max_pair)
        add_to_vocabulary(max_pair_fused)
        # print(max_pair_fused)

        # Update token pair and count dicts
        tokens_new = {}
        for k, v in tokens.items():
            new = []
            i = 0
            len_k = len(k)
            while i < len_k:
                # Try to match a pair
                if i < len_k - 1 and (k[i], k[i+1]) == max_pair:
                    fused = b''.join((k[i], k[i+1]))
                    new.append(fused)

                    # Update prev pair count
                    if i > 0:
                        prev = (k[i-1], k[i])
                        counts[prev] -= v
                        new_prev = (k[i-1], fused)
                        counts[new_prev] += v

                    # Update next pair count (will be created in next iteration if i+2 exists)
                    if i < len_k - 2:
                        next_ = (k[i+1], k[i+2])
                        counts[next_] -= v
                        new_next = (fused, k[i+2])
                        counts[new_next] += v

                    i += 2  # skip the merged pair
                else:
                    new.append(k[i])
                    i += 1

            if len(new) > 1:
                tokens_new[tuple(new)] = v
            
            # if max_pair_fused == b'ning' and k != tuple(new):
            #     print(max_pair_fused, k, tuple(new))
        
        tokens = tokens_new

    return vocabulary, merges

def serialize_vocabulary(vocabulary, file_path="vocab.json"):
    # Convert {b'hello': 42} → {'hello': 42}
    vocab_serializable = {int(k): v.decode("utf-8", errors="replace") for k, v in vocabulary.items()}

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

def serialize_merges(merges: list[tuple[bytes, bytes]], file_path="merges.json"):
    # Convert {b'hello': 42} → {'hello': 42}
    merges_serializable = [ (m[0].decode("utf-8", errors="replace"), m[1].decode("utf-8", errors="replace")) for m in merges]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)