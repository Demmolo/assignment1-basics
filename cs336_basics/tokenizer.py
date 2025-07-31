from collections import defaultdict
import json
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


    def tokenize(self, text: str, verbose=False):
        tokens: dict[tuple[bytes], int] = defaultdict(int)

        split_pattern = '|'.join(map(re.escape, self.special_tokens))
        
        print("Pretokenizing...")
        
        for chunk in re.split(split_pattern, text):
            # Pretokenize
            for token in re.finditer(PAT, chunk):
                tokens[tuple([bytes([t]) for t in token.group(0).encode('utf-8')]) ] += 1

        print("Pretokenization done")

        # print([(bytes(k).decode('utf8'), v) for k, v in tokens.items()])

        ##### TESTING
        # tokens = {"low".encode('utf-8'): 5, "lower".encode('utf-8'): 2, "widest".encode('utf-8'): 3, "newest".encode('utf-8'): 6}
        
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
    with open("data/TinyStoriesV2-GPT4-train.txt") as data:
        vocab, merges = tokenizer.tokenize(data.read(), verbose=True)

    tokenizer.serialize_vocabulary("tiny_stories_vocab.json")

if __name__ == "__main__":

    tokenize_tiny_stories()

    # tokenizer = Tokenizer(vocab_length=1000)
    # with open("data/TinyStoriesV2-GPT4-valid.txt") as data:
    #     vocab, merges = tokenizer.tokenize(data.read(), verbose=True)

    # tokenizer.serialize_vocabulary()
    # print(merges)