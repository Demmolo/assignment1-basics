from cs336_basics.train_bpe import train_bpe, serialize_merges, serialize_vocabulary

def tokenize_tiny_stories():
    file_path = "data/TinyStoriesV2-GPT4-train.txt"

    vocab, merges = train_bpe(file_path, verbose=True)

    serialize_vocabulary(vocab, "tiny_stories_vocab.json")
    serialize_merges(merges, "tiny_stories_merges.json")

def tokenize_owt():
    file_path = "data/owt_train.txt"
    
    vocab, merges = train_bpe(file_path, verbose=True)

    serialize_vocabulary(vocab, "owt_vocab.json")
    serialize_merges(merges, "owt_merges.json")

if __name__ == "__main__":

    # tokenize_tiny_stories()
    # tokenize_owt()

    f = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(f, verbose=True, parallelize=False)

    serialize_vocabulary(vocab, "tiny_stories_vocab.json")
    serialize_merges(merges, "tiny_stories_merges.json")