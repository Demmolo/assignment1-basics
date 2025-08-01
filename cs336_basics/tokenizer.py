from collections import defaultdict
import json
import regex as re
import tqdm
from concurrent.futures import ProcessPoolExecutor



import os
from typing import BinaryIO, Iterable, Iterator



class Tokenizer:

    def __init__(self, vocab_length=512, special_tokens=["<|endoftext|>"]):


        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass


