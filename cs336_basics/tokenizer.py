import regex as re
import os
from collections import Counter, defaultdict
from cs336_basics.utils.io import GPT2_PRE_TOKEN_PATTERN, WHITESPACE_PATTERN, get_tokenizer_from_vocab_merges_path, gpt2_bytes_to_unicode
import logging
from tqdm import tqdm
import time
from typing import Iterable, Iterator, Tuple, List, Dict
import tiktoken
import json
logging.basicConfig(level=logging.INFO)

def get_pairs(ids: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """ Return a set of pairs in int ids """
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs

def update(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """ Update the ids by merging the pairs """
    new_ids = []
    i = 0
    while i < len(ids):
        curr_pair = tuple(ids[i:i+2])
        if curr_pair == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def save_voacb_and_merge(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
                            vocab_path: str, merges_path: str):
    byte_to_unicode = gpt2_bytes_to_unicode()

    # Reverse the mapping from unicode characters to bytes
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}
    
    # Convert the byte tokens in the vocab back to string tokens using the unicode mapping
    reversed_vocab = {''.join([byte_to_unicode[b] for b in bytes_token]):k
                      for k, bytes_token in vocab.items()}

    # Convert the byte sequences in merges back to string tokens
    reversed_merges = [' '.join([''.join([byte_to_unicode[b] for b in merge[0]]),
                                 ''.join([byte_to_unicode[b] for b in merge[1]])])
                       for merge in merges]

    # Save the vocab dictionary as a JSON file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(reversed_vocab, f, ensure_ascii=False)
        logging.info(f"Vocab saved to {vocab_path}")
    
    # Save the merges list to a file
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in reversed_merges:
            f.write(merge + '\n')
        logging.info(f"Merges saved to {merges_path}")

def train_bpe(input_path: str | os.PathLike, 
              vocab_size: int, 
              special_tokens: list[str], 
              is_save: bool = False, 
              output_dir: str | os.PathLike = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file and save the vocab and merges to the output directory if is_save is True.
    
    Args:
        input_path: str | os.PathLike
            Path to the input file.
        vocab_size: int
            The size of the vocabulary.
        special_tokens: list[str]
            The special tokens to add to the vocabulary.
        is_save: bool
            Whether to save the vocab and merges to the output directory.
        output_dir: str | os.PathLike
            The output directory.
    Returns:
        vocab: dict[int, bytes]
            The vocabulary.
        merges: list[tuple[bytes, bytes]]
            The merges.
    """
    vocab = {i: bytes([i]) for i in range(256)} # token_id -> token_bytes
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    merges = [] # list of (token1_bytes, token2_bytes)
    # pre-tokenize input file
    pat = GPT2_PRE_TOKEN_PATTERN
    start_time = time.time()
    with open(input_path, "r") as f:
        text = f.read()
    end_time = time.time()
    logging.info(f"File reading took {end_time - start_time:.2f} seconds")
    # remove special tokens
    for special_token in special_tokens:
        text = text.replace(special_token, "")
    all_tokens = re.findall(pat, text)
    all_token_counters = Counter(all_tokens)
    pre_token_freq = {}
    start_time = time.time()
    for token, freq in tqdm(all_token_counters.items()):
        # token = 'iron'
        # tuple of bytes: (b'i', b'r', b'o', b'n')
        tuple_of_bytes = tuple([bytes([c]) for c in token.encode("utf-8")]) # diff of bytes(97)[this is a init function, =[0]*97] and bytes([97])
        pre_token_freq[tuple_of_bytes] = freq
    end_time = time.time()
    logging.info(f"Pre-tokenization took {end_time - start_time:.2f} seconds")
    
    pair_freq = defaultdict(int)
    for tuple_of_bytes, freq in tqdm(pre_token_freq.items()):
        for i in range(len(tuple_of_bytes) - 1):
            pair_freq[(tuple_of_bytes[i], tuple_of_bytes[i + 1])] += freq
    # pair_freq # {((b'E', b'v'), 5), ((b'E', b'n'), 5), ((b'i', b'r'), 11)}
    count = 0
    start_time = time.time()
    while len(vocab) < vocab_size:
        most_freq_pair = max(pair_freq.items(), key=lambda x: (x[1], x))[0] # sort by freq, then by lexicographical order of the pair, return (b'i', b'r')
        # update merges
        # logging.info(f"Merge number {count}: {most_freq_pair}")
        count += 1
        merges.append(most_freq_pair)
        # update vocab
        new_token = most_freq_pair[0] + most_freq_pair[1]
        vocab[len(vocab)] = new_token
        # update pair_freq after merging most_freq_pair[0] # (b'i', b'r')
        new_pre_token_freq = {}
        for tuple_of_bytes, freq in pre_token_freq.items():
            i = 0
            for i in range(len(tuple_of_bytes) - 1):
                pair = tuple_of_bytes[i:i+2]
                if pair == most_freq_pair:
                    prefix = tuple_of_bytes[:i]
                    suffix = tuple_of_bytes[i+2:]
                    tuple_of_bytes = prefix + (new_token,) + suffix # update tuple_of_bytes while merging
                    # Update pair frequencies
                    if prefix:
                        pair_freq[(prefix[-1], new_token)] += freq
                        pair_freq[(prefix[-1], most_freq_pair[0])] -= freq
                    if suffix:
                        pair_freq[(new_token, suffix[0])] += freq
                        pair_freq[(most_freq_pair[1], suffix[0])] -= freq
                    pair_freq[most_freq_pair] -= freq
            # Update the pre-token frequency table
            new_pre_token_freq[tuple_of_bytes] = freq
        # update pre_token_freq
        pre_token_freq = new_pre_token_freq
    end_time = time.time()
    logging.info(f"Training took {end_time - start_time:.2f} seconds")
    logging.info(f"Performed {count} merges, average time per merge: {(end_time - start_time) / count:.2f} seconds")
        
    return vocab, merges
                
class BPE:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab # int -> bytes
        self.special_tokens = [] if special_tokens is None else special_tokens
        if self.special_tokens: # add special tokens to vocab if not already present
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token_bytes
        self.vocab_bytes_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merges_int = {}
        for byte1, byte2 in merges:
            self.merges_int[(self.vocab_bytes_to_id[byte1], self.vocab_bytes_to_id[byte2])] = self.vocab_bytes_to_id[byte1 + byte2]
        self.merges = merges
    
    @classmethod
    def from_files(cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str]):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)
    
    def _encode_part(self, text):
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        else:
            text_chunks = re.findall(GPT2_PRE_TOKEN_PATTERN, text)
            result = []
            for chunk in text_chunks:
                text_bytes = chunk.encode("utf-8")
                ids = [self.vocab_bytes_to_id[bytes([b])] for b in text_bytes]
                while len(ids)>=2:
                    pairs = get_pairs(ids)
                    high_priority_pair = min(pairs, key=lambda pair: self.merges_int.get(pair, float('inf')))
                    if high_priority_pair not in self.merges_int:
                        break
                    new_id = self.merges_int[high_priority_pair]
                    ids = update(ids, high_priority_pair, new_id)
                result.extend(ids)
            return result
    
    def split_by_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        
        # Sort by length and escape special regex characters
        sorted_tokens = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
        # Create pattern that captures the special tokens
        pattern = f'({"|".join(sorted_tokens)})'
        # Split and keep both tokens and text between them
        parts = re.split(pattern, text)
        # Filter out empty strings
        return [part for part in parts if part]
    
    def encode(self, text: str) -> list[int]:
        # pre-tokenize text, e.g "low low lower lowest"
        # merges: [(b'l', b'o'), (b'o', b'w')]
        parts = self.split_by_special_tokens(text)
        ids = []
        for chunk in tqdm(parts,
                          desc=f"Encoding {len(parts)} documents"):
            if chunk in self.special_tokens:
                ids.append(self.vocab_bytes_to_id[chunk.encode("utf-8")])
            else:
                ids.extend(self._encode_part(chunk))
        return ids
                
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[list[int]]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id
    
    def decode(self, tokens: list[int]) -> str:
        # decode token_ids to text
        # token_ids: [1, 2, 3, 4]
        # vocab: {1: b'l', 2: b'o', 3: b'w', 4: b'e', 5: b's', 6: b't'}
        # text: "lowes"
        return b''.join([self.vocab[t] for t in tokens]).decode('utf-8', errors='replace')
    
    def save(self, vocab_path: str | os.PathLike, merges_path: str | os.PathLike):
        save_voacb_and_merge(self.vocab, self.merges, vocab_path, merges_path)
                

if __name__ == "__main__":
    
    # Test BPE_v1
    path = "/Users/runshengliu/github/CS336-stanford-llm/spring2024-assignment1-basics/tests/fixtures/corpus.en"
    # path = "/root/autodl-tmp/github/CS336-Assignments/spring2024-assignment1-basics/tests/fixtures/corpus.en"
    # path = "/root/autodl-tmp/github/CS336-Assignments/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab, merges = train_bpe(path, 300, ["<endoftext>"], is_save=True, output_dir="./assets/test_my_bpe")
    tokenizer = BPE(vocab, merges, ["<endoftext>"])
    
    tokenizer.save(vocab_path="./assets/test_my_bpe/vocab.json", merges_path="./assets/test_my_bpe/merges.txt")
    tokenizer2 = BPE.from_files(vocab_path="./assets/test_my_bpe/vocab.json", merges_path="./assets/test_my_bpe/merges.txt", special_tokens=["<endoftext>"])
    assert vocab == tokenizer2.vocab
    assert merges == tokenizer2.merges
    
    tokenizer_gpt2 = BPE.from_files(vocab_path="/Users/runshengliu/github/CS336-stanford-llm/spring2024-assignment1-basics/tests/fixtures/gpt2_vocab.json", 
                                    merges_path="/Users/runshengliu/github/CS336-stanford-llm/spring2024-assignment1-basics/tests/fixtures/gpt2_merges.txt", 
                                    special_tokens=["<|endoftext|>"])
    tokenizer_gpt2.save(vocab_path="./assets/test_my_bpe/vocab.json", merges_path="./assets/test_my_bpe/merges.txt")
    
    # Test BPE_v2
    # tokenizer = BPE_v2.from_files(vocab_path="./assets/bpe_v1/vocab.txt", merges_path="./assets/bpe_v1/merges.txt", special_tokens=["<endoftext>"])
    
    
    # FIXTURES_PATH = '/Users/runshengliu/github/CS336-stanford-llm/spring2024-assignment1-basics/tests/fixtures'
    # VOCAB_PATH = os.path.join(FIXTURES_PATH, "gpt2_vocab.json")
    # MERGES_PATH = os.path.join(FIXTURES_PATH, "gpt2_merges.txt")
    # vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, 
    #                                                      merges_path=MERGES_PATH,
    #                                                      )
    # tokenizer = BPE_v2(vocab, merges, special_tokens=["<|endoftext|>"])
    
    
    # # text_string = "low low lower lowest"
    
    # # 1. test empty string roundtrip
    # test_string = "mohammedmad salah is king of egypt"
    # encoded_ids = tokenizer.encode(test_string)
    # decoded_string = tokenizer.decode(encoded_ids)
    # assert test_string == decoded_string
    # # pass
    
    # # 2. test empty string roundtrip with tiktoken
    # reference_tokenizer = tiktoken.get_encoding("gpt2")
    # test_string = ""

    # reference_ids = reference_tokenizer.encode(test_string)
    # ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    # tokenized_string = [tokenizer.decode([x]) for x in ids]
    # assert tokenized_string == []

    # assert tokenizer.decode(ids) == test_string
    # assert reference_tokenizer.decode(reference_ids) == test_string
    
    # # 3. test single unicode character matches tiktoken
    # test_string = "🙃"

    # reference_ids = reference_tokenizer.encode(test_string)
    # ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    # assert tokenizer.decode(ids) == test_string
    # assert reference_tokenizer.decode(reference_ids) == test_string
    
    # # 4. test unicode string with special tokens matches tiktoken
    # test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # encoded_ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # # Ensure the special <|endoftext|> token is preserved
    # assert tokenized_string.count("<|endoftext|>") == 3

    # decoded_string = tokenizer.decode(encoded_ids)
    # assert test_string == decoded_string
    
    # # 5. test overlapping special tokens
    # vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, 
    #                                                      merges_path=MERGES_PATH,
    #                                                      )
    # tokenizer = BPE_v2(vocab, merges, special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    # ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in ids]
    # # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    # assert tokenized_string.count("<|endoftext|>") == 1
    # assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # # Test roundtrip
    # assert tokenizer.decode(ids) == test_string