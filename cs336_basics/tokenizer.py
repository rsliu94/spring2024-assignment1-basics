import regex as re
import os
from collections import Counter, defaultdict
from cs336_basics.utils.io import GPT2_PRE_TOKEN_PATTERN, WHITESPACE_PATTERN
import logging
from tqdm import tqdm
import time
logging.basicConfig(level=logging.INFO)

class BPE_v1:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        # init vocab with 256 byte values and special tokens
        self.vocab = {i: bytes([i]) for i in range(256)} # token_id -> token_bytes
        for i, token in enumerate(special_tokens):
            self.vocab[256 + i] = token.encode("utf-8")
        self.merges = [] # list of (token1_bytes, token2_bytes)
        
    def pre_tokenize(self, input_path: str | os.PathLike, special_tokens: list[str]):
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
        return pre_token_freq

    def train(self, input_path: str | os.PathLike, special_tokens: list[str]):
        # pre-tokenize input file
        pre_token_freq = self.pre_tokenize(input_path, special_tokens) # ('\xa4', b'a', b'l'): 1
        pair_freq = defaultdict(int)
        for tuple_of_bytes, freq in tqdm(pre_token_freq.items()):
            for i in range(len(tuple_of_bytes) - 1):
                pair_freq[(tuple_of_bytes[i], tuple_of_bytes[i + 1])] += freq
        # pair_freq # {((b'E', b'v'), 5), ((b'E', b'n'), 5), ((b'i', b'r'), 11)}
        count = 0
        start_time = time.time()
        while len(self.vocab) < self.vocab_size:
            most_freq_pair = max(pair_freq.items(), key=lambda x: (x[1], x))[0] # sort by freq, then by lexicographical order of the pair, return (b'i', b'r')
            # update merges
            # logging.info(f"Merge number {count}: {most_freq_pair}")
            count += 1
            self.merges.append(most_freq_pair)
            # update vocab
            new_token = most_freq_pair[0] + most_freq_pair[1]
            self.vocab[len(self.vocab)] = new_token
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
            
        return self.vocab, self.merges

    def save(self, output_dir: str | os.PathLike):
        """Save vocabulary and merges to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vocab
        vocab_path = os.path.join(output_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token_id, token_bytes in sorted(self.vocab.items()):
                # Convert bytes to hex representation for readability
                hex_repr = token_bytes.hex()
                f.write(f"{token_id}\t{hex_repr}\n")
        
        # Save merges
        merges_path = os.path.join(output_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for token1, token2 in self.merges:
                hex1, hex2 = token1.hex(), token2.hex()
                f.write(f"{hex1} {hex2}\n")
        logging.info(f"Saved vocabulary and merges to {output_dir}")
                
    def load(self, input_dir: str | os.PathLike):
        """Load vocabulary and merges from disk"""
        # Load vocab
        vocab_path = os.path.join(input_dir, "vocab.txt")
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                token_id, hex_repr = line.strip().split("\t")
                self.vocab[int(token_id)] = bytes.fromhex(hex_repr)
                
        # Load merges
        merges_path = os.path.join(input_dir, "merges.txt")
        self.merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                hex1, hex2 = line.strip().split()
                self.merges.append((bytes.fromhex(hex1), bytes.fromhex(hex2)))

if __name__ == "__main__":
    tokenizer = BPE_v1(300, ["<endoftext>"])
    # path = "/Users/runshengliu/github/CS336-stanford-llm/spring2024-assignment1-basics/tests/fixtures/corpus.en"
    # path = "/root/autodl-tmp/github/CS336-Assignments/spring2024-assignment1-basics/tests/fixtures/corpus.en"
    path = "/root/autodl-tmp/github/CS336-Assignments/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab, merges = tokenizer.train(path, ["<endoftext>"])
    tokenizer.save("./assets/bpe_v1")
    