# bpe
import re
import regex
import os
import string
from collections import defaultdict
indices = list(map(int, string))
merges: dict[tuple[int, int], int] = {}
print(type(indices))

def merge_token_squence(token_seq, best_pair, new_token):
    new_seq = []
    i = 0
    while i < len(token_seq):
        if i < len(token_seq) - 1 and token_seq[i] == best_pair[0] and token_seq[i+1] == best_pair[1]:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    return new_seq

def run_train_bpe(input_file, vocab_size, special_tokens):
    vocab = {x:bytes([x]) for x in range(256)}
    existings_tokens = set(vocab.values())
    for special_token in special_tokens:
        if len(vocab) >= vocab_size:
            raise ValueError("vocab_size is too small for special tokens")
        str_bytes = special_token.encode('utf-8')
        if str_bytes in vocab.values():
            continue
        vocab[len(vocab)] = str_bytes
        existings_tokens.add(str_bytes)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    chunks = regex.split('|'.join(map(regex)))


class bpe:
    def __init__(self, vocab_size: int, merges_file: str):
        self.vocab_size = vocab_size
        self.merges_file = merges_file
        self.indices = list(map(int, string.encode('utf-8')))
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab = {x:bytes([x]) for x in range(256)}

        nb_merges = 100
        for i in range(nb_merges):
            counts = {}
            for index1, index2 in zip(self.indices, self.indices[1:]):
                pair = (index1, index2)
                if pair not in counts:
                    counts[pair] = 0
                counts[pair] += 1

            pair = max(counts, key=counts.get)
            index1, index2 = pair
            new_index = self.vocab_size + i
            merges[pair] = new_index
            self.vocab[new_index] = bytes([index1, index2])
            # update indices
            new_indices = []
            i = 0
            while i < len(self.indices):
                if i < len(self.indices) - 1 and self.indices[i] == index1 and self.indices[i+1] == index2:
                    new_indices.append(new_index)
                    i += 2
                else:
                    new_indices.append(self.indices[i])
                    i += 1
            self.indices = new_indices




class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
    
    def _bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}
    
    def get_stats(self, token_ids):
        pairs = defaultdict(int)
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i+1])
            pairs[pair] += 1
        return pairs
    
    def merge(self, token_ids, pair, mew_id):
        new_ids = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i+1] == pair[1]:
                new_ids.append(mew_id)
                i += 2
            else:
                new_ids.append(token_ids[i])
                i += 1
        return new_ids
    
    def train(self, text, verbose=False):

        token_ids = list(map(int, text.encode('utf-8')))
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            stats = self.get_stats(token_ids)
            if not stats:
                break

            most_frequent_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            token_ids = self.merge(token_ids, most_frequent_pair, new_id)

            self.merges[most_frequent_pair] = new_id
            self.vocab[new_id] = bytes([most_frequent_pair[0], most_frequent_pair[1]])

        return self.merges, self.vocab
    
    def encode(self, text):
        token_ids = list(map(int, text.encode('utf-8')))
        changed = True
        while changed and len(token_ids) > 1:
            changed = False
            stats = self.get_stats(token_ids)
            for pair, new_id in self.merges.items():
                if pair in stats and stats[pair] > 0:
                    token_ids = self.merge(token_ids, pair, new_id)
                    changed = True
                    break
        return token_ids
    
    def decode(self, token_ids):
        text = []
        for token_id in token_ids:
            if token_id in self.vocab:
                text.append(self.vocab[token_id])
            else:
                try:
                    text.append(bytes([token_id]))
                except:
                    text.append(b'?')
        text_bytes = b''.join(text)
        try:
            text = text_bytes.decode('utf-8')
        except:
            text = text_bytes.decode('utf-8', errors='replace')
        return text

            

