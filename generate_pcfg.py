import random
from transformers import GPT2Tokenizer
import os
import pickle
import numpy as np
import json

MAX_SEQUENCE_LENGTH = 100
DATASET_SIZE = 500

class PCFG_G2:
    def __init__(self):
        self.rules = {
            "S": [(["A"], 0.4), (["B"], 0.4), (["c"], 0.2)],
            "A": [(["a"], 0.5), (["A", "aa"], 0.5)],
            "B": [(["b"], 0.5), (["B", "b"], 0.5)],
        }
        self.name = 'LinearRecursion'

class PCFG_G3:
    def __init__(self):
        self.rules = {
            "S": [(["A"], 1.0)],
            "A": [(["a", "B"], 0.5), (["a"], 0.5)],
            "B": [(["b", "A"], 0.5), (["b"], 0.5)],
        }
        self.name = 'MutualRecursion'

class PCFG_G4:
    def __init__(self):
        self.rules = {
            "S": [(["a", "S", "b"], 0.6), (["c"], 0.4)],
        }
        self.name = 'CenterEmbedding'

class PCFG_G5:
    def __init__(self):
        self.rules = {
            "S": [(["if", "C", "then", "S", "else", "T"], 0.4),
                  (["while", "C", "do", "S"], 0.2),
                  (["action"], 0.4)],
            "T": [(["t", "T"], 0.6), (["z"], 0.4)],
            "C": [(["cond"], 0.5), (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
        }
        self.name = 'ConditionalLoops'

class PCFG_G6:
    def __init__(self):
        self.rules = {
            "S": [(["E"], 1.0)],
            "E": [(["E", "+", "T"], 0.25), (["T"], 0.75)],
            "T": [(["T", "*", "F"], 0.25), (["F"], 0.75)],
            "F": [(["(", "E", ")"], 0.3),
                  (["val"], 0.4),
                  (["if", "C", "then", "F", "else", "F"], 0.3)],
            "C": [(["cond"], 0.5), (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
        }
        self.name = 'ArithmeticLogic'

class PCFG_G7:
    def __init__(self):
        self.rules = {
            "S": [(["(", "A", ")"], 0.3), (["*", "X", "X", "*"], 0.3), ([], 0.4)],
            "A": [(["B", "B", "B"], 0.4), (["a"], 0.6)],
            "B": [(["[", "S", "]"], 0.5), (["b"], 0.5)],
            "X": [(["Y", '"', "Y", '"', "Y"], 0.8), (["x"], 0.5)],
            "Y": [(["S", "y", "S"], 0.4), (["S", "X"], 0.3), (["y"], 0.3)],
        }
        self.name = 'NestedStructures'


class LanguageGenerator:
    def __init__(self, pfcg):
        self.rules = pfcg.rules

    def generate_sequence(self, max_length):
        sequence = []
        stack = ['S']  # Start with the start symbol

        while stack and len(sequence) < max_length:
            symbol = stack.pop()
            if symbol in self.rules:
                productions, probs = zip(*self.rules[symbol])
                production = random.choices(productions, weights=probs, k=1)[0]
                stack.extend(reversed(production))
            else:
                sequence.append(symbol)
        return sequence

    def generate_sequences(self, num_sequences, max_length):
        return [self.generate_sequence(max_length) for _ in range(num_sequences)]

def save_dataset(sequences, dataname):
    os.makedirs(dataname, exist_ok=True)
    jsonl_lines = []

    for seq in sequences:
        text = ' '.join(seq)
        jsonl_lines.append({"sequence": text})

    jsonl_path = os.path.join(dataname, 'full.jsonl')
    with open(jsonl_path, 'w') as f:
        for entry in jsonl_lines:
            f.write(json.dumps(entry) + '\n')

def split_and_tokenize_jsonl(jsonl_path, output_dir, train_ratio=0.9, seed=42):
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'bos_token': '<|bos|>'})

    # Load data
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]

    # split
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    for split_name, split_lines in [("train", train_lines), ("val", val_lines)]:
        token_ids = []
        for entry in split_lines:
            text = entry["sequence"]
            text_with_bos = tokenizer.bos_token + ' ' + text
            ids = tokenizer.encode(text_with_bos)
            token_ids.extend(ids)

        # Save .bin
        arr = np.array(token_ids, dtype=np.uint16)
        bin_path = os.path.join(output_dir, f"{split_name}.bin")
        arr.tofile(bin_path)

        # Save metadata
        meta = {
            'vocab_size': tokenizer.vocab_size + 1,
            'bos_token_id': tokenizer.bos_token_id,
        }
        meta_path = os.path.join(output_dir, f"meta_{split_name}.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)

        # Save split .jsonl 
        jsonl_out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(jsonl_out_path, 'w') as f:
            for entry in split_lines:
                f.write(json.dumps(entry) + '\n')


pcfg = PCFG_G2()
generator = LanguageGenerator(pcfg)
sequences = generator.generate_sequences(DATASET_SIZE, MAX_SEQUENCE_LENGTH)
save_dataset(sequences, f'data/{pcfg.name}/{pcfg.name}_{DATASET_SIZE}')
split_and_tokenize_jsonl(f'data/{pcfg.name}/{pcfg.name}_{DATASET_SIZE}/full.jsonl', f'data/{pcfg.name}/{pcfg.name}_{DATASET_SIZE}', train_ratio=0.9)