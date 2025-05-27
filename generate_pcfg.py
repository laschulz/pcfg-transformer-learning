import random
import os
import pickle
import numpy as np
import json
import argparse

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from nltk.grammar import PCFG
from nltk.parse import ViterbiParser

MAX_SEQUENCE_LENGTH = 100
DATASET_SIZE = 1000

GRAMMARS = {
    # G2  ─ LinearRecursion
    "LinearRecursion": {
        "S": [(["A"], .4), (["B"], .4), (["c"], .2)],
        "A": [(["a"], .5), (["A", "aa"], .5)],
        "B": [(["b"], .5), (["B", "b"], .5)],
    },

    # G3  ─ MutualRecursion
    "MutualRecursion": {
        "S": [(["A"], 1.0)],
        "A": [(["a", "B"], .5), (["a"], .5)],
        "B": [(["b", "A"], .5), (["b"], .5)],
    },

    # G4  ─ CenterEmbedding
    "CenterEmbedding": {
        "S": [(["a", "S", "b"], .6), (["c"], .4)],
    },

    # G5  ─ ConditionalLoops
    "ConditionalLoops": {
        "S": [
            (["if", "C", "then", "S", "else", "T"], .4),
            (["while", "C", "do", "S"], .2),
            (["action"], .4)],
        "T": [(["t", "T"], .6), (["z"], .4)],
        "C": [(["cond"], .5), (["not", "C"], .25), (["C", "and", "C"], .25)],
    },

    # G6  ─ ArithmeticLogic
    "ArithmeticLogic": {
        "S": [(["E"], 1.0)],
        "E": [(["E", "+", "T"], .25), (["T"], .75)],
        "T": [(["T", "*", "F"], .25), (["F"], .75)],
        "F": [
            (["(", "E", ")"], .3),
            (["val"], .4),
            (["if", "C", "then", "F", "else", "F"], .3)],
        "C": [(["cond"], .5), (["not", "C"], .25), (["C", "and", "C"], .25)],
    },

    # G7  ─ NestedStructures   (double-quote symbol included)
    "NestedStructures": {
        "S": [(["(", "A", ")"], .5), (["*", "X", "X", "*"], .5)],
        "A": [(["B", "B", "B"], .4), (["a"], .6)],
        "B": [(["[", "S", "]"], .5), (["b"], .5)],
        "X": [(["Y", '"', "Y", '"', "Y"], .6), (["x"], .4)],
        "Y": [(["S", "y", "S"], .4), (["S", "X"], .3), (["y"], .3)],
    },
}

# class PCFG_G2:
#     def __init__(self):
#         self.rules = {
#             "S": [(["A"], 0.4), (["B"], 0.4), (["c"], 0.2)],
#             "A": [(["a"], 0.5), (["A", "aa"], 0.5)],
#             "B": [(["b"], 0.5), (["B", "b"], 0.5)],
#         }
#         self.name = 'LinearRecursion'

# class PCFG_G3:
#     def __init__(self):
#         self.rules = {
#             "S": [(["A"], 1.0)],
#             "A": [(["a", "B"], 0.5), (["a"], 0.5)],
#             "B": [(["b", "A"], 0.5), (["b"], 0.5)],
#         }
#         self.name = 'MutualRecursion'

# class PCFG_G4:
#     def __init__(self):
#         self.rules = {
#             "S": [(["a", "S", "b"], 0.6), (["c"], 0.4)],
#         }
#         self.name = 'CenterEmbedding'

# class PCFG_G5:
#     def __init__(self):
#         self.rules = {
#             "S": [(["if", "C", "then", "S", "else", "T"], 0.4),
#                   (["while", "C", "do", "S"], 0.2),
#                   (["action"], 0.4)],
#             "T": [(["t", "T"], 0.6), (["z"], 0.4)],
#             "C": [(["cond"], 0.5), (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
#         }
#         self.name = 'ConditionalLoops'

# class PCFG_G6:
#     def __init__(self):
#         self.rules = {
#             "S": [(["E"], 1.0)],
#             "E": [(["E", "+", "T"], 0.25), (["T"], 0.75)],
#             "T": [(["T", "*", "F"], 0.25), (["F"], 0.75)],
#             "F": [(["(", "E", ")"], 0.3),
#                   (["val"], 0.4),
#                   (["if", "C", "then", "F", "else", "F"], 0.3)],
#             "C": [(["cond"], 0.5), (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
#         }
#         self.name = 'ArithmeticLogic'

# class PCFG_G7:
#     def __init__(self):
#         self.rules = PCFG.fromstring("""
#             S -> '(' A ')' [0.5] | '*' X X '*' [0.5]
#             A -> B B B [0.4] | 'a' [0.6]
#             B -> '[' S ']' [0.5] | 'b' [0.5]
#             X -> Y '"' Y '"' Y [0.8] | 'x' [0.5]
#             Y -> S 'y' S [0.4] | S X [0.3] | 'y' [0.3]
#         """)
#         self.name = 'NestedStructures'

def dict_to_pcfg(table):
    """Convert rule table to an nltk-parsable PCFG string."""
    lines = []
    for lhs, rhs_list in table.items():
        for rhs, p in rhs_list:
            rhs_str = " ".join("'" + t + "'" if t not in table else t
                               for t in rhs)
            lines.append(f"{lhs} -> {rhs_str} [{p}]")
    pcfg_str = "\n".join(lines)
    return pcfg_str

PARSERS = {name: ViterbiParser(PCFG.fromstring(dict_to_pcfg(tbl)))
           for name, tbl in GRAMMARS.items()}

def validate(sequence, grammar_name):
    """True iff sequence (list or space-string) derives from grammar."""
    text_value = sequence["text"]
    tokens = text_value.split() if isinstance(text_value, str) else list(text_value)
    try:
        return any(PARSERS[grammar_name].parse(tokens))
    except ValueError as err:          # token not covered by the grammar
        print(f"[VALIDATION ERROR - {grammar_name}] {tokens}\n{err}\n")
        return False

# class LanguageGenerator:
#     def __init__(self, pfcg):
#         self.rules = pfcg.rules

#     def generate_sequence(self, max_length):
#         sequence = []
#         stack = ['S']  # Start with the start symbol

#         while stack and len(sequence) < max_length:
#             symbol = stack.pop()
#             if symbol in self.rules:
#                 productions, probs = zip(*self.rules[symbol])
#                 production = random.choices(productions, weights=probs, k=1)[0]
#                 stack.extend(reversed(production))
#             else:
#                 sequence.append(symbol)
#         return sequence

#     def generate_sequences(self, num_sequences, max_length):
#         return [self.generate_sequence(max_length) for _ in range(num_sequences)]

# def save_dataset(sequences, dataname):
#     os.makedirs(dataname, exist_ok=True)
#     jsonl_lines = []

#     for seq in sequences:
#         text = ' '.join(seq)
#         jsonl_lines.append({"sequence": text})

#     jsonl_path = os.path.join(dataname, 'full.jsonl')
#     with open(jsonl_path, 'w') as f:
#         for entry in jsonl_lines:
#             f.write(json.dumps(entry) + '\n')


def save_dataset(seqs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/full.jsonl", "w") as f:
        for s in seqs:
            f.write(json.dumps({"sequence": " ".join(s)}) + "\n")

def sample(grammar_name,max_len=MAX_SEQUENCE_LENGTH):
    tbl=GRAMMARS[grammar_name]; seq,stack=[],['S']
    while stack and len(seq)<max_len:
        sym=stack.pop()
        if sym in tbl:
            prods,probs=zip(*tbl[sym])
            stack.extend(reversed(random.choices(prods,probs)[0]))
        else:
            seq.append(sym)
    return seq

def sample_many(grammar_name,n,max_len=MAX_SEQUENCE_LENGTH):
    return [sample(grammar_name,max_len) for _ in range(n)]


# def split_and_tokenize_jsonl(jsonl_path, tokenizer, output_dir, train_ratio=0.9):
#     os.makedirs(output_dir, exist_ok=True)

#     # Load data
#     with open(jsonl_path, 'r') as f:
#         lines = [json.loads(line) for line in f]

#     # Split
#     split_idx = int(len(lines) * train_ratio)
#     train_lines = lines[:split_idx]
#     val_lines = lines[split_idx:]

#     for split_name, split_lines in [("train", train_lines), ("val", val_lines)]:
#         token_ids = []
#         for entry in split_lines:
#             text = entry["sequence"]
#             text_with_bos = tokenizer.bos_token + ' ' + text
#             ids = tokenizer.encode(text_with_bos)
#             token_ids.extend(ids)

#         arr = np.array(token_ids)
#         bin_path = os.path.join(output_dir, f"{split_name}.bin")
#         arr.tofile(bin_path)

#         meta = {
#             'vocab_size': len(tokenizer),
#             'bos_token_id': tokenizer.bos_token_id,
#         }
#         with open(os.path.join(output_dir, f"meta_{split_name}.pkl"), 'wb') as f:
#             pickle.dump(meta, f)

#         with open(os.path.join(output_dir, f"{split_name}.jsonl"), 'w') as f:
#             for entry in split_lines:
#                 f.write(json.dumps(entry) + '\n')

def split_and_tokenize(jsonl_path, tok_fast, out_dir, train_ratio=.9):
    os.makedirs(out_dir, exist_ok=True)
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f]
    split = int(len(lines)*train_ratio)
    for name, data in [("train", lines[:split]), ("val", lines[split:])]:
        ids = []
        for row in data:
            ids.extend(tok_fast.encode(tok_fast.bos_token + " " + row["sequence"]))
        np.array(ids).tofile(f"{out_dir}/{name}.bin")
        with open(f"{out_dir}/meta_{name}.pkl", "wb") as m:
            pickle.dump({"vocab_size": len(tok_fast),
                        "bos_token_id": tok_fast.bos_token_id}, m)
        with open(f"{out_dir}/{name}.jsonl", "w") as j:
            for row in data: j.write(json.dumps(row)+"\n")

# def train_custom_tokenizer(jsonl_path, tokenizer_path, vocab_size=511):
#     # Read sequences from jsonl
#     with open(jsonl_path, 'r') as f:
#         lines = [json.loads(line)["sequence"] for line in f]

#     # Save all text to a temporary training file
#     raw_path = os.path.join(os.path.dirname(tokenizer_path), "tokenizer_raw.txt")
#     with open(raw_path, 'w') as f:
#         for line in lines:
#             f.write(line.strip() + '\n')

#     # Initialize tokenizer
#     tokenizer = Tokenizer(models.BPE())
#     tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
#     tokenizer.pre_tokenizer = Whitespace()

#     # Train it
#     trainer = trainers.BpeTrainer(
#         vocab_size=vocab_size,
#         special_tokens=["<|bos|>"]
#     )
#     tokenizer.train([raw_path], trainer)

#     # Add template for decoding (optional)
#     tokenizer.post_processor = TemplateProcessing(
#         single="<|bos|> $A",
#         special_tokens=[
#             ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
#         ],
#     )
#     # Save tokenizer to file
#     tokenizer_path = os.path.abspath(tokenizer_path)
#     tokenizer.save(tokenizer_path)

def train_custom_tokenizer(jsonl_path, tok_path, vocab=512):
    with open(jsonl_path) as f:
        corpus = [json.loads(l)["sequence"] for l in f]
    tmp = os.path.join(os.path.dirname(tok_path), "tmp_corpus.txt")
    with open(tmp, "w") as f: f.write("\n".join(corpus))
    tok = Tokenizer(models.BPE())
    tok.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()
    tok.train([tmp], trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|bos|>"]))
    tok.post_processor = TemplateProcessing(
        single="<|bos|> $A", special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>"))]
    )
    tok.save(os.path.abspath(tok_path))

def main():
    parser=argparse.ArgumentParser(
        description="Generate and optionally tokenise PCFG datasets.")
    parser.add_argument("--grammar","-g",
                        choices=sorted(GRAMMARS),
                        help="Name of the grammar to use.")
    parser.add_argument("--dataset_size","-n",type=int,default=1000,
                        help="Number of sequences to generate.")
    parser.add_argument("--max-len",type=int,default=100,
                        help="Max symbols per generated sequence.")
    args=parser.parse_args()

    # 1) generate
    sequences=sample_many(args.grammar,args.dataset_size,args.max_len)

    # 2) validate first & print quick stats
    valid_ratio=sum(validate(s,args.grammar) for s in sequences)/args.dataset_size
    print(f"Generated {args.dataset_size} sequences from '{args.grammar}' – "
          f"{valid_ratio:.1%} valid (should be 100%).")

    # 3) save dataset
    out_dir=f"data/{args.grammar}/{args.grammar}_{args.dataset_size}"
    save_dataset(sequences,out_dir)
    print("Saved dataset to",out_dir)

    # 4) optional BPE tokenisation
    tok_path=f"{out_dir}/tokenizer.json"
    train_custom_tokenizer(f"{out_dir}/full.jsonl",tok_path)
    tok_fast=PreTrainedTokenizerFast(tokenizer_file=tok_path,bos_token="<|bos|>")
    split_and_tokenize(f"{out_dir}/full.jsonl",tok_fast,out_dir)
    print("Tokeniser + splits written.")

if __name__=="__main__":
    main()