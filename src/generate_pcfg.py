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

from def_pcfgs import GRAMMARS

MAX_SEQUENCE_LENGTH = 100
DATASET_SIZE = 1000

def dict_to_pcfg(table):
    """Convert rule table to an nltk-parsable PCFG string."""
    lines = []
    for lhs, rhs_list in table.items():
        for rhs, p in rhs_list:
            rhs_str = " ".join("'" + t + "'" if t not in table else t for t in rhs)
            lines.append(f"{lhs} -> {rhs_str} [{p}]")
    pcfg_str = "\n".join(lines)
    return pcfg_str

PARSERS = {name: ViterbiParser(PCFG.fromstring(dict_to_pcfg(tbl)))
           for name, tbl in GRAMMARS.items()}

def validate(tokenizer, sequence, grammar_name):
    """True iff sequence (list or space-string) derives from grammar
        and if length of the sequence is <= 126 tokens after tokenization.
    """
    tokens = sequence.split() if isinstance(sequence, str) else list(sequence)
    tokens = [tok for tok in tokens if tok not in {"<|eos|>", "<|bos|>"}]

    encoded_ids = tokenizer.encode(' '.join(tokens), add_special_tokens=False)
    if len(encoded_ids) > 126:
        return False

    try:
        is_valid = any(PARSERS[grammar_name].parse(tokens))
        return is_valid
    except ValueError as err:          # token not covered by the grammar
        print(f"[VALIDATION ERROR - {grammar_name}] {tokens}\n{err}\n")
        return False

def save_dataset(test_pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/test.jsonl", "w") as f:
        for seq, log_prob in test_pairs:
            f.write(json.dumps({
                "sequence": seq,
                "real_log_prob": log_prob
            }) + "\n")


def sample(grammar_name, start_symbol, max_len=MAX_SEQUENCE_LENGTH):
    tbl = GRAMMARS[grammar_name]

    while True:
        seq = []
        stack = [start_symbol]
        log_prob = 0.0

        # Grow seq until either stack empties or seq reaches max_len
        while stack and len(seq) < max_len:
            sym = stack.pop()
            if sym in tbl:
                prods, probs = zip(*tbl[sym])
                idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
                chosen_prod = prods[idx]
                chosen_prob = probs[idx]

                # Accumulate log-probability of this production choice
                log_prob += np.log(chosen_prob)

                stack.extend(reversed(chosen_prod))
            else:
                seq.append(sym)

        # If stack emptied before hitting max_len, we got a fully terminating sequence
        if not stack:
            return " ".join(seq), log_prob

def sample_many(grammar_name, start_symbol, n, max_len=MAX_SEQUENCE_LENGTH):
    return [sample(grammar_name, start_symbol, max_len) for _ in range(n)]

def count_valid_sequences(tokenizer, sequences, grammar_name):
    total = len(sequences)
    valid = sum(validate(tokenizer, seq, grammar_name) for seq, _ in sequences)
    print(f"[VALIDITY] {valid}/{total} sequences valid ({100 * valid / total:.2f}%)")
    return valid, total

def split_and_tokenize(sequences, tok_fast, out_dir, train_ratio=.9):
    os.makedirs(out_dir, exist_ok=True)
    split = int(len(sequences) * train_ratio)
    for name, data in [("train", sequences[:split]), ("val", sequences[split:])]:
        ids = []
        for seq in data:
            ids.extend(tok_fast.encode(tok_fast.bos_token + " " + seq + " " + tok_fast.eos_token))
        np.array(ids, dtype=np.uint32).tofile(f"{out_dir}/{name}.bin")
        with open(f"{out_dir}/{name}.jsonl", "w") as j:
            for seq in data:
                j.write(json.dumps({"sequence": seq}) + "\n")
    with open(f"{out_dir}/meta.pkl", "wb") as m:
        pickle.dump({
            "vocab_size": len(tok_fast),
            "bos_token_id": tok_fast.bos_token_id,
            "eos_token_id": tok_fast.eos_token_id
        }, m)

def build_fixed_tokenizer(grammar_name: str, tok_path: str, special_tokens=None):
    """
    Create a WordLevel tokenizer whose vocab = exactly the terminals
    of GRAMMARS[grammar_name], plus any provided special_tokens.
    Save it to tok_path and return a PreTrainedTokenizerFast wrapper.
    """
    if special_tokens is None:
        special_tokens = ["<|bos|>", "<|eos|>"]

    # 1) Extract all grammar terminals
    tbl = GRAMMARS[grammar_name]
    terminals = set()
    for lhs, productions in tbl.items():
        for rhs, _ in productions:
            for sym in rhs:
                # anything not in tbl’s keys is a terminal
                if sym not in tbl:
                    terminals.add(sym)

    # 2) Build a WordLevel vocabulary: map each terminal → unique ID.
    #    We reserve IDs for special_tokens first, then assign terminals.
    vocab = {}
    idx = 0
    for tok in special_tokens:
        vocab[tok] = idx
        idx += 1
    for term in sorted(terminals):
        vocab[term] = idx
        idx += 1

    # 3) Create a WordLevel model with that vocabulary
    wordlevel = models.WordLevel(vocab=vocab)
    tokenizer = Tokenizer(wordlevel)

    # 4) Use whitespace as pre‐tokenizer so that each terminal stays intact
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 5) Post‐processor: wrap with <|bos|> and <|eos|>
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair=None,
        special_tokens=[
            ("<|bos|>", vocab["<|bos|>"]),
            ("<|eos|>", vocab["<|eos|>"]),
        ],
    )

    # 6) Save to disk
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    tokenizer.save(tok_path)

    # 7) Wrap in PreTrainedTokenizerFast so you can call .encode/.decode easily
    tok_fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )

    print(vocab)
    return tok_fast

def generate_pcfg(grammar, start_symbol, dataset_size, max_len, tokenizer_path):
    # 1) generate
    train_sequences = sample_many(grammar, start_symbol, dataset_size, max_len)
    test_sequences = sample_many(grammar, start_symbol, 500, max_len)
    str_sequences = [seq for seq, _ in train_sequences]

    # 2) save dataset
    out_dir = f"../data/{grammar}/{grammar}_{dataset_size}"
    save_dataset(test_sequences, out_dir)

    # 3) BPE tokenisation
    if tokenizer_path:     #load tokenizer from path
        tok_path = tokenizer_path 
        tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token="<|bos|>", eos_token="<|eos|>")
        print(tok_fast.get_vocab())
    else:
        tok_path = f"{out_dir}/tokenizer.json"
        tok_fast = build_fixed_tokenizer(grammar, tok_path)

    split_and_tokenize(str_sequences, tok_fast, out_dir)

    count_valid_sequences(tok_fast, test_sequences, grammar)
    count_valid_sequences(tok_fast, train_sequences, grammar)

def parse_args():
    parser=argparse.ArgumentParser(
        description="Generate and optionally tokenise PCFG datasets.")
    parser.add_argument("--grammar", choices=sorted(GRAMMARS),
                        help="Name of the grammar to use.")
    parser.add_argument("--dataset_size","-n",type=int,default=1000,
                        help="Number of sequences to generate.")
    parser.add_argument("--start_symbol", type=str,default="L0"),
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_len",type=int,default=100,
                        help="Max symbols per generated sequence.")
    return parser.parse_args()

def main():
    args=parse_args()
    generate_pcfg(args.grammar, args.start_symbol, args.dataset_size, args.max_len, args.tokenizer_path)

if __name__=="__main__":
    main()