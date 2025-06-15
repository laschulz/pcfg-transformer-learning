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
    "LinearRecursion": {
        "S": [(["Aa"], .5), (["Bb"], .4), (["c"], .1)],
        "A": [(["a"], .3), (["A", "a"], .7)],
        "B": [(["b"], .2), (["B", "b"], .8)],
    },

    "MutualRecursion": {
        "S": [(["a", "B"], .7), (["a"], .3)],
        "B": [(["b", "S"], .7), (["b"], .3)],
    },

    "CenterEmbedding": {
        "S": [(["a", "S", "b"], .8), (["c"], .2)],
    },

    "ConditionalLoops": {
        "S": [
            (["if", "C", "then", "S", "else", "T"], .4),
            (["while", "C", "do", "S"], .2),
            (["action"], .4)],
        "T": [(["t", "T"], .6), (["z"], .4)],
        "C": [(["cond"], .5), (["not", "C"], .25), (["C", "and", "C"], .25)],
    },

    "ArithmeticLogic": {
        "S": [(["S", "+", "T"], .25), (["T"], .75)],
        "T": [(["T", "*", "F"], .25), (["F"], .75)],
        "F": [
            (["(", "S", ")"], .1),
            (["val"], .8),
            (["if", "C", "then", "F", "else", "F"], .1)],
        "C": [(["cond"], .5), (["not", "C"], .25), (["C", "and", "C"], .25)],
    },
    
    "ConditionalLoops_plus": {
        "S": [
            (["if", "C", "then", "S", "else", "T"], 0.4),
            (["while", "C", "do", "S"], 0.2),
            (["action"], 0.4)],
        "T": [(["t", "T"], 0.6), (["z"], 0.4)],
        "C": [(["x", "R", "y"], 0.5), (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
        "R": [(["x", "==", "y"], 0.2), (["x", "<=", "y"], 0.2), (["x", "<", "y"], 0.2), 
              (["x", ">=", "y"], 0.2), (["x", ">", "y"], 0.2)]
    },

    "ArithmeticLogic_plus": {
        "S": [(["S", "+", "T"], 0.25), (["T"], 0.75)],
        "T": [(["T", "*", "F"], 0.25), (["F"], 0.75)],
        "F": [
            (["(", "S", ")"], 0.3), (["V"], 0.4),
            (["if", "C", "then", "F", "else", "F"], 0.3)],
        "V": [(["V", "0"], 0.2), (["V", "1"], 0.2), (["1"], 0.6)],
        "C": [(["R"], 0.5), (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
        "R": [(["x", "==", "y"], 0.2), (["x", "<=", "y"], 0.2), (["x", "<", "y"], 0.2), 
              (["x", ">=", "y"], 0.2), (["x", ">", "y"], 0.2)]
    },

    # G7  ─ NestedStructures
    "NestedStructures": {
        "S": [(["(", "A", ")"], .5), (["*", "X", "X", "*"], .5)],
        "A": [(["B", "B"], .4), (["a"], .6)],
        "B": [(["[", "S", "]"], .3), (["b"], .7)],
        "X": [(["Y", '"', "Y"], .3), (["x"], .7)],
        "Y": [(["S", "y", "S"], .15), (["S", "X"], .15), (["y"], .7)]
    },

    # Condition
    "Conditionals_oneLevel": {
        "S": [(["C"], 1.0)],
        "C": [(["cond"], .5), (["not", "C"], .25), (["C", "and", "C"], .25)]
    },

    "Conditionals_twoLevels": {
        "S": [(["C"], 1.0)],
        "C": [([ "R"], 0.5),
            (["not", "C"], 0.25), (["C", "and", "C"], 0.25)],
        "R": [(["x", "==", "y"], 0.2), (["x", "<=", "y"], 0.2), (["x", "<", "y"], 0.2), 
              (["x", ">=", "y"], 0.2), (["x", ">", "y"], 0.2)]
    }, 

    "Comparisons": {
        "S": [(["R"], 1.0)],
        "R": [(["x", "==", "y"], 0.2), (["x", "<=", "y"], 0.2), (["x", "<", "y"], 0.2), 
              (["x", ">=", "y"], 0.2), (["x", ">", "y"], 0.2)]
    },

    "BinaryValues": {
        "S": [(["V"], 1.0)],
        "V": [(["V", "0"], 0.2), (["V", "1"], 0.2), (["1"], 0.6)]
    },

    # Ts
    "Ts": {
        "S": [(["T"], 1.0)],
        "T": [(["t", "T"], .6), (["z"], .4)],
    }
}

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

def compute_min_len(tbl):
    """
    Given a grammar table `tbl: dict[str, List[(rhs: List[str], prob: float)]]`,
    return a dict mapping each nonterminal -> minimal number of terminals required
    to derive from it. Terminals count as length 1.
    """
    # Initialize min_len for each nonterminal as “infinity” (unknown)
    min_len = {NT: float('inf') for NT in tbl}

    # Iteratively relax until fixpoint
    changed = True
    while changed:
        changed = False
        for NT, productions in tbl.items():
            best = min_len[NT]
            for rhs, _prob in productions:
                total = 0
                valid = True
                for sym in rhs:
                    if sym in tbl:
                        length = min_len[sym]
                        if length == float('inf'):
                            valid = False
                            break
                        total += length
                    else:
                        # sym is a terminal → length 1
                        total += 1
                if valid and total < best:
                    best = total
            if best < min_len[NT]:
                min_len[NT] = best
                changed = True
    return min_len

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
                "sequence": " ".join(seq),
                "real_log_prob": log_prob
            }) + "\n")


def sample(grammar_name, max_len=MAX_SEQUENCE_LENGTH):
    tbl = GRAMMARS[grammar_name]

    while True:
        seq = []
        stack = ["S"]
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
            return seq, log_prob
        # Otherwise, seq hit max_len with stack nonempty → discard and retry


def sample_greedy(grammar_name, max_len=MAX_SEQUENCE_LENGTH):
    """
    Generate exactly one terminal sequence from grammar_name:
      • Randomly expand until (len(seq) + total_min_len(stack) >= max_len), then
      • Switch to greedy: pick the production that minimizes
         (rhs_min + leftover_stack_min), allowing seq to exceed max_len.
    Returns (seq_list_of_str, log_prob).
    """
    tbl = GRAMMARS[grammar_name]
    min_len_map = compute_min_len(tbl)

    # Precompute rhs_min for each nonterminal
    rhs_min = {}
    for NT, productions in tbl.items():
        rhs_mins = []
        for rhs, _prob in productions:
            s = 0
            for child in rhs:
                s += (min_len_map[child] if child in min_len_map else 1)
            rhs_mins.append(s)
        rhs_min[NT] = rhs_mins

    # Initialize stack and a running "stack_min_sum" = sum(min_len_map[sym] or 1)
    seq = []
    stack = ["S"]
    stack_min_sum = min_len_map["S"]      # because "S" is on the stack initially
    log_prob = 0.0
    greedy = False

    # main loop:
    while stack:
        # (A) Check if we need to switch to greedy mode
        if not greedy and (len(seq) + stack_min_sum >= max_len):
            greedy = True

        sym = stack.pop()
        # Subtract its min_len contribution
        stack_min_sum -= (min_len_map[sym] if sym in min_len_map else 1)

        if sym in tbl:
            # We have a nonterminal. Pull its productions and probs:
            prods, probs = zip(*tbl[sym])
            n_prods = len(prods)

            if not greedy:
                idx = random.choices(range(n_prods), weights=probs, k=1)[0]
                chosen_prod = prods[idx]
                chosen_prob = probs[idx]
            else:
                # greedy-finish: pick production minimizing (rhs_min + leftover_min)
                leftover_min = stack_min_sum
                best_i = None
                best_len = float("inf")
                best_prob = -1.0

                # Loop over productions once (all arithmetic is O(1) or O(len(rhs)) per production)
                for i in range(n_prods):
                    p = probs[i]
                    min_r = rhs_min[sym][i]
                    min_after = min_r + leftover_min

                    # We want the production that minimizes min_after; tie‐break by larger p
                    if (min_after < best_len) or (min_after == best_len and p > best_prob):
                        best_len = min_after
                        best_i = i
                        best_prob = p

                chosen_prod = prods[best_i]
                chosen_prob = probs[best_i]

            # Record log‐prob and push chosen_prod’s children onto stack
            log_prob += np.log(chosen_prob)
            for child in reversed(chosen_prod):
                stack.append(child)
                stack_min_sum += (min_len_map[child] if child in min_len_map else 1)

        else:
            # sym is a terminal symbol: always append if greedy or if seq < max_len
            if greedy or len(seq) < max_len:
                seq.append(sym)

    return seq, log_prob

def sample_many(grammar_name,n,max_len=MAX_SEQUENCE_LENGTH):
    return [sample(grammar_name,max_len) for _ in range(n)]

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

# can remove this later
def train_custom_tokenizer(corpus, tok_path, vocab=512):
    tok = Tokenizer(models.BPE())
    tok.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|bos|>", "<|eos|>"])
    tok.train_from_iterator(corpus, trainer=trainer)

    tok.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>", special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>")), ("<|eos|>", tok.token_to_id("<|eos|>"))]
    )
    tok.save(os.path.abspath(tok_path))
    return PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token="<|bos|>", eos_token="<|eos|>")


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
    train_sequences = sample_many(args.grammar, args.dataset_size, args.max_len)
    test_sequences = sample_many(args.grammar, 100, args.max_len)
    str_sequences = [" ".join(seq) for seq, _ in train_sequences]

    # 2) save dataset
    out_dir = f"../data/{args.grammar}/{args.grammar}_{args.dataset_size}"
    save_dataset(test_sequences, out_dir)

    # 3) BPE tokenisation
    tok_path = f"{out_dir}/tokenizer.json"
    tok_fast = build_fixed_tokenizer(args.grammar, tok_path)
    #tok_fast = train_custom_tokenizer(str_sequences, tok_path)

    split_and_tokenize(str_sequences, tok_fast, out_dir)

    count_valid_sequences(tok_fast, test_sequences, args.grammar)
    count_valid_sequences(tok_fast, train_sequences, args.grammar)


if __name__=="__main__":
    main()