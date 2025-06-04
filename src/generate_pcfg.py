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
        "S": [(["A"], .5), (["B"], .4), (["c"], .1)],
        "A": [(["a"], .3), (["A", "aa"], .7)],
        "B": [(["b"], .2), (["B", "b"], .8)],
    },

    # G3  ─ MutualRecursion
    "MutualRecursion": {
        "S": [(["A"], 1.0)],
        "A": [(["a", "B"], .8), (["a"], .2)],
        "B": [(["b", "A"], .8), (["b"], .2)],
    },

    # G4  ─ CenterEmbedding
    "CenterEmbedding": {
        "S": [(["a", "S", "b"], .8), (["c"], .2)],
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
        "S": [(["(", "A", ")"], .5), (["*", "X", "X", "*"], .5)], #change this
        "A": [(["B", "B"], .4), (["a"], .6)],
        "B": [(["[", "S", "]"], .3), (["b"], .7)],
        "X": [(["Y", '"', "Y"], .6), (["x"], .4)],
        "Y": [(["S", "y", "S"], .15), (["S", "X"], .15), (["y"], .7)],
    },
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

    try:
        is_valid = any(PARSERS[grammar_name].parse(tokens))
    except ValueError as err:          # token not covered by the grammar
        print(f"[VALIDATION ERROR - {grammar_name}] {tokens}\n{err}\n")
        return False

    encoded_ids = tokenizer.encode(' '.join(tokens), add_special_tokens=False)

    if len(encoded_ids) > 126:
        return False
    
    return is_valid

def save_dataset(test_pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/test.jsonl", "w") as f:
        for seq, log_prob in test_pairs:
            f.write(json.dumps({
                "sequence": " ".join(seq),
                "real_log_prob": log_prob
            }) + "\n")

# def sample(grammar_name, max_len=MAX_SEQUENCE_LENGTH):
#     tbl = GRAMMARS[grammar_name]
#     seq = []
#     stack = ['S']
#     log_prob = 0.0

#     # 1) Expand normally until you either empty the stack or hit max_len
#     while stack and len(seq) < max_len:
#         sym = stack.pop()
#         if sym in tbl:
#             prods, probs = zip(*tbl[sym])
#             # Random sampling as long as we're safely below max_len
#             chosen_prod = random.choices(prods, probs)[0]
#             stack.extend(reversed(chosen_prod))
#             log_prob += np.log(probs[prods.index(chosen_prod)])
#         else:
#             seq.append(sym)

#     # 2) Greedy “force‐finish” for any remaining non‐terminals
#     #    Continue until stack is empty, always picking the highest‐prob terminal‐producing
#     #    rule if possible; otherwise highest‐prob overall. Append terminals immediately.
#     while stack:
#         sym = stack.pop()
#         if sym in tbl:
#             prods, probs = zip(*tbl[sym])

#             # Look for productions that yield only terminals
#             terminal_prods = []
#             term_probs = []
#             for p, p_prob in zip(prods, probs):
#                 if all((child not in tbl) for child in p):
#                     terminal_prods.append(p)
#                     term_probs.append(p_prob)

#             if terminal_prods:
#                 # pick highest‐prob purely‐terminal production
#                 term_idx = term_probs.index(max(term_probs))
#                 chosen_prod = terminal_prods[term_idx]
#                 chosen_prob = term_probs[term_idx]
#             else:
#                 # no purely‐terminal choice—pick the highest‐prob production overall
#                 overall_idx = probs.index(max(probs))
#                 chosen_prod = prods[overall_idx]
#                 chosen_prob = probs[overall_idx]

#             log_prob += np.log(chosen_prob)
#             # push children of chosen_prod back onto stack (in reverse order)
#             # so that we keep expanding until only terminals remain
#             for child in reversed(chosen_prod):
#                 stack.append(child)
#         else:
#             # Already a terminal—append immediately
#             seq.append(sym)

#     return seq, log_prob


def sample(grammar_name, max_len=MAX_SEQUENCE_LENGTH):
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

# def train_custom_tokenizer(corpus, tok_path, vocab=512):
#     tmp = os.path.join(os.path.dirname(tok_path), "tmp_corpus.txt")
#     with open(tmp, "w") as f:
#         f.write("\n".join(corpus))
#     tok = Tokenizer(models.BPE())
#     tok.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
#     tok.pre_tokenizer = Whitespace()
#     tok.train([tmp], trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|bos|>", "<|eos|>"]))
#     tok.post_processor = TemplateProcessing(
#         single="<|bos|> $A <|eos|>", special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>")), ("<|eos|>", tok.token_to_id("<|eos|>"))]
#     )
#     tok.save(os.path.abspath(tok_path))
#     return tok

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
    tok_fast = train_custom_tokenizer(str_sequences, tok_path)

    split_and_tokenize(str_sequences, tok_fast, out_dir)

    count_valid_sequences(tok_fast, test_sequences, args.grammar)
    count_valid_sequences(tok_fast, train_sequences, args.grammar)

if __name__=="__main__":
    main()