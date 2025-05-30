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
        "A": [(["B", "B", "B"], .4), (["a"], .6)],
        "B": [(["[", "S", "]"], .5), (["b"], .5)],
        "X": [(["Y", '"', "Y", '"', "Y"], .6), (["x"], .4)],
        "Y": [(["S", "y", "S"], .4), (["S", "X"], .3), (["y"], .3)],
    },
}

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
    tokens = sequence.split() if isinstance(sequence, str) else list(sequence)
    tokens = [tok for tok in tokens if tok not in {"<|eos|>", "<|bos|>"}]

    try:
        return any(PARSERS[grammar_name].parse(tokens))
    except ValueError as err:          # token not covered by the grammar
        print(f"[VALIDATION ERROR - {grammar_name}] {tokens}\n{err}\n")
        return False

def save_dataset(seqs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/test.jsonl", "w") as f:
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

# def split_and_tokenize(jsonl_path, tok_fast, out_dir, train_ratio=.9):
#     os.makedirs(out_dir, exist_ok=True)
#     with open(jsonl_path) as f:
#         lines = [json.loads(l) for l in f]
#     split = int(len(lines)*train_ratio)
#     for name, data in [("train", lines[:split]), ("val", lines[split:])]:
#         ids = []
#         for row in data:
#             ids.extend(tok_fast.encode(tok_fast.bos_token + " " + row["sequence"] + " " + tok_fast.eos_token))
#         np.array(ids, dtype=np.uint32).tofile(f"{out_dir}/{name}.bin")
#         with open(f"{out_dir}/{name}.jsonl", "w") as j:
#             for row in data: j.write(json.dumps(row)+"\n")
#     with open(f"{out_dir}/meta.pkl", "wb") as m:
#         pickle.dump({"vocab_size": len(tok_fast),
#                     "bos_token_id": tok_fast.bos_token_id, 
#                     "eos_token_id": tok_fast.eos_token_id}, m)

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

# def train_custom_tokenizer(jsonl_path, tok_path, vocab=512):
#     with open(jsonl_path) as f:
#         corpus = [json.loads(l)["sequence"] for l in f]
#     tmp = os.path.join(os.path.dirname(tok_path), "tmp_corpus.txt")
#     with open(tmp, "w") as f: f.write("\n".join(corpus))
#     tok = Tokenizer(models.BPE())
#     tok.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
#     tok.pre_tokenizer = Whitespace()
#     tok.train([tmp], trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|bos|>", "<|eos|>"]))
#     tok.post_processor = TemplateProcessing(
#         single="<|bos|> $A <|eos|>", special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>")), ("<|eos|>", tok.token_to_id("<|eos|>"))]
#     )
#     tok.save(os.path.abspath(tok_path))

def train_custom_tokenizer(corpus, tok_path, vocab=512):
    tmp = os.path.join(os.path.dirname(tok_path), "tmp_corpus.txt")
    with open(tmp, "w") as f:
        f.write("\n".join(corpus))
    tok = Tokenizer(models.BPE())
    tok.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()
    tok.train([tmp], trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|bos|>", "<|eos|>"]))
    tok.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>", special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>")), ("<|eos|>", tok.token_to_id("<|eos|>"))]
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
    sequences = sample_many(args.grammar, args.dataset_size, args.max_len)
    test_sequences = sample_many(args.grammar, 100, args.max_len)
    str_sequences = [" ".join(seq) for seq in sequences]
    str_test_sequences = [" ".join(seq) for seq in test_sequences]

    # 2) save dataset
    out_dir = f"data/{args.grammar}/{args.grammar}_{args.dataset_size}"
    save_dataset(test_sequences, out_dir)


    # 3) optional BPE tokenisation
    tok_path = f"{out_dir}/tokenizer.json"
    train_custom_tokenizer(str_sequences, tok_path)
    tok_fast = PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token="<|bos|>", eos_token="<|eos|>")
    split_and_tokenize(str_sequences, tok_fast, out_dir)

if __name__=="__main__":
    main()