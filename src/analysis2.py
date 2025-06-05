from collections import defaultdict
from nltk import Tree
from nltk.grammar import PCFG
from nltk.parse import ViterbiParser
import json
import argparse

from generate_pcfg import PARSERS

def count_rules_in_tree(tree: Tree, rule_counts: dict):
    """
    Recursively count rule usage in a parse tree.
    """
    if isinstance(tree, Tree) and len(tree) > 0:
        lhs = tree.label()
        rhs = [child.label() if isinstance(child, Tree) else child for child in tree]
        rule = f"{lhs} -> {' '.join(rhs)}"
        rule_counts[rule] += 1
        for child in tree:
            if isinstance(child, Tree):
                count_rules_in_tree(child, rule_counts)

def analyze_sequences(sequences, pcfg):
    """
    Analyze a list of terminal-only sequences using a PCFG.
    Returns:
      - rule_counts: dict of how often each rule was applied
      - invalid_seqs: list of sequences that couldn't be parsed
    """
    rule_counts = defaultdict(int)
    invalid_seqs = []

    for seq in sequences:
        tokens = seq.split() if isinstance(seq, str) else seq
        try:
            parses = list(pcfg.parse(tokens))
            if not parses:
                invalid_seqs.append(tokens)
                continue
            best_parse = parses[0]
            count_rules_in_tree(best_parse, rule_counts)
        except ValueError:
            invalid_seqs.append(tokens)

    return rule_counts, invalid_seqs

def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze sequences with a PCFG")
    parser.add_argument("--grammar", type=str, required=True, help="Path to the PCFG grammar file")
    parser.add_argument("--epoch", type=str, required=True, help="Epoch number for analysis")
    return parser.parse_args()


def main():
    args = argument_parser()


    pcfg = PARSERS[args.grammar]
    epoch = args.epoch

    results_path = "../results/results_log.json"
    with open(results_path, "r") as f:
        all_results = json.load(f)
    sequences = all_results[args.grammar][epoch]["generated_sequences"]

    rule_counts, invalid_seqs = analyze_sequences(sequences, pcfg)
    print("Rule counts:", rule_counts)
    print("Invalid sequences:", invalid_seqs)

if __name__ == "__main__":
    main()