import json
from collections import defaultdict
from nltk import Nonterminal, ViterbiParser, Production, PCFG
from generate_pcfg import PARSERS
import math
import os
import torch
import torch.nn.functional as F
from model import GPT, FourLayer, TwoLayer, OneLayer, SixLayer
from train import map_model_name
from transformers import PreTrainedTokenizerFast
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Set
from eval import compare_model_vs_real_probs_subgrammar
from nltk.parse import pchart

NT2COLOR = {
    "L0":       "#1f77b4",
    "L1_direct": "#0373fc", 
    "L1":       "#ff7f0e", # orange
    "L1_2":     "#d62728",
    "L1_3":     "#e377c2",  # gray
    "L2":       "#2ca02c",  # red
    "L2_2":     "#0e582d",  # purple
    "L2_3":     "#8c564b",  # brown
    "L4":       "#7f7f7f",  # pink,
    "overhead": "#bcbd22",  # yellow
    "uebergang_direct": "#17becf",  # cyan
}

# def to_cnf(parser: ViterbiParser) -> Tuple[Dict[str, list], Set[Nonterminal], Nonterminal]:
#     """
#     Convert an NLTK PCFG into CNF, but *also* keep track of unit‐productions.
#     Returns:
#       - cnf_rules: a dict with three keys: "unary_term", "unit", and "binary"
#           • "unary": list of (A, terminal, prob) for rules A→"a"
#           • "unit":       list of (A, B, prob) for rules A→B  (both Nonterminals)
#           • "binary":     list of (A, B, C, prob) for rules A→B C
#       - nonterminals: set of all Nonterminal symbols
#       - start_symbol: the grammar’s start Nonterminal
#     """
#     grammar = parser.grammar()
#     cnf_rules = {
#         "unary": [],        # A → "a"  (terminal)
#         "unit": [],         # A → B    (unit production between nonterminals)
#         "binary": []        # A → B C
#     }
#     nonterminals: Set[Nonterminal] = set()
#     start_symbol = grammar.start()

#     # Step 1: Introduce temporary Nonterminals to replace terminals inside long RHS
#     temp_rules: Dict[Any, Nonterminal] = {}
#     next_temp_id = 0

#     # First pass: Create temp_NTs for any terminal that appears in a binary/multi rule
#     for production in grammar.productions():
#         lhs = production.lhs()
#         rhs = production.rhs()
#         nonterminals.add(lhs)

#         # If this RHS is length>1 and has a terminal, introduce a temp_NT
#         if len(rhs) > 1:
#             for symbol in rhs:
#                 if not isinstance(symbol, Nonterminal):
#                     if symbol not in temp_rules:
#                         temp_nt = Nonterminal(f"_TEMP{next_temp_id}")
#                         next_temp_id += 1
#                         temp_rules[symbol] = temp_nt
#                         nonterminals.add(temp_nt)
#                         # Add the “_TEMPx → terminal” as a unary‐term rule with prob = 1.0
#                         cnf_rules["unary"].append((temp_nt, symbol, 1.0))

#     # Second pass: turn every production into either unary, unit, or binary
#     for production in grammar.productions():
#         lhs = production.lhs()
#         rhs = production.rhs()
#         prob = production.prob()
#         nonterminals.add(lhs)

#         # CASE 1: A → terminal  (pure terminal rule)
#         if len(rhs) == 1 and not isinstance(rhs[0], Nonterminal):
#             cnf_rules["unary"].append((lhs, rhs[0], prob))

#         # CASE 2: A → B   (unit production: both sides are Nonterminal)
#         elif len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
#             cnf_rules["unit"].append((lhs, rhs[0], prob))

#         # CASE 3: A → B C   (both are Nonterminals, already CNF)
#         elif len(rhs) == 2 and all(isinstance(sym, Nonterminal) for sym in rhs):
#             cnf_rules["binary"].append((lhs, rhs[0], rhs[1], prob))

#         # CASE 4: A → B a  or A → a B  (mixed terminal + nonterminal, length = 2)
#         elif len(rhs) == 2:
#             new_rhs = []
#             for sym in rhs:
#                 if isinstance(sym, Nonterminal):
#                     new_rhs.append(sym)
#                 else:
#                     # replace terminal with its temp_NT
#                     new_rhs.append(temp_rules[sym])
#             cnf_rules["binary"].append((lhs, new_rhs[0], new_rhs[1], prob))

#         # CASE 5: A → X1 X2 X3 ... (length > 2)
#         else:  # len(rhs) > 2
#             # First replace any terminal in RHS with temp_NT
#             processed_rhs: List[Nonterminal] = []
#             for sym in rhs:
#                 if isinstance(sym, Nonterminal):
#                     processed_rhs.append(sym)
#                 else:
#                     processed_rhs.append(temp_rules[sym])

#             # Then binarize step by step
#             current_lhs = lhs
#             for i in range(len(processed_rhs) - 2):
#                 new_nt = Nonterminal(f"_BIN{next_temp_id}")
#                 next_temp_id += 1
#                 nonterminals.add(new_nt)

#                 # First binary: current_lhs → processed_rhs[i] new_nt
#                 # Use original prob only on the first introduced rule
#                 rule_prob = prob if i == 0 else 1.0
#                 cnf_rules["binary"].append(
#                     (current_lhs, processed_rhs[i], new_nt, rule_prob)
#                 )
#                 current_lhs = new_nt

#             # Final binary: current_lhs → second_last last
#             last_two = processed_rhs[-2:]
#             cnf_rules["binary"].append((current_lhs, last_two[0], last_two[1], 1.0))

#     return cnf_rules, nonterminals, start_symbol

def cyk_parse(tokens: List[str], cnf_rules, start_symbol, find_suffix=False) -> Dict:
    """
    Run CYK algorithm on tokens with the given grammar.
    
        Returns:
      - valid: Whether the full sequence is valid
      - rule_counts: Counter of rules used in parse
      - valid_length: Length of valid prefix/suffix
      - invalid_part: Invalid part of the sequence
    """

    N = len(tokens)
    
    # Initialize charts
    chart = [[{} for _ in range(N+1)] for _ in range(N+1)]
    back = [[{} for _ in range(N+1)] for _ in range(N+1)]
    
    # Fill in base cases (length 1 spans)
    for i in range(N):
        t = tokens[i]
        for (A, terminal, prob) in cnf_rules["unary"]:
            if terminal == t:
                # record A → t with probability=prob
                if A not in chart[i][i+1] or prob > chart[i][i+1][A]:
                    chart[i][i+1][A] = prob
                    back[i][i+1][A] = ("term", terminal)

        # apply unit‐closure in cell [i, i+1]
        queue = list(chart[i][i+1].keys())
        while queue:
            B = queue.pop()
            pB = chart[i][i+1][B]
            for (A, B_rhs, unit_prob) in cnf_rules["unit"]:
                if B_rhs == B:
                    new_p = unit_prob * pB
                    if A not in chart[i][i+1] or new_p > chart[i][i+1][A]:
                        chart[i][i+1][A] = new_p
                        back[i][i+1][A] = ("unit", B)
                        queue.append(A)
    
    # Fill in the chart for longer spans
    for length in range(2, N+1):
        for start in range(N-length+1):
            end = start + length
            for split in range(start+1, end):
                left_cell = chart[start][split]
                right_cell = chart[split][end]
                if not left_cell or not right_cell:
                    continue
                # for each binary rule A → B C
                for (A, B, C, bprob) in cnf_rules["binary"]:
                    if B in left_cell and C in right_cell:
                        new_p = bprob * left_cell[B] * right_cell[C]
                        if A not in chart[start][end] or new_p > chart[start][end][A]:
                            chart[start][end][A] = new_p
                            back[start][end][A] = ("binary", split, B, C)

            # after placing all binary‐derived nonterminals in [start,end], do unit‐closure
            if chart[start][end]:
                queue = list(chart[start][end].keys())
                while queue:
                    X = queue.pop()
                    pX = chart[start][end][X]
                    for (A, B_rhs, unit_prob) in cnf_rules["unit"]:
                        if B_rhs == X:
                            new_p = unit_prob * pX
                            if A not in chart[start][end] or new_p > chart[start][end][A]:
                                chart[start][end][A] = new_p
                                back[start][end][A] = ("unit", X)
                                queue.append(A)
    
    # Extract rule counts from the chart
    rule_counts = defaultdict(int)
    
    def extract_rules(i, j, symbol):
        if i == j-1:  # Terminal case
            bp = back[i][j].get(symbol)
            if bp and bp[0] == "term":
                rule = f"{symbol} -> {bp[1]}"
                rule_counts[rule] += 1
            elif bp and bp[0] == "unit":  # Handle unit productions at leaf level
                child = bp[1]
                # Only count rules without temporary nonterminals
                if not str(child).startswith('_'):
                    rule = f"{symbol} -> {child}"
                    rule_counts[rule] += 1
                    extract_rules(i, j, child)
                else:
                    # If child is temporary, still follow it but don't count the rule
                    extract_rules(i, j, child)
        else:  # Non-terminal case spanning multiple tokens
            bp = back[i][j].get(symbol)
            if bp and bp[0] == "binary":  
                _, split, left_sym, right_sym = bp
                # Only count rules without temporary nonterminals
                if not (str(left_sym).startswith('_') or str(right_sym).startswith('_')):
                    rule = f"{symbol} -> {left_sym} {right_sym}"
                    rule_counts[rule] += 1
                # Continue recursion even for temporary symbols
                extract_rules(i, split, left_sym)
                extract_rules(split, j, right_sym)
            elif bp and bp[0] == "unit":
                child = bp[1]
                # Only count rules without temporary nonterminals
                if not str(child).startswith('_'):
                    rule = f"{symbol} -> {child}"
                    rule_counts[rule] += 1
                    extract_rules(i, j, child)
                else:
                    # If child is temporary, still follow it but don't count the rule
                    extract_rules(i, j, child)
    
    # Find the longest valid span
    if find_suffix:
        # Find longest valid suffix: look for each [i,N] span where i is the start point
        valid_start = N  # Default: no valid suffix
        for i in range(N):
            if start_symbol in chart[i][N]:
                valid_start = i
                break  # Found the leftmost (longest) valid suffix
                
        valid_length = N - valid_start
        invalid_part = tokens[:valid_start] if valid_start > 0 else []
        
        # Extract rules only if we found a valid suffix
        if valid_length > 0:
            extract_rules(valid_start, N, start_symbol)
            
    else:
        # Find longest valid prefix (original behavior): look for each [0,j] span
        valid_length = 0
        for j in range(1, N+1):
            if start_symbol in chart[0][j]:
                valid_length = j
        
        invalid_part = tokens[valid_length:] if valid_length < N else []
        
        # Extract rules only if we found a valid prefix
        if valid_length > 0:
            extract_rules(0, valid_length, start_symbol)
    
    # Check if full sequence is valid
    valid = (valid_length == N)
    return {
        "valid": valid,
        "rule_counts": dict(rule_counts),
        "valid_length": valid_length,
        "invalid_part": invalid_part
    }

# Function to find non-overlapping longest valid substrings
def find_non_overlapping_longest(tokens, cnf_rules, C: Nonterminal):
    """
    Returns a list of (i, j, text) for the longest non-overlapping substrings
    tokens[i:j] valid under nonterminal C.
    """
    # Gather all valid spans
    hits = []
    N = len(tokens)
    for i in range(N):
        for j in range(i+1, N+1):
            substr = tokens[i:j]
            if cyk_parse(substr, cnf_rules, C)["valid"]:
                hits.append((i, j, " ".join(substr)))
    # Sort by length descending
    hits.sort(key=lambda x: x[1] - x[0], reverse=True)
    # Greedily select non-overlapping spans
    selected = []
    occupied = set()
    for i, j, text in hits:
        if any(pos in occupied for pos in range(i, j)):
            continue
        selected.append((i, j, text))
        for pos in range(i, j):
            occupied.add(pos)

    return selected, len(selected)

def find_subsequences(tokens: List[str], C: Nonterminal, start_marker, end_marker) -> List[Tuple[int, int, str]]:
    """
    Find all subsequences in `tokens` delimited by start and end markers for nonterminal C.
    Assumes every start marker has a matching end marker and sequences are well-formed.

    Returns a list of (start_index, end_index, subseq_text), where end_index is exclusive.
    """

    # start_marker = f"s{C.symbol()}"
    # end_marker = f"e{C.symbol()}"
    spans: List[Tuple[int, int, str]] = []

    i = 0
    N = len(tokens)
    while i < N:
        if tokens[i] == start_marker:
            # find the corresponding end marker
            for j in range(i + 1, N):
                if tokens[j] == end_marker:
                    # exclude markers
                    subseq = " ".join(tokens[i+1:j])
                    spans.append((i+1, j, subseq))
                    i = j  # skip ahead to end marker
                    break
        i += 1
    return spans

def seq_log_pcfg(parser: ViterbiParser, text: str) -> float:
    toks = text.split()
    parses = list(parser.parse(toks))
    return math.log(parses[0].prob())

def analyze_hierarchy_per_epoch(sequences, cnf_rules, C: Nonterminal, invalid_terminals):
    total_count = 0
    valid_count = 0
    selected_texts = []
    for seq in sequences:
        tokens = seq.split()

        spans, count = find_non_overlapping_longest(tokens, cnf_rules, C)
        total_count += count
        for i, j, text in spans:
            valid = False
            if i == 1 or j == len(tokens):
                valid_count += 1
                continue
            if tokens[i-1] not in invalid_terminals:
                valid = True
            if tokens[j] in invalid_terminals:
                valid = valid and True
            if valid:
                valid_count += 1

            selected_texts.extend(text for (_,_,text) in spans) 

    return valid_count, total_count, selected_texts

# def get_terminals_for_nonterminal(grammar_rules):
#     terminals = set()
#     for rhs, terminal, prob in grammar_rules['unary']:
#         terminals.add(terminal)
#     return terminals

def prepare_test_sequences(parser, nt, main_dir, top_level: bool):
    with open(f"{main_dir}/test.jsonl", 'r') as f:
        test_sequences = [json.loads(line)["sequence"] for line in f]

    relevant_test_sequences = []
    probabilities = []

    for seq in test_sequences:
        tokens = seq.split()
        spans = find_subsequences(tokens, nt, f"s{nt.symbol()}", f"e{nt.symbol()}") if not top_level else [(0, len(tokens), seq)]
        for start, end, subseq in spans:
            prob = seq_log_pcfg(parser, subseq)
            relevant_test_sequences.append((start, end, seq))
            probabilities.append(prob)

    test_sequences_with_probs = list(zip(relevant_test_sequences, probabilities))
    return test_sequences_with_probs, len(relevant_test_sequences)

def prepare_overhead_sequences(main_dir):
    with open(f"{main_dir}/test.jsonl", 'r') as f:
        test_sequences = [json.loads(line)["sequence"] for line in f]

    relevant_test_sequences = []
    probabilities = []

    for seq in test_sequences:
        tokens = seq.split()
        for i, token in enumerate(tokens):
            if token == "sL2" or token == "sL2_3":
                relevant_test_sequences.append((i, i, seq))
                #probabilities.append(0.0)
                probabilities.append(math.log(0.5))
        # also append EOS
        relevant_test_sequences.append((len(tokens), len(tokens), seq))
        probabilities.append(0.0)
    test_sequences_with_probs = list(zip(relevant_test_sequences, probabilities))
    return test_sequences_with_probs, len(relevant_test_sequences)

def prepare_eos_test_sequences(main_dir, tokenizer):
    with open(f"{main_dir}/test.jsonl", 'r') as f:
        test_sequences = [json.loads(line)["sequence"] for line in f]

    relevant_test_sequences = []
    probabilities = []

    for seq in test_sequences:
        tokens = seq.split()
        for i, token in enumerate(tokens):
            if token == "eL2" and tokens[i+1] == "sL2_3":
                relevant_test_sequences.append((i, i+1, seq))
                probabilities.append(0.0)
                break
    test_sequences_with_probs = list(zip(relevant_test_sequences, probabilities))
    return test_sequences_with_probs, len(relevant_test_sequences)

def analyze_hieararchy_all_epochs(grammar_name, nonTerminal, subgrammar, to_epoch, dataset_size, model_name, train_type):

    # Initialize model and load results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(map_model_name(model_name)).to(device)

    main_dir = f"../data/{grammar_name}/{grammar_name}_{dataset_size}"
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{main_dir}/tokenizer.json",
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )
 
    if subgrammar == "overhead":
        test_sequences, num_sequences = prepare_overhead_sequences(main_dir)
        num_sequences = 500
    elif subgrammar == "eos_test":
        test_sequences, num_sequences = prepare_eos_test_sequences(main_dir, tokenizer)
    else:
        parser = PARSERS[subgrammar]
        nt = Nonterminal(nonTerminal)
        test_sequences, num_sequences = prepare_test_sequences(parser, nt, main_dir, subgrammar == grammar_name)
    print(num_sequences)


    # Load master results file that contains all grammars
    master_results_path = "../results/hierarchy_analysis.json"
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            all_grammar_results = json.load(f)
    else:
        all_grammar_results = {}
    
    # Initialize grammar entry if it doesn't exist
    if model_name not in all_grammar_results:
        all_grammar_results[model_name] = {}

    if grammar_name not in all_grammar_results[model_name]:
        all_grammar_results[model_name][grammar_name] = {}

    if nonTerminal not in all_grammar_results[model_name][grammar_name]:
        all_grammar_results[model_name][grammar_name][nonTerminal] = {}

    # Load epoch 
    checkpoints_dir = f"{main_dir}/{model_name}/{train_type}"
    for ckpt in sorted(os.listdir(checkpoints_dir)):
        if not ckpt.endswith(".pt"): 
            continue 
        epoch_int = int(ckpt.split('_')[1].split('.')[0])  # Extract epoch number from filename
        if to_epoch and epoch_int > to_epoch:
            continue
        print(f"Analyzing epoch: {ckpt}")
        model.load_state_dict(
            torch.load(os.path.join(checkpoints_dir, ckpt), map_location=device)
        )

        diffs = compare_model_vs_real_probs_subgrammar(model, tokenizer, test_sequences, device)
        kl_divergence = sum([d['abs_logprob_diff'] for d in diffs]) / num_sequences 
        rel_kl_divergence = kl_divergence * num_sequences / 500  #TODO: this only works for non-recursive top grammar

        # Add results for this epoch under the grammar's entry
        if ckpt not in all_grammar_results[model_name][grammar_name][nonTerminal]:
            all_grammar_results[model_name][grammar_name][nonTerminal][ckpt] = {}
            
        all_grammar_results[model_name][grammar_name][nonTerminal][ckpt] = {
            "kl_divergence": kl_divergence,
            "rel_kl_divergence": rel_kl_divergence,
            "non_terminal": nonTerminal,
            "subgrammar": subgrammar
        }
    
    # Write the updated master results file
    with open(master_results_path, 'w') as f:
        json.dump(all_grammar_results, f, indent=4)
    
    print(f"Updated hierarchy analysis results for {grammar_name} in {master_results_path}")
    
def plot_kl_accuracy(results_path: str, grammar_name: str, model_name: str,to_epoch: int = None):
    """
    Generate separate line charts for KL divergence and accuracy per subgrammar, keeping distinct colors.
    """
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    grammar_data = all_results[model_name][grammar_name]

    # Helper to extract epoch number
    def epoch_num(key: str) -> int:
        m = re.search(r'(\d+)', key)
        return int(m.group(1)) if m else 0

    nonterminals = list(grammar_data.keys())

    # Plot KL divergence
    plt.figure(figsize=(8, 5))
    for idx, nt in enumerate(nonterminals):
        epochs, kl_vals = [], []
        color = NT2COLOR[nt]
        for ckpt, data in grammar_data[nt].items():
            e = epoch_num(ckpt)
            if to_epoch and e > to_epoch:
                continue
            epochs.append(e)
            kl_vals.append(data['kl_divergence'])
        order = np.argsort(epochs)
        epochs_arr = np.array(epochs)[order]
        kl_arr = np.array(kl_vals)[order]
        plt.plot(epochs_arr, kl_arr, marker='o', linestyle='-', label=nt, color=color)
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title(f'KL Divergence over Epochs for {grammar_name}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../results/kl_divergence_plot_{model_name}_{grammar_name}.png")


def plot_subsequence_lengths(results_path: str, grammar_name: str, to_epoch: int = None):
    """
    Generate a grouped bar chart of the number of subsequences per epoch for each subgrammar, with distinct colors.
    """
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    grammar_data = all_results[grammar_name]

    # Helper to extract epoch number
    def epoch_num(key: str) -> int:
        m = re.search(r'(\d+)', key)
        return int(m.group(1)) if m else 0

    nonterminals = list(grammar_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(nonterminals)))

    # Gather epochs
    epochs_set = set()
    counts_by_nt = {nt: {} for nt in nonterminals}
    for idx, nt in enumerate(nonterminals):
        for ckpt, data in grammar_data[nt].items():
            e = epoch_num(ckpt)
            if to_epoch and e > to_epoch:
                continue
            epochs_set.add(e)
            counts_by_nt[nt][e] = data.get('total_count', 0)
    epochs = sorted(epochs_set)

    # Plot grouped bars
    x = np.array(epochs)
    width = 0.8 / len(nonterminals)
    plt.figure(figsize=(10, 5))
    for idx, nt in enumerate(nonterminals):
        counts = [counts_by_nt[nt].get(e, 0) for e in epochs]
        positions = x + (idx - len(nonterminals)/2 + 0.5) * width
        plt.bar(positions, counts, width=width, label=nt, color=colors[idx])
    plt.xlabel('Epoch')
    plt.ylabel('Number of Subsequences')
    plt.title(f'Subsequence Counts per Epoch for {grammar_name}')
    plt.xticks(epochs)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../results/subsequence_counts_plot_{grammar_name}.png")


# Update the main function to optionally generate plots
def main():
    args = argument_parser()
    if args.plot_only:
        plot_kl_accuracy("../results/hierarchy_analysis.json", args.grammar, args.model, args.to_epoch)
        return
    analyze_hieararchy_all_epochs(args.grammar, args.nonTerminal, args.subgrammar, args.to_epoch, args.dataset_size, args.model, args.train_type)     
    plot_kl_accuracy("../results/hierarchy_analysis.json", args.grammar, args.model, args.to_epoch)

# Update argument parser to include plot-only option
def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze hierarchy in PCFG Transformer learning.")
    parser.add_argument("--grammar", type=str, required=True, help="The grammar to analyze.")
    parser.add_argument("--dataset_size", type=int, required=True, help="Size of the dataset to analyze.")
    parser.add_argument("--plot_only", action='store_true', help="If set, only generate plots without analysis.")
    parser.add_argument("--nonTerminal", type=str, required=True, help="Number of epochs to analyze.")
    parser.add_argument("--subgrammar", type=str, required=True, help="Subgrammar to use for analysis.")
    parser.add_argument("--to_epoch", type=int, default=None, help="Number of epochs to analyze.")
    parser.add_argument("--model", type=str, choices=["TwoLayer", "FourLayer", "SixLayer", "OneLayer"], default="FourLayer", help="Type of GPT model to use")
    parser.add_argument("--train_type", type=str, default="new", help="Type of training to analyze (new, continued)")

    return parser.parse_args()
        
if __name__ == "__main__":
    main()