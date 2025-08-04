import json
from collections import defaultdict
from nltk import Nonterminal, ViterbiParser, Production, PCFG
from generate_pcfg import PARSERS, sample
import math, random 
import os
import torch
import torch.nn.functional as F
from model import GPT
from train import map_model_name
from transformers import PreTrainedTokenizerFast
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Set
from eval import compare_model_vs_real_probs_subgrammar
from def_pcfgs import GRAMMARS


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

def find_subsequences(tokens: List[str], C: Nonterminal, start_marker, end_marker) -> List[Tuple[int, int, str]]:
    """
    Find all subsequences in `tokens` delimited by start and end markers for nonterminal C.
    Assumes every start marker has a matching end marker and sequences are well-formed.

    Returns a list of (start_index, end_index, subseq_text), where end_index is exclusive.
    """

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

def generate_test_subgrammar_set(grammar_name: str, nt: str, num_sequences: int):
    print(grammar_name)
    grammar = GRAMMARS[grammar_name]
    start_symbol = list(grammar)[0]

    sequences = []
    while len(sequences) < num_sequences:
        seq, _ = sample(grammar_name, start_symbol)
        if f"s{nt.symbol()}" in seq and f"e{nt.symbol()}" in seq:
            sequences.append(seq)
    return sequences

def prepare_test_sequences(parser, nt, main_dir, top_level: bool, grammar_name: str):
    if top_level:
        with open(f"{main_dir}/test.jsonl", 'r') as f:
            test_sequences = [json.loads(line)["sequence"] for line in f]
    else:
        test_sequences = generate_test_subgrammar_set(grammar_name, nt, 500)

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
    elif subgrammar == "eos_test":
        test_sequences, num_sequences = prepare_eos_test_sequences(main_dir, tokenizer)
    else:
        parser = PARSERS[subgrammar]
        nt = Nonterminal(nonTerminal)
        test_sequences, num_sequences = prepare_test_sequences(parser, nt, main_dir, subgrammar == grammar_name, grammar_name)
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