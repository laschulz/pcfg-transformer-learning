import json
from nltk import Nonterminal, ViterbiParser
from generate_pcfg import PARSERS, sample, sample_many
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

RUN_PYTHON_GRAMMAR = False  # Set to True if you want to run the Python grammar analysis
DIVIDER = 600  # Used for normalizing KL divergence in plots

NT2COLOR = {
    "L0":       "#1f77b4",
    "L0_direct": "#fc03c2", 
    "L1_direct": "#ff7f0e",
    "L1":       "#ff7f0e",
    "L1_2":     "#d62728",
    "L1_2b":    "#d6275e", 
    "L1_3":     "#e377c2",
    "L1_3c":    "#9467bd", 
    "L2":       "#d62770", 
    "L2_2":     "#1ea054", 
    "L2_3":     "#1f77b4", 
    "L2_3b":    "#2B9E91", 
    "L2_3c":    "#892362",  
    "L4":       "#7f7f7f",
    "STMTS":    "#bcbd22",  # yellow
    "STMTS_direct": "#2B9E91",
    "overhead": "#25d1ec", 
    "uebergang_direct": "#17becf",  # cyan
}

def epoch_step_num(key: str) -> Tuple[int, int]:
    m = re.match(r"epoch_(\d+)(?:_(\d+))?\.pt$", key)
    epoch = int(m.group(1))
    step = int(m.group(2)) if m.group(2) else 0
    return epoch, step

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
    if spans == []:
        print("no spans found for", ' '.join(tokens))
    return spans

def seq_log_pcfg(parser: ViterbiParser, text: str) -> float:
    toks = text.split()
    parses = list(parser.parse(toks))
    return math.log(parses[0].prob())

def generate_from_nt(grammar_name: str, nt: str, num_sequences: int):
    start_symbol = Nonterminal(nt)  # <- start from the nonterminal directly
    sequences = []
    while len(sequences) < num_sequences:
        seq, _ = sample(grammar_name, start_symbol)
        sequences.append(seq)
    return sequences

def generate_test_subgrammar_set(grammar_name: str, nt: str, num_sequences: int):
    grammar_name = f"{grammar_name}" #_subgrammar" # we need the subgrammar version with the start and end markers
    grammar = GRAMMARS[grammar_name]
    start_symbol = list(grammar)[0]
    print(grammar_name)
    print(start_symbol)
    print(nt)

    sequences = []
    while len(sequences) < num_sequences: # TODO: change this
        seq, _ = sample(grammar_name, start_symbol, max_len=200)
        if f"s{nt.symbol()}" in seq and f"e{nt.symbol()}" in seq:
            sequences.append(seq)
    return sequences

def prepare_test_sequences(parser, nt, main_dir, top_level: bool, grammar_name: str):
    if top_level:
        with open(f"{main_dir}/test.jsonl", 'r') as f:
            test_sequences = [json.loads(line)["sequence"] for line in f]
    else:
        test_sequences = generate_test_subgrammar_set(grammar_name, nt, 1000)
        print(len(test_sequences), "is length of test sequences for subgrammar", nt.symbol())

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
    return test_sequences_with_probs, len(test_sequences_with_probs)

def prepare_overhead_sequences(grammar_name, start_symbol): # todo change here
    grammar = GRAMMARS[grammar_name]
    start_symbol = list(grammar)[0]
    test_sequences = sample_many(grammar_name, start_symbol, 1000, max_len=200)
    print(len(test_sequences), "is length of test sequences for overhead")

    relevant_test_sequences = []
    probabilities = []

    for seq, _ in test_sequences:
        tokens = seq.split()
        for i, token in enumerate(tokens):
            if token == "sL2_1" or token == "sL2_3" or token == "sL2_2":
                relevant_test_sequences.append((i, i, seq))
                probabilities.append(0.0)
                #probabilities.append(math.log(1.0))
        # also append EOS
        relevant_test_sequences.append((len(tokens), len(tokens), seq))
        probabilities.append(0.0)
    test_sequences_with_probs = list(zip(relevant_test_sequences, probabilities))
    return test_sequences_with_probs, len(relevant_test_sequences)

def prepare_eos_test_sequences(main_dir):
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

def analyze_hieararchy_all_epochs(grammar_name, nonTerminal, subgrammar,
                                  to_epoch, dataset_name, model_name, train_type, seed, prob_of_occuring = 1.0):

    # Initialize model and load results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(map_model_name(model_name)).to(device)

    main_dir = f"../data/{grammar_name}/{grammar_name}_{dataset_name}"

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{main_dir}/tokenizer.json",
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )

    # if "_" in dataset_name: # TODO: this is always the case
    #     #compound_stmt_count = 0
    #     test_sequences = []
    #     with open(f"{main_dir}/test.jsonl", "r") as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             obj = json.loads(line)
    #             tokens = obj["sequence"].split()
    #             #compound_stmt_count += tokens.count("compound_stmt")
    #             test_sequences.append(((0, len(tokens), obj["sequence"]), obj["real_log_prob"]))
    #     num_sequences = len(test_sequences)
    #     #print(compound_stmt_count)
    nt = Nonterminal(nonTerminal)
    if subgrammar == "overhead":
        test_sequences, num_sequences = prepare_overhead_sequences(grammar_name, nt)
        nonTerminal = "overhead"
        num_sequences = 1000
    elif subgrammar == "eos_test":
        test_sequences, num_sequences = prepare_eos_test_sequences(main_dir)
    else:
        parser = PARSERS[subgrammar]
        test_sequences, num_sequences = prepare_test_sequences(parser, nt, main_dir, subgrammar == grammar_name, grammar_name) # note: was subgrammar
    

    # Load master results file that contains all grammars
    master_results_path = "../results/hierarchy_analysis.json"
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            all_grammar_results = json.load(f)
    else:
        all_grammar_results = {}
    
    seed = str(seed)
    # Initialize grammar entry if it doesn't exist
    all_grammar_results.setdefault(model_name, {})
    all_grammar_results[model_name].setdefault(grammar_name, {})
    all_grammar_results[model_name][grammar_name].setdefault(seed, {})
    all_grammar_results[model_name][grammar_name][seed][nonTerminal] = {}

    # Load epoch 
    checkpoints_dir = f"{main_dir}/{model_name}/{train_type}/seed_{seed}"
    for ckpt in sorted(os.listdir(checkpoints_dir)):
        if not ckpt.endswith(".pt"): 
            continue 
        epoch_int = int(ckpt.split('_')[1].split('.')[0])
        if to_epoch and epoch_int > to_epoch:
            continue
        print(f"Analyzing epoch: {ckpt}")
        model.load_state_dict(
            torch.load(os.path.join(checkpoints_dir, ckpt), map_location=device)
        )

        diffs = compare_model_vs_real_probs_subgrammar(model, tokenizer, test_sequences, device)
        kl_divergence = sum([d['abs_logprob_diff'] for d in diffs]) / 1000 * prob_of_occuring
        
        # Add results for this epoch under the grammar's entry
        all_grammar_results[model_name][grammar_name][seed][nonTerminal][ckpt] = {
            "kl_divergence": kl_divergence,
            "non_terminal": nonTerminal,
            "subgrammar": subgrammar
        }
    
    # Write the updated master results file
    with open(master_results_path, 'w') as f:
        json.dump(all_grammar_results, f, indent=4)
    
    print(f"Updated hierarchy analysis results for {grammar_name} in {master_results_path}")
    return all_grammar_results
    
def plot_kl_accuracy(results_path: str, grammar_name: str, model_name: str, seed, to_epoch: int = None, for_paper: bool = False):
    """
    Generate separate line charts for KL divergence and accuracy per subgrammar, keeping distinct colors.
    """
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    seeds = [seed]
    plt.figure(figsize=(8, 5))
    for seed1 in seeds:
        grammar_data = all_results[model_name][grammar_name][f'{seed1}']

        nonterminals = list(grammar_data.keys())
        print(nonterminals)

        # Plot KL divergence
        for idx, nt in enumerate(nonterminals):
            data_points = []
            color = NT2COLOR[nt]
            #color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for ckpt, data in grammar_data[nt].items():
                e , s = epoch_step_num(ckpt)
                if to_epoch and e > to_epoch:
                    continue
                data_points.append((e, s, data['kl_divergence']))
            
            # Sort by epoch first, then by step
            data_points.sort(key=lambda x: (x[0], x[1]))
            
            if data_points:
                x_values = [e + s/DIVIDER for e, s, _ in data_points] 
                y_values = [kl for _, _, kl in data_points]
            
            if for_paper:
                if "_direct" in nt:
                    legend_label = "from scratch"
                else:
                    legend_label = "with pretraining"
            else:
                if nt == "L1" or nt == "overhead":
                    legend_label = nt
                else:
                    legend_label = f"subgrammar_{nt}" # f"{nt}_{seed1}"
                
            plt.plot(x_values, y_values, marker='o', linestyle='-', label=legend_label, color=color)
            
    plt.xlabel('Percentage of Training Data seen', fontsize=14)
    plt.ylabel('KL Divergence', fontsize=14)
    #plt.yscale('log')
    plt.title('Compositionality of KL Divergence', fontsize=18)
    #plt.title(f'KL Divergence over Epochs for {grammar_name}')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"../results/kl_divergence_plot_{model_name}_{grammar_name}.png")
    plt.close()

def plot_combined_kl_accuracy(results_path: str, grammar_model_pairs: List[Tuple[str, str, int]], to_epoch: int = None, for_paper: bool = False):
        """
        Generate a single line chart for KL divergence across multiple grammar-model pairs.
        """
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        plt.figure(figsize=(10, 6))
        
        for grammar_name, model_name, nt, seed in grammar_model_pairs:
                grammar_data = all_results[model_name][grammar_name][f'{seed}']

                data_points = []
                color = NT2COLOR.get(nt, "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
                
                for ckpt, data in grammar_data[nt].items():
                    e, s = epoch_step_num(ckpt)
                    if to_epoch and e > to_epoch:
                        continue
                    data_points.append((e, s, data['kl_divergence']))
                    
                data_points.sort(key=lambda x: (x[0], x[1]))
                    
                if data_points:
                    x_values = [e + s/DIVIDER for e, s, _ in data_points] 
                    y_values = [kl for _, _, kl in data_points]
                        
                    label = f"{model_name}-{grammar_name}-{nt}"
                    plt.plot(x_values, y_values, marker='o', linestyle='-', label=label, color=color)
        
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.title('KL Divergence over Epochs (Combined)')
        plt.grid(alpha=0.3)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(f"../results/combined_kl_divergence_plot.png")
        plt.close()

def collect_kl_table(results_path: str, grammar_name: str, model_name: str, nonTerminal: str):
   #load results from json
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    csv_path = "../results/kl_table.csv"
    if os.path.exists(csv_path):
        with open(csv_path, 'a') as f:
            f.write("\n")
    else:
        with open(csv_path, 'w') as f:
            f.write("nonTerminal,seed,kl_divergence\n")
    #iterate over all seeds for this grammar and model and loop over all seeds and take last epoch of this nonTerminal 
    seeds = all_results[model_name][grammar_name].keys()
    for seed in seeds:
        grammar_data = all_results[model_name][grammar_name][f'{seed}']
        nonterminals = list(grammar_data.keys())
        if nonTerminal in nonterminals:
            last_epoch = max(epoch_step_num(ckpt)[0] for ckpt in grammar_data[nonTerminal].keys())
            kl_value = grammar_data[nonTerminal][f'epoch_{last_epoch}_{0}.pt']['kl_divergence']
            with open(csv_path, 'a') as f:
                f.write(f"{nonTerminal},{seed},{kl_value}\n")
    return

# Update the main function to optionally generate plots
def main():
    args = argument_parser()
    if args.create_table:
        collect_kl_table("../results/hierarchy_analysis.json", args.grammar, args.model, args.nonTerminal)
        return
    if args.plot_only:
        plot_kl_accuracy("../results/hierarchy_analysis.json", args.grammar, args.model, args.seed, args.to_epoch)
    #     plot_combined_kl_accuracy(
    #     "../results/hierarchy_analysis.json", 
    #     [
    #         ("PythonPCFG_symbol", "TwoLayer", "STMTS", 2),
    #         ("PythonPCFG", "TwoLayer", "compound_stmt",2),
    #         ("PythonPCFG", "TwoLayer", "STMTS_direct", 2),
    #         ("PythonPCFG", "TwoLayer_LARGER", "STMTS_direct", 2),
    #     ],
    #     to_epoch=50
    # )
        return
    analyze_hieararchy_all_epochs(args.grammar, args.nonTerminal, args.subgrammar, args.to_epoch, args.dataset_name, args.model, args.train_type, args.seed, args.prob_of_occurring)     
    plot_kl_accuracy("../results/hierarchy_analysis.json", args.grammar, args.model, args.seed, args.to_epoch)

# Update argument parser to include plot-only option
def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze hierarchy in PCFG Transformer learning.")
    parser.add_argument("--grammar", type=str, required=True, help="The grammar to analyze.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Size of the dataset to analyze.")
    parser.add_argument("--plot_only", action='store_true', help="If set, only generate plots without analysis.")
    parser.add_argument("--nonTerminal", type=str, required=True)
    parser.add_argument("--subgrammar", type=str, required=True, help="Subgrammar to use for analysis.")
    parser.add_argument("--to_epoch", type=int, default=None, help="Number of epochs to analyze.")
    parser.add_argument("--model", type=str, default="FourLayer", help="Type of GPT model to use")
    parser.add_argument("--train_type", type=str, default="new", help="Type of training to analyze (new, continued)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--create_table", action='store_true', help="If set, create a KL divergence table.")
    parser.add_argument("--prob_of_occurring", type=float, default=1.0, help="Probability of subgrammar occuring")

    return parser.parse_args()
        
if __name__ == "__main__":
    main()