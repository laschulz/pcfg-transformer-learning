import json
from nltk import Nonterminal, ViterbiParser
from analysis2 import to_cnf, cyk_parse
from generate_pcfg import PARSERS
import math
import os
import torch
import torch.nn.functional as F
from model import GPT, FourLayer
from transformers import PreTrainedTokenizerFast
import argparse

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

# Get logits for tokens in a sequence
def get_sequence_token_logits(model, sequence, vocab):
    """Get logits for each token in the sequence"""
    # Convert sequence to tensor
    tokens = sequence.split()

    # Create input tensor
    input_ids = torch.tensor([[vocab.get(t, 0) for t in tokens]])

    # Get model output
    with torch.no_grad():
        logits, _ = model(input_ids, full_logits=True)
    
    # Get log probabilities
    log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
    
    # Extract logits for observed tokens
    token_logits = []
    for i in range(len(tokens)-1):  # -1 because we predict the next token
        next_token = tokens[i+1]
        next_token_id = vocab.get(next_token, 0)
        token_logit = log_probs[0, i, next_token_id].item()
        token_logits.append(token_logit)
    
    return token_logits

# Get logits for a specific subsequence within a full sequence
def get_subsequence_logits_in_context(model, full_sequence, start_idx, end_idx, vocab):
    """Get logits for tokens in a subsequence within the context of the full sequence"""
    # Get full sequence logits
    full_logits = get_sequence_token_logits(model, full_sequence, vocab)
    
    # Extract logits for the subsequence (offset by 1 since we predict next token)
    subseq_logits = full_logits[start_idx:end_idx]
    
    # Calculate total log probability for the subsequence
    total_log_prob = sum(subseq_logits)
    
    return {
        "token_logits": subseq_logits,
        "log_prob": total_log_prob,
        "prob": math.exp(total_log_prob)
    }

def seq_log_pcfg(parser: ViterbiParser, text: str) -> float:
    toks   = text.split()
    parses = list(parser.parse(toks))
    return math.log(parses[0].prob())

def analyze_hierarchy_per_epoch(sequences, cnf_rules, C: Nonterminal, parser, model, vocab):
    total_count = 0
    valid_count = 0
    diffs = 0
    selected_texts = []
    for seq in sequences:
        tokens = seq.split()

        spans, count = find_non_overlapping_longest(tokens, cnf_rules, C)
        total_count += count
        for i, j, text in spans:
            valid = False
            if tokens[i-1] not in {"cond", "not", "and"}:
                valid = True
            if j<len(tokens)-1 and tokens[j+1] in {"cond", "not", "and"}:
                valid = valid and True
            if valid:
                valid_count += 1

            lp_pcfg = seq_log_pcfg(parser, text)
            lp_neural = get_subsequence_logits_in_context(model, seq, i, j, vocab)
            diffs += abs(lp_neural["log_prob"] - lp_pcfg)

            selected_texts.extend(text for (_,_,text) in spans) 
    return diffs, valid_count, total_count, selected_texts

def analyze_hieararchy_all_epochs(grammar_name):
    # Looking into Conditionals subgrammar
    parser = PARSERS["Conditionals"]
    cnf_rules, _ , _ = to_cnf(parser)
    C = Nonterminal("C")

    # Initialize model and load results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(FourLayer()).to(device)
    with open("../results/results_log.json") as f:
        all_results = json.load(f)
    sequences_all_epochs = all_results[grammar_name]

    main_dir = f"../data/{grammar_name}/{grammar_name}_1000"
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{main_dir}/tokenizer.json",
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )
    vocab = tokenizer.get_vocab()

    results_path = f"../results/hierarchy_analysis_{grammar_name}.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results_log = json.load(f)
    else:
        results_log = {}
    
    # Load epoch 
    checkpoints_dir = f"{main_dir}/FourLayer"
    for ckpt in sorted(os.listdir(checkpoints_dir)):
        if not ckpt.endswith(".pt"): #or ckpt in results_log[pcfg]:
            continue  # Skip already analyzed checkpoints
        print(f"Analyzing epoch: {ckpt}")
        model.load_state_dict(
            torch.load(os.path.join(checkpoints_dir, ckpt), map_location=device)
        )

        # I hope ckpt = epoch
        sequences = sequences_all_epochs[ckpt]["generated_sequences"]
        diffs, valid_count, total_count, selected_texts = analyze_hierarchy_per_epoch(sequences, cnf_rules, C, parser, model, vocab)

        
        results_log[ckpt] = {
            "diffs": diffs,
            "valid_count": valid_count,
            "total_count": total_count,
            "selected_texts": selected_texts
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_log, f, indent=4)

def plot_subgrammar(grammar_name):
    """
    Plot KL divergence between neural model and PCFG predictions across epochs.
    Also plot accuracy (valid/total) and number of subsequences as secondary metrics.
    
    Args:
        grammar_name: Name of the grammar to plot results for
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import re
    
    # Load results
    results_path = f"../results/hierarchy_analysis_{grammar_name}.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract epoch numbers and metrics
    epochs = []
    diffs = []
    accuracies = []
    valid_counts = []  # Add this to track the number of valid subsequences
    total_counts = []  # Add this to track the total number of subsequences
    
    # Sort checkpoints by epoch number
    def extract_epoch_num(epoch_str):
        # Extract number from strings like "epoch_10.pt"
        match = re.search(r'epoch_(\d+)', epoch_str)
        return int(match.group(1)) if match else 0
    
    # Sort the keys by epoch number
    sorted_keys = sorted(results.keys(), key=extract_epoch_num)
    
    for key in sorted_keys:
        # Extract epoch number
        match = re.search(r'epoch_(\d+)', key)
        epoch_num = int(match.group(1))
        
        # Get metrics
        data = results[key]
        diff = data["diffs"]
        valid_count = data["valid_count"]
        total_count = data["total_count"]
        accuracy = valid_count / total_count if total_count > 0 else 0
        
        # Normal case: add data point
        epochs.append(epoch_num)
        diffs.append(diff)
        accuracies.append(accuracy)
        valid_counts.append(valid_count)
        total_counts.append(total_count)
    
    # Create two subplot figures - one for differences/accuracy and one for subsequence counts
    fig, (ax1, ax4) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    
    # Setup the first plot (differences and accuracy) - similar to before
    ax2 = ax1.twinx()
    
    # Plot difference on the primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Difference (KL Divergence)', color=color1)
    line1 = ax1.plot(epochs, diffs, marker='o', linestyle='-', color=color1, label='KL Divergence')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot accuracy on the secondary y-axis
    color2 = 'tab:red'
    ax2.set_ylabel('Accuracy (valid/total)', color=color2)
    line2 = ax2.plot(epochs, accuracies, marker='s', linestyle='--', color=color2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set percentage format for accuracy axis
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # Add average difference per valid sample
    avg_diffs = [diff/valid for diff, valid in zip(diffs, valid_counts)]
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
    ax3.set_ylabel('Avg Diff per Sample', color='tab:green')
    line3 = ax3.plot(epochs, avg_diffs, marker='^', linestyle='-.', color='tab:green', label='Avg Diff/Sample')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    
    # Update legend for first plot
    lines1 = line1 + line2 + line3
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Add gridlines to first plot
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Model Performance vs. PCFG - {grammar_name}')
    
    # Now create the second plot for subsequence counts
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Number of Subsequences')
    
    # Plot only total subsequences (removing valid subsequences bar)
    bar_width = 0.4  # Made wider since we only have one bar now
    bar_positions1 = np.array(epochs)
    
    bars1 = ax4.bar(bar_positions1, total_counts, bar_width, label='Total Subsequences', color='darkblue', alpha=0.7)
    
    # Add value labels above bars (only for total counts)
    for i, v in enumerate(total_counts):
        ax4.text(bar_positions1[i], v + 0.5, str(v), ha='center', fontsize=8)
        
    # Add legend and gridlines to second plot
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_title(f'Subsequence Counts per Epoch - {grammar_name}')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"../results/hierarchy_plot_{grammar_name}.png", dpi=300, bbox_inches='tight')
    
    # Create a separate plot just for total subsequence counts
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bar_positions1, total_counts, bar_width, label='Total Subsequences', color='darkblue', alpha=0.7)
        
# Add value labels for total counts only
    for i, v in enumerate(total_counts):
        ax.text(bar_positions1[i], v + 0.5, str(v), ha='center')
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of Subsequences')
    ax.set_title(f'Subsequence Counts per Epoch - {grammar_name}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"../results/subsequence_counts_{grammar_name}.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to ../results/hierarchy_plot_{grammar_name}.png and ../results/subsequence_counts_{grammar_name}.png")

# Update the main function to optionally generate plots
def main():
    args = argument_parser()
    if args.plot_only:
        plot_subgrammar(args.grammar)
        return
    analyze_hieararchy_all_epochs(args.grammar)
    plot_subgrammar(args.grammar)

# Update argument parser to include plot-only option
def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze hierarchy in PCFG Transformer learning.")
    parser.add_argument("--grammar", type=str, required=True, help="The grammar to analyze.")
    parser.add_argument("--plot_only", action='store_true', help="If set, only generate plots without analysis.")
    return parser.parse_args()
        
if __name__ == "__main__":
    main()