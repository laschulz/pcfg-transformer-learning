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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

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
    full_logits = get_sequence_token_logits(model, "EOS " + full_sequence, vocab) # Add EOS at the beginning
    
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

def analyze_hierarchy_per_epoch(sequences, cnf_rules, C: Nonterminal, parser, model, vocab, invalid_terminals):
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
            if i == 1 or j == len(tokens):
                valid_count += 1
                continue
            if tokens[i-1] not in invalid_terminals:
                valid = True
            if tokens[j] in invalid_terminals:
                valid = valid and True
            if valid:
                valid_count += 1

            lp_pcfg = seq_log_pcfg(parser, text)
            lp_neural = get_subsequence_logits_in_context(model, seq, i, j, vocab)
            diffs += abs(lp_neural["log_prob"] - lp_pcfg)

            selected_texts.extend(text for (_,_,text) in spans) 
    return diffs, valid_count, total_count, selected_texts

def get_terminals_for_nonterminal(grammar_rules):
    terminals = set()
    for rhs, terminal, prob in grammar_rules['unary']:
        terminals.add(terminal)
    return terminals

def analyze_hieararchy_all_epochs(grammar_name, nonTerminal, subgrammar, to_epoch):
    # Looking into Conditionals subgrammar
    parser = PARSERS[subgrammar]
    cnf_rules, _ , _ = to_cnf(parser)
    C = Nonterminal(nonTerminal)
    terminal_list = get_terminals_for_nonterminal(cnf_rules)
    print(terminal_list)

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

    # Load master results file that contains all grammars
    master_results_path = "../results/hierarchy_analysis.json"
    if os.path.exists(master_results_path):
        with open(master_results_path, 'r') as f:
            all_grammar_results = json.load(f)
    else:
        all_grammar_results = {}
    
    # Initialize grammar entry if it doesn't exist
    if grammar_name not in all_grammar_results:
        all_grammar_results[grammar_name] = {}
    
    if nonTerminal not in all_grammar_results[grammar_name]:
        all_grammar_results[grammar_name][nonTerminal] = {}
    
    # Load epoch 
    checkpoints_dir = f"{main_dir}/FourLayer"
    for ckpt in sorted(os.listdir(checkpoints_dir)):
        if not ckpt.endswith(".pt"): #or ckpt in all_grammar_results.get(grammar_name, {}).get(ckpt, False):
            continue 
        epoch_int = int(ckpt.split('_')[1].split('.')[0])  # Extract epoch number from filename
        if to_epoch and epoch_int > to_epoch:
            continue
        print(f"Analyzing epoch: {ckpt}")
        model.load_state_dict(
            torch.load(os.path.join(checkpoints_dir, ckpt), map_location=device)
        )

        sequences = sequences_all_epochs[ckpt]["generated_sequences"]
        diffs, valid_count, total_count, selected_texts = analyze_hierarchy_per_epoch(sequences, cnf_rules, C, parser, model, vocab, terminal_list)

        # Add results for this epoch under the grammar's entry
        if ckpt not in all_grammar_results[grammar_name][nonTerminal]:
            all_grammar_results[grammar_name][nonTerminal][ckpt] = {}
            
        all_grammar_results[grammar_name][nonTerminal][ckpt] = {
            "diffs": diffs,
            "valid_count": valid_count,
            "total_count": total_count,
            "selected_texts": selected_texts,
            "non_terminal": nonTerminal,  # Store which nonterminal was used
            "subgrammar": subgrammar      # Store which subgrammar was used
        }
    
    # Write the updated master results file
    with open(master_results_path, 'w') as f:
        json.dump(all_grammar_results, f, indent=4)
    
    print(f"Updated hierarchy analysis results for {grammar_name} in {master_results_path}")

# might not need this
# def plot_subgrammar(grammar_name, nonTerminal):
#     """
#     Plot KL divergence between neural model and PCFG predictions across epochs.
#     Also plot accuracy (valid/total) and number of subsequences as secondary metrics.
    
#     Args:
#         grammar_name: Name of the grammar to plot results for
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker
#     import numpy as np
#     import re
    
#     # Load results
#     results_path = f"../results/hierarchy_analysis.json"
#     with open(results_path, 'r') as f:
#         results = json.load(f)
    
#     results = results[grammar_name][nonTerminal]
    
#     # Extract epoch numbers and metrics
#     epochs = []
#     diffs = []
#     accuracies = []
#     valid_counts = []  # Add this to track the number of valid subsequences
#     total_counts = []  # Add this to track the total number of subsequences
    
#     # Sort checkpoints by epoch number
#     def extract_epoch_num(epoch_str):
#         # Extract number from strings like "epoch_10.pt"
#         match = re.search(r'epoch_(\d+)', epoch_str)
#         return int(match.group(1)) if match else 0
    
#     # Sort the keys by epoch number
#     sorted_keys = sorted(results.keys(), key=extract_epoch_num)
    
#     for key in sorted_keys:
#         # Extract epoch number
#         match = re.search(r'epoch_(\d+)', key)
#         epoch_num = int(match.group(1))
        
#         # Get metrics
#         data = results[key]
#         diff = data["diffs"]
#         valid_count = data["valid_count"]
#         total_count = data["total_count"]
#         accuracy = valid_count / total_count if total_count > 0 else 0
        
#         # Normal case: add data point
#         epochs.append(epoch_num)
#         diffs.append(diff)
#         accuracies.append(accuracy)
#         valid_counts.append(valid_count)
#         total_counts.append(total_count)
    
#     # Create two subplot figures - one for differences/accuracy and one for subsequence counts
#     fig, (ax1, ax4) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    
#     # Setup the first plot (differences and accuracy) - similar to before
#     ax2 = ax1.twinx()
    
#     # Plot difference on the primary y-axis
#     color1 = 'tab:blue'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Difference (KL Divergence)', color=color1)
#     line1 = ax1.plot(epochs, diffs, marker='o', linestyle='-', color=color1, label='KL Divergence')
#     ax1.tick_params(axis='y', labelcolor=color1)
    
#     # Plot accuracy on the secondary y-axis
#     color2 = 'tab:red'
#     ax2.set_ylabel('Accuracy (valid/total)', color=color2)
#     line2 = ax2.plot(epochs, accuracies, marker='s', linestyle='--', color=color2, label='Accuracy')
#     ax2.tick_params(axis='y', labelcolor=color2)
    
#     # Set percentage format for accuracy axis
#     ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
#     # Add average difference per valid sample
#     avg_diffs = [diff/valid for diff, valid in zip(diffs, valid_counts)]
#     ax3 = ax1.twinx()
#     ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
#     ax3.set_ylabel('Avg Diff per Sample', color='tab:green')
#     line3 = ax3.plot(epochs, avg_diffs, marker='^', linestyle='-.', color='tab:green', label='Avg Diff/Sample')
#     ax3.tick_params(axis='y', labelcolor='tab:green')
    
#     # Update legend for first plot
#     lines1 = line1 + line2 + line3
#     labels1 = [l.get_label() for l in lines1]
#     ax1.legend(lines1, labels1, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
#     # Add gridlines to first plot
#     ax1.grid(True, alpha=0.3)
#     ax1.set_title(f'Model Performance vs. PCFG - {grammar_name}')
    
#     # Now create the second plot for subsequence counts
#     ax4.set_xlabel('Epoch')
#     ax4.set_ylabel('Number of Subsequences')
    
#     # Plot only total subsequences (removing valid subsequences bar)
#     bar_width = 0.4  # Made wider since we only have one bar now
#     bar_positions1 = np.array(epochs)
    
#     bars1 = ax4.bar(bar_positions1, total_counts, bar_width, label='Total Subsequences', color='darkblue', alpha=0.7)
    
#     # Add value labels above bars (only for total counts)
#     for i, v in enumerate(total_counts):
#         ax4.text(bar_positions1[i], v + 0.5, str(v), ha='center', fontsize=8)
        
#     # Add legend and gridlines to second plot
#     ax4.legend(loc='upper right')
#     ax4.grid(True, alpha=0.3, axis='y')
#     ax4.set_title(f'Subsequence Counts per Epoch - {grammar_name}_{nonTerminal}')
    
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig(f"../results/hierarchy_plot_{grammar_name}_{nonTerminal}.png", dpi=300, bbox_inches='tight')
    
#     # Create a separate plot just for total subsequence counts
#     fig2, ax = plt.subplots(figsize=(10, 5))
#     ax.bar(bar_positions1, total_counts, bar_width, label='Total Subsequences', color='darkblue', alpha=0.7)
        
# # Add value labels for total counts only
#     for i, v in enumerate(total_counts):
#         ax.text(bar_positions1[i], v + 0.5, str(v), ha='center')
        
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Number of Subsequences')
#     ax.set_title(f'Subsequence Counts per Epoch - {grammar_name} - nonTerminal {nonTerminal}')
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
#     plt.savefig(f"../results/subsequence_counts_{grammar_name}_{nonTerminal}.png", dpi=300)
#     plt.show()
#     plt.close()
    
def plot_subgrammar(grammar_name, to_epoch):
    """
    Plot KL divergence between neural model and PCFG predictions across epochs
    for multiple nonterminals.
    
    Args:
        grammar_name: Name of the grammar to plot results for
    """
    
    # Load results
    results_path = f"../results/hierarchy_analysis.json"
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    # Fixed colors for each metric
    kl_div_color = '#0072B2'    # Blue
    accuracy_color = '#D55E00'  # Red/orange
    avg_diff_color = '#009E73'  # Green
    
    # Use different line styles and markers for nonterminals
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # Create the main figure with two subplots
    fig, (ax1, ax4) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
    
    # Create secondary axes for the top plot
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
    
    # For storing all lines for the legend
    all_lines = []
    all_labels = []
    
    # Common function to extract epoch numbers
    def extract_epoch_num(epoch_str):
        match = re.search(r'epoch_(\d+)', epoch_str)
        return int(match.group(1)) if match else 0
    
    # To store bar positions and heights for the subplot
    bar_data = {}
    all_epochs = set()
    
    # Process each nonterminal
    grammar_data = all_results[grammar_name]
    nonTerminals = list(grammar_data.keys())
    
    # For bar chart - use different colors for nonterminals
    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(nonTerminals)))
    
    for nt_idx, nonTerminal in enumerate(nonTerminals):
        # Get the data for this nonterminal
        nt_data = grammar_data[nonTerminal]
        
        # Extract metrics for this nonterminal
        epochs = []
        diffs = []
        accuracies = []
        valid_counts = []
        total_counts = []
        
        # Sort epoch keys by epoch number
        sorted_keys = sorted(nt_data.keys(), key=extract_epoch_num)
        
        for epoch_key in sorted_keys:
            match = re.search(r'epoch_(\d+)', epoch_key)
            epoch_num = int(match.group(1))
            if to_epoch and epoch_num > to_epoch:
                continue
                
            # Get metrics for this epoch
            data = nt_data[epoch_key]
            diff = data["diffs"]
            valid_count = data["valid_count"]
            total_count = data["total_count"]
            accuracy = valid_count / total_count if total_count > 0 else 0
            
            # Add data points
            epochs.append(epoch_num)
            all_epochs.add(epoch_num)
            diffs.append(diff)
            accuracies.append(accuracy)
            valid_counts.append(valid_count)
            total_counts.append(total_count)
            
            # Store bar data for the subplot
            if epoch_num not in bar_data:
                bar_data[epoch_num] = {}
            bar_data[epoch_num][nonTerminal] = total_count
        
        # Calculate average diff per sample
        avg_diffs = [d/v for d, v in zip(diffs, valid_counts) if v > 0]
        
        # Select line style and marker for this nonterminal (cycle through available options)
        style = line_styles[nt_idx % len(line_styles)]
        marker = markers[nt_idx % len(markers)]
        
        # Plot KL divergence - Always blue, varying line styles and markers
        line1 = ax1.plot(epochs, diffs, marker=marker, linestyle=style, color=kl_div_color, 
                        label=f'KL Div ({nonTerminal})')
        all_lines.extend(line1)
        all_labels.extend([f'KL Div ({nonTerminal})'])
        
        # Plot accuracy - Always red/orange, varying line styles and markers
        line2 = ax2.plot(epochs, accuracies, marker=marker, linestyle=style, color=accuracy_color,
                        label=f'Acc ({nonTerminal})')
        all_lines.extend(line2)
        all_labels.extend([f'Acc ({nonTerminal})'])
        
        # Plot avg diff per sample - Always green, varying line styles and markers
        if avg_diffs:  # Only if we have valid data points
            line3 = ax3.plot(epochs[:len(avg_diffs)], avg_diffs, marker=marker, linestyle=style,
                            color=avg_diff_color, label=f'Avg Diff ({nonTerminal})')
            all_lines.extend(line3)
            all_labels.extend([f'Avg Diff ({nonTerminal})'])
    
    # Configure the first plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Difference (KL Divergence)', color=kl_div_color)
    ax1.tick_params(axis='y', labelcolor=kl_div_color)
    
    ax2.set_ylabel('Accuracy (valid/total)', color=accuracy_color)
    ax2.tick_params(axis='y', labelcolor=accuracy_color)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    ax3.set_ylabel('Avg Diff per Sample', color=avg_diff_color)
    ax3.tick_params(axis='y', labelcolor=avg_diff_color)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Model Performance vs. PCFG - {grammar_name}')
    
    # Add legend for the first plot - organize by metrics and nonterminals
    # Create custom legend elements to group by metric type
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Group 1: KL Divergence (blue)
    for nt_idx, nonTerminal in enumerate(nonTerminals):
        style = line_styles[nt_idx % len(line_styles)]
        marker = markers[nt_idx % len(markers)]
        legend_elements.append(Line2D([0], [0], color=kl_div_color, marker=marker, linestyle=style,
                                      label=f'KL Div ({nonTerminal})'))
    
    # Group 2: Accuracy (red/orange)
    for nt_idx, nonTerminal in enumerate(nonTerminals):
        style = line_styles[nt_idx % len(line_styles)]
        marker = markers[nt_idx % len(markers)]
        legend_elements.append(Line2D([0], [0], color=accuracy_color, marker=marker, linestyle=style,
                                      label=f'Acc ({nonTerminal})'))
    
    # Group 3: Avg Diff (green)
    for nt_idx, nonTerminal in enumerate(nonTerminals):
        style = line_styles[nt_idx % len(line_styles)]
        marker = markers[nt_idx % len(markers)]
        legend_elements.append(Line2D([0], [0], color=avg_diff_color, marker=marker, linestyle=style,
                                      label=f'Avg Diff ({nonTerminal})'))
    
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=min(3, len(nonTerminals)), fontsize='small')
    
    # Now create the bar chart for subsequence counts - still use different colors for nonterminals
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Number of Subsequences')
    
    # Sort all epochs
    all_epochs = sorted(all_epochs)
    bar_width = 0.8 / len(nonTerminals)
    
    # Plot bars for each nonterminal
    for nt_idx, nonTerminal in enumerate(nonTerminals):
        positions = []
        counts = []
        
        for epoch in all_epochs:
            if epoch in bar_data and nonTerminal in bar_data[epoch]:
                positions.append(epoch + (nt_idx - len(nonTerminals)/2 + 0.5) * bar_width)
                counts.append(bar_data[epoch][nonTerminal])
        
        color = bar_colors[nt_idx]
        bars = ax4.bar(positions, counts, bar_width, label=nonTerminal, 
                      color=color, alpha=0.7)
        
        # Add value labels above bars
        for i, v in enumerate(counts):
            ax4.text(positions[i], v + 0.5, str(v), ha='center', fontsize=8)
    
    # Configure the second plot
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_title(f'Subsequence Counts per Epoch - {grammar_name}')
    
    # Set x-ticks to be at the actual epoch numbers
    ax4.set_xticks(all_epochs)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"../results/hierarchy_plot_{grammar_name}.png", dpi=300, bbox_inches='tight')
    
    # # Create a separate plot just for total subsequence counts
    # fig2, ax = plt.subplots(figsize=(10, 5))
    
    # # Plot bars for each nonterminal
    # for nt_idx, nonTerminal in enumerate(nonTerminals):
    #     positions = []
    #     counts = []
        
    #     for epoch in all_epochs:
    #         if epoch in bar_data and nonTerminal in bar_data[epoch]:
    #             positions.append(epoch + (nt_idx - len(nonTerminals)/2 + 0.5) * bar_width)
    #             counts.append(bar_data[epoch][nonTerminal])
        
    #     color = bar_colors[nt_idx]
    #     ax.bar(positions, counts, bar_width, label=nonTerminal, 
    #           color=color, alpha=0.7)
        
    #     # Add value labels
    #     for i, v in enumerate(counts):
    #         ax.text(positions[i], v + 0.5, str(v), ha='center', fontsize=8)
    
    # # Configure the separate plot
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Number of Subsequences')
    # ax.set_title(f'Subsequence Counts per Epoch - {grammar_name}')
    # ax.legend()
    # ax.grid(True, alpha=0.3, axis='y')
    # ax.set_xticks(all_epochs)
    
    # plt.tight_layout()
    # plt.savefig(f"../results/subsequence_counts_{grammar_name}.png", dpi=300)
    # plt.show()
    # plt.close()

# Update the main function to optionally generate plots
def main():
    args = argument_parser()
    if args.plot_only:
        plot_subgrammar(args.grammar)
        return
    analyze_hieararchy_all_epochs(args.grammar, args.nonTerminal, args.subgrammar, args.to_epoch)
    plot_subgrammar(args.grammar, args.to_epoch)

# Update argument parser to include plot-only option
def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze hierarchy in PCFG Transformer learning.")
    parser.add_argument("--grammar", type=str, required=True, help="The grammar to analyze.")
    parser.add_argument("--plot_only", action='store_true', help="If set, only generate plots without analysis.")
    parser.add_argument("--nonTerminal", type=str, required=True, help="Number of epochs to analyze.")
    parser.add_argument("--subgrammar", type=str, required=True, help="Subgrammar to use for analysis.")
    parser.add_argument("--to_epoch", type=int, default=None, help="Number of epochs to analyze.")

    return parser.parse_args()
        
if __name__ == "__main__":
    main()