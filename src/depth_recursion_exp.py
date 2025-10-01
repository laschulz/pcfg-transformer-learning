from model import GPT
import torch
import torch.nn.functional as F
import argparse
from transformers import PreTrainedTokenizerFast
import matplotlib.pyplot as plt
import numpy as np

from generate_pcfg import generate_pcfg
from train import map_model_name, trainer

def generate_sequences(case, n, prefix=""):
    sequences = []
    
    if case == 1:
        # Pattern: "( a ) ( a ) ... ( a ) ("
        for i in range(0, n+1):
            sequences.append("( a ) " * i + "(")
            
    elif case == 2:
        # Pattern: "(, ((, ((( ..."
        for i in range(1, 230): 
            sequences.append("( " * i)
    elif case == 3:
        for i in range(1, 230):
            sequences.append(prefix + " ( " * i)
    
    sequences = [seq.strip() for seq in sequences]
    sequences = [seq.replace("  ", " ") for seq in sequences]
    return sequences
    
def ground_truth_logit(tokenizer):
    """Generates ground truth logits for nested parenthesis grammar."""
    probs = torch.zeros(1, 5)  # Match dimensions with model output
    vocab = tokenizer.get_vocab()
    
    probs[0, vocab["("]] = 0.8
    probs[0, vocab["a"]] = 0.2

    return probs

def get_logits(model, tokenizer, sequences, device):
    model.eval()
    results = []

    with torch.no_grad():
        for seq in sequences:
            encoded = tokenizer.encode(seq, return_tensors="pt").to(device)
            encoded = encoded[:,:-1]

            # Forward pass
            logits, _ = model(encoded)
            probs = F.softmax(logits.squeeze(1), dim=-1)
            results.append(probs.detach().cpu())
    return results


def analyze_case(model, tokenizer, gt_logit,case, prefix, seeds, path, to_epoch, device):
    diffs = {}
    for s in seeds:
        checkpoint_path = f'{path}/TwoLayer_LARGE/new/seed_{s}/epoch_{to_epoch}_0.pt'
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        model.eval()
        sequences = generate_sequences(case, 80, prefix)
        logits = get_logits(model, tokenizer, sequences, device)

        diffs[s] = []
        for l in logits:
            diff_vec = torch.abs(l - gt_logit)
            diff = diff_vec.sum().item()
            diffs[s].append(diff)

    plot_logit_differences(sequences, diffs, case, prefix)


def plot_logit_differences(sequences, diffs, case, prefix):
    """
    Plot the differences between model logits and ground truth logits
    with increasing sequence depth for multiple seeds.
    
    Draw individual seeds in gray and average across seeds as main line.
    Seeds are labeled at the right side of each seed line.
    
    Args:
        sequences: List of input sequences
        diffs: Dictionary of seed -> list of differences
        case: Case number for the filename
        plateau_start: Depth where plateau begins
    """
    # Prepare x-axis labels (sequence depths or indices)
    x = list(range(len(sequences)))
    
    plt.figure(figsize=(4.5, 3))
    
    # Calculate average across all seeds
    valid_seeds = [seed for seed in diffs.keys() if seed != 3]
    all_values = np.array([diffs[seed] for seed in valid_seeds])
    avg_values = np.mean(all_values, axis=0)
    
    # Plot individual seed lines in gray first
    for i, seed in enumerate(valid_seeds):
        y = np.array(diffs[seed])
        line = plt.plot(x, y, color='#80A0DB', alpha=0.5, linewidth=1)
    
    # Plot the average line as bold and colored
    plt.plot(x, avg_values, color="#002FD8", linewidth=2.5, label='Average')
    
    # Highlight plateau region only if case == 2
    if case == 1 or case == 2 or case == 3:
        if case == 1:
            plateau_start = 70
            plt.xlim(-2, len(x)+13)
            tick_positions = [i for i in x if i % 20 == 0]
        if case == 2:
            plateau_start = 220
            #plt.axvspan(plateau_start, len(x)-1, color="lightgray", alpha=0.2, label="Plateau region")
            plt.xlim(-5, len(x)+33)
            tick_positions = [i for i in x if i % 40 == 0]
        elif case == 3:
            plateau_start = 220
            #plt.axvspan(plateau_start, len(x)-1, color="lightgray", alpha=0.2, label="Plateau region")
            plt.xlim(-5, len(x)+30)
            tick_positions = [i for i in x if i % 40 == 0]
        
        # Add plateau mean line for average
        if plateau_start < len(avg_values):
            plateau_mean = np.mean(avg_values[plateau_start:])
            plt.hlines(plateau_mean, xmin=plateau_start, xmax=len(x)-1,
                      colors='blue', linestyles="dashed")
            plt.text(len(x), plateau_mean, f"{plateau_mean:.3f}",
                    color='blue', va="center", fontweight='bold')
    
    plt.xlabel('Sequence Depth')
    # plt.yticks([]) # uncomment to hide y-axis ticks
    plt.ylabel('Prediction Error')
    plt.legend(loc='upper left')
    plt.xticks()
    plt.yticks()
    
    # Add ticks only every 5 steps
    tick_labels = [f"{i}" for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.ylim(0, 0.25)
    
    plt.tight_layout()
    save_path = f'../results/logit_differences_case_{case}_{prefix}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--to_epoch', type=int, default=5)
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--generate', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    config = map_model_name("TwoLayer_LARGE")
    model = GPT(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "../data/nestedParentheses/nestedParentheses_50000_L4"

    # 1. generate tokenizer
    generate_pcfg("nestedParentheses", "L4", 50000, 200, None)
    tokenizer_path = f'{path}/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token="<|bos|>", eos_token="<|eos|>")

    # 2. train models
    for s in args.seeds:
        trainer(model, "nestedParentheses", config, "50000_L4", None, 
                5, args.to_epoch, 0, device, s, safe_only_last=True)
    
    gt_logit = ground_truth_logit(tokenizer).to(device)

    # 3. analyze all cases
    analyze_case(model, tokenizer, gt_logit, 1, "", args.seeds, path, args.to_epoch, device)
    analyze_case(model, tokenizer, gt_logit, 2, "", args.seeds, path, args.to_epoch, device)
    analyze_case(model, tokenizer, gt_logit, 3, "( a ) ( a ) ( a ) ( a ) ( a ) ( a )", args.seeds, path, args.to_epoch, device)
    analyze_case(model, tokenizer, gt_logit, 3, "( ( ( ( ( ( ( ( ( a ) ) ) ) ) ) ) ) )", args.seeds, path, args.to_epoch, device)
    analyze_case(model, tokenizer, gt_logit, 3, "( a ) ( a ) ( a ) ) ( a a ) ( a ) ( a )", args.seeds, path, args.to_epoch, device)

    if args.prefix: # in case further prefices are to be analyzed
        analyze_case(model, tokenizer, gt_logit, 3, args.prefix, args.seeds, path, args.to_epoch, device)

    if args.generate:
        pass
        # if args.generate:
        #     for i, seq in enumerate(sequences):
        #         seq_new = seq + " a ) ( a ) ( ( ( a )"
        #         print(seq_new)
        #         idx = tokenizer.encode(seq_new, return_tensors="pt").to(device)
        #         idx = idx[:,:-1] # remove eos token
        #         print(idx)
        #         generated_text = model.generate(idx, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id, eos_prob_threshold=0.95)
        #         # how to decode this correctly?
        #         ids = generated_text[0][0]
        #         print(ids)
        #         text_decoded = tokenizer.decode(ids, skip_special_tokens=True)
        #         print(text_decoded)

        #         c = text_decoded.count("(")
        #         q = text_decoded.count(")")
        #         print(c - q)
        #         break
        #     return 

if __name__ == "__main__":
    main()