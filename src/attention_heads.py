import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import GPT
from train import map_model_name
from generate_pcfg import sample_many, GRAMMARS
from transformers import PreTrainedTokenizerFast
import argparse
import os
import glob
import json

"""
Right now, this is comparing general sequences to subgrammar sequences (subgrammar only, no stuff around). This allow us to see a ratio.
Is this the best way to do it? Probably now but let's see. 
"""

TESTSET_SIZE = 1000
MAX_LEN = 200

def extract_attention_patterns(model, tokenizer, sequences, device):
    """
    Extract per-head attention weights for each input sequence.
    Returns: Dict[str, Dict[layer_idx, Dict[head_idx, [T,T] tensor]]]
    """
    import torch

    model.eval()
    attention_patterns = {}

    # Resolve attention blocks once and register hooks on them.
    attn_blocks = []
    for layer_idx, block in enumerate(model.transformer.h):
        attn_mod = getattr(block, "attn", None) or getattr(block, "self_attn", None)
        if attn_mod is not None:
            attn_blocks.append((layer_idx, attn_mod))

    # We'll store the "current sequence key" so hooks know which input they're handling.
    current_seq_key = {"value": None}  # mutable holder so closure can read updates

    def make_hook(layer_idx):
        def hook(module, inp, out):
            seq_key = current_seq_key["value"]
            if seq_key is None:
                return

            # Try to obtain attention weights in a few robust ways:
            weights = None

            # 1) If the module caches them (recommended minimal patch)
            if hasattr(module, "last_attn"):
                weights = module.last_attn

            # 2) Or, if the module returns them in `out` (some impls do)
            if weights is None and isinstance(out, (tuple, list)):
                for t in out:
                    if torch.is_tensor(t) and t.ndim == 4 and t.shape[-1] == t.shape[-2]:
                        weights = t
                        break

            if weights is None:
                # Couldn’t find attention weights; silently skip.
                # If you want to enforce, raise here with a clear message.
                return

            # weights: [B, H, T, T]
            att = weights.detach().cpu()
            if seq_key not in attention_patterns:
                attention_patterns[seq_key] = {}
            if layer_idx not in attention_patterns[seq_key]:
                attention_patterns[seq_key][layer_idx] = {}

            H = att.shape[1]
            # Store each head’s [T, T] map (use batch 0)
            for h in range(H):
                attention_patterns[seq_key][layer_idx][h] = att[0, h]
        return hook

    hooks = [mod.register_forward_hook(make_hook(li)) for li, mod in attn_blocks]

    # Normalize input list (accept dicts with "sequence" or plain strings)
    def to_text(x):
        if isinstance(x, dict):
            if "sequence" in x:
                return x["sequence"]
            if "text" in x:
                return x["text"]
            raise KeyError(f"Expected 'sequence' or 'text' in dict, got keys: {list(x.keys())}")
        elif isinstance(x, str):
            return x
        else:
            raise TypeError(f"Unsupported sequence type: {type(x)}")

    with torch.no_grad():
        for item in sequences:
            seq = to_text(item)
            # Encode using your tokenizer; grab only ids and pass positionally.
            enc = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)  # [1, T] typically

            # If your model expects [T] rather than [1, T], uncomment:
            # if ids.dim() == 2 and ids.size(0) == 1:
            #     ids = ids[0]

            current_seq_key["value"] = seq
            _ = model(ids)  # POSITIONAL call, not kwargs

    for h in hooks:
        h.remove()

    return attention_patterns


def analyze_attention_for_subgrammar(model, tokenizer, general_sequences, subgrammar_sequences, 
                                     output_dir, subgrammar_name, device):
    """
    Analyze attention patterns to see if specific heads respond more to subgrammar sequences.
    """

    def head_stat(attn_matrix, type="entropy"):
        if type == "entropy":
            p = attn_matrix.clamp_min(1e-12)
            ent = (-p * p.log()).sum(dim=-1).mean().item()
            return -ent  # higher = more focused
        elif type == "mean":
            T = attn_matrix.size(0)
            # Collect attention from each position i onto i-1
            diag_vals = [attn_matrix[i, i - 1].item() for i in range(1, T)]
            return float(np.mean(diag_vals)) if diag_vals else 0.0
        elif type == "topk":
            return attn_matrix.topk(3, dim=-1).values.mean().item()
        else:
            return attn_matrix.topk(3, dim=-1).values.mean().item()

    # Extract attention patterns
    general_patterns = extract_attention_patterns(model, tokenizer, general_sequences, device)
    subgrammar_patterns = extract_attention_patterns(model, tokenizer, subgrammar_sequences, device)

    # Compare attention patterns
    # 1. For each layer and head, compute average attention weight
    general_avg = {}
    subgrammar_avg = {}
    
    # Process general sequences
    for seq_key, layers in general_patterns.items():
        for layer_idx, heads in layers.items():
            if layer_idx not in general_avg:
                general_avg[layer_idx] = {}
            
            for head_idx, attn_matrix in heads.items():
                if head_idx not in general_avg[layer_idx]:
                    general_avg[layer_idx][head_idx] = []
                
                # Use mean attention weight as the metric
                general_avg[layer_idx][head_idx].append(head_stat(attn_matrix))
    
    # Process subgrammar sequences
    for seq_key, layers in subgrammar_patterns.items():
        for layer_idx, heads in layers.items():
            if layer_idx not in subgrammar_avg:
                subgrammar_avg[layer_idx] = {}
            
            for head_idx, attn_matrix in heads.items():
                if head_idx not in subgrammar_avg[layer_idx]:
                    subgrammar_avg[layer_idx][head_idx] = []
                
                # Use mean attention weight as the metric
                subgrammar_avg[layer_idx][head_idx].append(head_stat(attn_matrix))
    
    # Compute average for each head
    for layer_idx in general_avg:
        for head_idx in general_avg[layer_idx]:
            general_avg[layer_idx][head_idx] = np.mean(general_avg[layer_idx][head_idx])
    
    for layer_idx in subgrammar_avg:
        for head_idx in subgrammar_avg[layer_idx]:
            subgrammar_avg[layer_idx][head_idx] = np.mean(subgrammar_avg[layer_idx][head_idx])
    
    # Calculate attention ratio (subgrammar / general)
    # Higher values indicate heads that respond more to the subgrammar
    attention_ratio = {}
    for layer_idx in subgrammar_avg:
        if layer_idx not in attention_ratio:
            attention_ratio[layer_idx] = {}
            
        for head_idx in subgrammar_avg[layer_idx]:
            general_val = general_avg.get(layer_idx, {}).get(head_idx, 1e-6)
            subgrammar_val = subgrammar_avg[layer_idx][head_idx]
            
            attention_ratio[layer_idx][head_idx] = subgrammar_val / general_val
    
    # Plot the ratio as a heatmap
    plot_attention_ratios(attention_ratio, output_dir, subgrammar_name)
    
    # Return the attention ratios for further analysis
    return attention_ratio

def plot_attention_ratios(attention_ratio, output_dir, subgrammar_name):
    """Plot attention ratios as a heatmap."""
    # Convert the nested dictionary to a 2D numpy array
    layers = sorted(attention_ratio.keys())
    n_layers = len(layers)
    n_heads = max(len(attention_ratio[layer]) for layer in layers)
    
    ratio_matrix = np.zeros((n_layers, n_heads))
    for i, layer in enumerate(layers):
        for head, ratio in attention_ratio[layer].items():
            ratio_matrix[i, head] = ratio
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(ratio_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Head {i}" for i in range(n_heads)],
                yticklabels=[f"Layer {i}" for i in layers])
    plt.title(f"Attention Ratio for {subgrammar_name} (Higher = More Specific to Subgrammar)")
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{subgrammar_name}_attention_ratio.png"), dpi=300)
    plt.close()
    
    # Also plot with a more extreme color scale to highlight differences
    plt.figure(figsize=(12, 8))
    sns.heatmap(ratio_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=1.0,
                xticklabels=[f"Head {i}" for i in range(n_heads)],
                yticklabels=[f"Layer {i}" for i in layers])
    plt.title(f"Attention Ratio for {subgrammar_name} (Red = More Specific to Subgrammar)")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{subgrammar_name}_attention_ratio_centered.png"), dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze attention patterns for subgrammars")
    parser.add_argument("--model", type=str, required=True, help="Model architecture name")
    parser.add_argument("--grammar", type=str, required=True, help="The grammar to analyze")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to base directory where tokenizer is")
    parser.add_argument("--subgrammar", type=str, required=True, help="Name of the subgrammar being analyzed")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # find the latest epoch and then load this
    checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'epoch_*.pt'))

    if checkpoint_files:
        # Extract epoch numbers and find the largest one
        latest_checkpoint = max(checkpoint_files, 
                            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    model_path = latest_checkpoint     
    
    # load model
    model = GPT(map_model_name(args.model)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    for block in model.transformer.h:
        if hasattr(block, "attn"):
            block.attn.record_attn = True
        
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{args.base_dir}/tokenizer.json",
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )
    
    # Load sequences for supergrammar
    general_sequences_file = os.path.join(args.base_dir, "test.jsonl")
    with open(general_sequences_file, "r") as f:
        general_sequences = [json.loads(line) for line in f if line.strip()][:TESTSET_SIZE]
    general_sequences = [seq["sequence"] for seq in general_sequences]
    
    # generate sequences for subgrammar by extracting the start symbol
    subgram = GRAMMARS[args.subgrammar]
    start_symbol = list(subgram)[0]
    subgrammar_sequences = sample_many(args.subgrammar, start_symbol, TESTSET_SIZE, MAX_LEN)
    subgrammar_sequences = [seq for seq, _ in subgrammar_sequences]

    # Run the analysis
    attention_ratio = analyze_attention_for_subgrammar(
        model, tokenizer, general_sequences, subgrammar_sequences, 
        "../results/attention_heads", args.subgrammar, device
    )
    
    # Print most responsive heads sorted to this subgrammar
    print(f"Top heads for {args.subgrammar}:")
    all_ratios = []
    for layer, heads in attention_ratio.items():
        for head, ratio in heads.items():
            all_ratios.append((layer, head, ratio))
    
    sorted_heads = sorted(all_ratios, key=lambda x: x[2], reverse=True)
    for layer, head, ratio in sorted_heads:
        print(f"  Layer {layer}, Head {head}: {ratio:.2f}x more attention")

if __name__ == "__main__":
    main()