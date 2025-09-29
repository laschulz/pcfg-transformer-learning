import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import GPT
from train import map_model_name
from transformers import PreTrainedTokenizerFast

def calculate_average_similarity(similarity_matrices):
    """Calculate the average similarity for each layer across all sequence pairs"""
    avg_similarities = {}
    
    for layer_key, matrix in similarity_matrices.items():
        # For each layer, compute average similarity excluding the diagonal (self-similarity)
        n = matrix.shape[0]
        # Create a mask to exclude the diagonal
        mask = ~np.eye(n, dtype=bool)
        # Calculate average of off-diagonal elements
        avg_similarities[layer_key] = np.mean(matrix[mask])
    
    return avg_similarities

def compute_cos_sim(a, b):
    """Compute cosine similarity between two vectors"""
    a = a.flatten()
    b = b.flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())

def collect_activations_per_input(model, tokenizer, inputs, agg):
    """
    Capture layer-wise activations for ONE model over a list of input strings.
    Returns: dict[str -> dict[layer_key -> tensor]]
      - if agg == "tokens": tensor is [T_i, D] for each input i (no padding)
      - if agg == "last":   tensor is [D]     (last non-pad token per input)
      - if agg == "mean":   tensor is [D]     (mean over tokens)
      - if agg == "mean_unit": [D]            (mean of unit-normalized tokens)
    """
    model.eval()

    n_layers = len(model.transformer.h)
    per_input = {}

    # Define hook storage
    raw = {f"attn_{i}": None for i in range(n_layers)}
    raw.update({f"mlp_{i}": None for i in range(n_layers)})

    def get_hook(key):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if x.ndim == 2:  # [T,D] -> [1,T,D]
                x = x.unsqueeze(0)
            raw[key] = x.detach().cpu()  # [1, T, D]
        return hook

    # Register hooks once
    hooks = []
    for i, block in enumerate(model.transformer.h):
        attn_mod = getattr(block, "attn", None) or getattr(block, "mha", None) or getattr(block, "self_attn", None)
        mlp_mod  = getattr(block, "mlp", None) or getattr(block, "ffn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(get_hook(f"attn_{i}")))
        if mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(get_hook(f"mlp_{i}")))

    # Process each sequence separately
    with torch.no_grad():
        for s in inputs:
            enc = tokenizer(s, return_tensors="pt")
            input_ids = enc["input_ids"]
            _ = model(input_ids)

            # Aggregate per layer
            per_input[s] = {}
            for key, X in raw.items():
                if X is None:
                    continue
                Xi = X[0]
                if agg == "tokens": # this produces a matrix, rest is a vector
                    rep = Xi
                elif agg == "last":
                    rep = Xi[-1]
                elif agg == "mean":
                    rep = Xi.mean(dim=0)
                elif agg == "mean_unit":
                    Zi = Xi / (Xi.norm(dim=-1, keepdim=True) + 1e-8)
                    rep = Zi.mean(dim=0)
                else:
                    raise ValueError(f"Unknown agg={agg}")
                per_input[s][key] = rep

    # Remove hooks
    for h in hooks:
        h.remove()

    return per_input

def compute_sequence_similarity_matrices(model, tokenizer, sequences):
    """Compute CKA similarity between all pairs of sequences for all layers"""
    # Get activations for each sequence - use tokens to get full representations
    activations = collect_activations_per_input(model, tokenizer, sequences, agg="mean_unit")
    
    # Get all available layer keys from the activations
    # Just take the keys from the first sequence
    first_seq = sequences[0]
    available_layers = list(activations[first_seq].keys())
    
    # Initialize dictionary to store similarity matrices for each layer
    similarity_matrices = {layer_key: np.zeros((len(sequences), len(sequences))) 
                          for layer_key in available_layers}
    
    # Compute CKA for each pair of sequences for each layer
    for layer_key in available_layers:
        for i in range(len(sequences)):
            for j in range(len(sequences)):
                # Get activations for sequence i and j at this layer
                act_i = activations[sequences[i]][layer_key]
                act_j = activations[sequences[j]][layer_key]
                    
                similarity_matrices[layer_key][i, j] = compute_cos_sim(act_i, act_j)
    
    return similarity_matrices

def compute_cross_sequence_similarity_matrices(model, tokenizer, seqs_A, seqs_B):
    """
    Compute cosine similarity between every sequence in seqs_A and every sequence in seqs_B
    for all layers.
    
    Returns:
        dict[layer_key -> np.ndarray of shape (len(seqs_A), len(seqs_B))]
    """
    # Collect activations for both sets
    acts_A = collect_activations_per_input(model, tokenizer, seqs_A, agg="mean_unit")
    acts_B = collect_activations_per_input(model, tokenizer, seqs_B, agg="mean_unit")
    
    available_layers = list(acts_A[seqs_A[0]].keys())
    matrices = {layer_key: np.zeros((len(seqs_A), len(seqs_B))) 
                for layer_key in available_layers}
    
    for layer_key in available_layers:
        for i, sa in enumerate(seqs_A):
            for j, sb in enumerate(seqs_B):
                ai = acts_A[sa][layer_key]
                bj = acts_B[sb][layer_key]
                matrices[layer_key][i, j] = compute_cos_sim(ai, bj)
    
    return matrices


def plot_all_similarity_heatmaps(similarity_matrices, sequences, output_base_path, type, seed=None):
    """Create heatmap visualizations for all layers"""
    # Create shortened labels for readability
    labels = []
    for i, seq in enumerate(sequences):
        if len(seq) > 20:
            labels.append(f"{i}: {seq[:20]}...")
        else:

            labels.append(f"{i}: {seq}")
    
    # Create a plot for each layer
    for layer_key, matrix in similarity_matrices.items():
        # Generate output path
        if seed is not None:
            output_path = f"{output_base_path}_{layer_key}_seed{seed}_{type}.png"
        else:
            output_path = f"{output_base_path}_{layer_key}_{type}.png"
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(f"CKA Activation Similarity Between Sequences (Layer: {layer_key})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved similarity heatmap for {layer_key} to {output_path}")
        plt.close()

def load_sequences_from_file(path):
    """Read sequences from a text file, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections by empty line(s)
    parts = [block.strip().splitlines() for block in content.strip().split("\n\n")]

    # Clean up whitespace
    parts = [[line.strip() for line in block if line.strip()] for block in parts]

    if len(parts) == 1:
        return parts[0], None
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(f"Expected 1 or 2 blocks of sequences, got {len(parts)}")

def main():
    parser = argparse.ArgumentParser(description="Analyze sequence activation similarity using CKA")
    parser.add_argument("--model", type=str, required=True, help="Model architecture name")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seeds to analyze (for multiple models)")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load")
    parser.add_argument("--train_type", type=str, required=True, help="Type of training (e.g., new, continued)")
    parser.add_argument("--output", type=str, default="../results/seq_similarity", help="Base output path")
    parser.add_argument("--sequence_file", type=str, default="sequences.txt", help="File containing sequences to analyze")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the heatmaps")
    args = parser.parse_args()
    
    # Load sequences
    seq_A, seq_B = load_sequences_from_file(args.sequence_file)

    # Dictionary to store matrices for each seed and layer
    all_matrices = {}
    all_avg_similarities = {}
    
    for seed in args.seeds:
        # Load model for this seed
        seed_path = os.path.join(args.base_dir, f"{args.model}/{args.train_type}/seed_{seed}/epoch_{args.epoch}_0.pt") 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT(map_model_name(args.model)).to(device)
        model.load_state_dict(torch.load(seed_path, map_location=device))
        
        tokenizer_path = os.path.join(args.base_dir, "tokenizer.json")
        # Load tokenizer
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<|bos|>",
            eos_token="<|eos|>"
        )
        
        # Compute similarity matrices for all layers
        if seq_B is None:
            similarity_matrices = compute_sequence_similarity_matrices(
                model, tokenizer, seq_A
            )
            avg_similarities = calculate_average_similarity(similarity_matrices)
        else:
            similarity_matrices = compute_cross_sequence_similarity_matrices(
                model, tokenizer, seq_A, seq_B
            )
            avg_similarities = {
                layer: matrix.mean() for layer, matrix in similarity_matrices.items()
            }
        
        # Store matrices for this seed
        all_matrices[seed] = similarity_matrices
        all_avg_similarities[seed] = avg_similarities
        
        # Plot results for this seed
        if args.plot:
            plot_all_similarity_heatmaps(similarity_matrices, seq_A, args.output, args.train_type, seed)

    print("\n==== AVERAGE ACROSS ALL SEEDS ====")
    # Get all layer keys
    layer_keys = set()
    for seed_data in all_avg_similarities.values():
        layer_keys.update(seed_data.keys())
        
    # For each layer, compute average across all seeds
    print(f"for {args.train_type}:")
    for layer in sorted(layer_keys):
        values = []
        for seed in args.seeds:
            if layer in all_avg_similarities[seed]:
                values.append(all_avg_similarities[seed][layer])
        if values:
            avg_across_seeds = sum(values) / len(values)
            print(f"  {layer}: {avg_across_seeds:.4f}")
if __name__ == "__main__":
    main()