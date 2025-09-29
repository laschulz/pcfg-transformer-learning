# 1. train model
# 2. load model
# 3. run 
from model import GPT
import torch
import torch.nn.functional as F
from train import map_model_name
import argparse
from transformers import PreTrainedTokenizerFast
import matplotlib.pyplot as plt
import numpy as np


# SEQUENCES=["sL1 sL2 cond eL2", #same
#            "sL1 sL2 not cond eL2", #same
#            "sL1 sL2 cond eL2 sL2 not cond eL2", #1 loop
#            "sL1 sL2 cond and cond eL2 sL2 cond eL2", # 1 loop
#            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2", # 2 loops
#            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2", # 2 loops
#            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2", # 3 loops
#            "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2", # 3 loops
#            "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2", # 4 loops
#             "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2", # 4 loops
#             "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond eL2", # 5 loops
#             "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2", # 5 loops
#             "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2", # 6 loops
#             "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2", # 6 loops
#             "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2 sL2 cond eL2", # 7 loops
#             "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2", # 7 loops
#             "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2 sL2 cond eL2 sL2 cond and cond eL2", # 8 loops
#             "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2", # 8 loops
#             "sL1 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 cond and not cond eL2 sL2 cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2", # 9 loops
#             "sL1 sL2 cond and cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 cond and cond and not cond eL2 sL2 not cond eL2 sL2 cond and cond eL2 sL2 not cond eL2 sL2 cond eL2 sL2 not cond eL2 sL2 not cond and cond and cond eL2" # 9 loops
# ]

# SEQUENCES=[
#     "sL4", # 0 loops
#     "sL4 ( a ) eL4 sL4", # 1 loop, 6 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4", # 2 loops, 11 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4", # 3 loops, 16 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4 ( d ) eL4 sL4", # 4 loops, 21 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4 ( d ) eL4 sL4 ( e ) eL4 sL4", # 5 loops, 26 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4 ( d ) eL4 sL4 ( e ) eL4 sL4 ( f ) eL4 sL4", # 6 loops, 31 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4 ( d ) eL4 sL4 ( e ) eL4 sL4 ( f ) eL4 sL4 ( g ) eL4 sL4", # 7 loops, 36 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4 ( d ) eL4 sL4 ( e ) eL4 sL4 ( f ) eL4 sL4 ( g ) eL4 sL4 ( h ) eL4 sL4", # 8 loops, 41 tokens
#     "sL4 ( a ) eL4 sL4 ( b ) eL4 sL4 ( c ) eL4 sL4 ( d ) eL4 sL4 ( e ) eL4 sL4 ( f ) eL4 sL4 ( g ) eL4 sL4 ( h ) eL4 sL4 ( i ) eL4 sL4", # 9 loops, 46 tokens
# ]


def generate_sequences(case, n, prefix=""):
    sequences = []
    
    if case == 1:
        # Pattern: "( a ) ( a ) ... ( a ) ("
        for i in range(0, n+1):
            sequences.append("( a ) " * i + "(")
            
    elif case == 2:
        # Pattern: "(, ((, ((( ..."
        for i in range(1, 230): #230
            sequences.append("( " * i)
    elif case == 3:
        for i in range(1, 230): #2*n+1
            sequences.append(prefix + " ( " * i)
    
    sequences = [seq.strip() for seq in sequences]
    sequences = [seq.replace("  ", " ") for seq in sequences]
    return sequences
    
def ground_truth_logit(tokenizer):
    """
    Creates a ground truth logit vector where "(" has probability 0.7 and
    letters a-y each have probability 0.3/25, formatted as log probabilities
    after softmax to match model output format.
    """
    # Get vocabulary size
    probs = torch.zeros(1, 5)  # Match dimensions with model output
    
    # Get token IDs from tokenizer
    vocab = tokenizer.get_vocab()
    
    # if case == 1:
    #     # Find letter tokens
    #     probs[0, vocab["a"]] = 1.0
    # elif case == 2:
    #     probs[0, vocab["("]] = 0.7
    #     probs[0, 1] = 0.3
    # elif case == 3:
    probs[0, vocab["("]] = 0.8
    probs[0, vocab["a"]] = 0.2
    # Find letter tokens
    # letter_ids = []
    # for letter in "abcdefghijklmnopqrstuvwxy":  # a through y
    #     if letter in vocab:
    #         letter_ids.append(vocab[letter])
    
    # if letter_ids:
    #     letter_prob = 1# 0.3 * 0.04
    #     for lid in letter_ids:
    #         probs[0, lid] = letter_prob

    return probs

def collect_activations(model, tokenizer, sequences, agg=None):
    # 1) Prepare empty buffers
    n_layers = len(model.transformer.h)
    acts = {f"attn_{i}": [] for i in range(n_layers)}
    acts.update({f"mlp_{i}": [] for i in range(n_layers)})

    # 2) Define hook factory
    def get_hook(key):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            B, L, D = out.shape
            if agg is None:
                rep = out.reshape(B*L, D)
            elif agg == "mean":
                rep = out.mean(dim=1)  # (B, D) -> (1, D)
            elif agg == "mean_unit":
                z = out / (out.norm(dim=-1, keepdim=True) + 1e-8)  # per-token unit
                rep = z.mean(dim=1)  
            acts[key].append(rep.detach().cpu())
        return hook

    # 3) Register hooks
    hooks = []
    for i, block in enumerate(model.transformer.h):
        # be robust to naming
        attn_mod = getattr(block, "attn", None) or getattr(block, "mha", None) or getattr(block, "self_attn", None)
        mlp_mod  = getattr(block, "mlp",  None) or getattr(block, "ffn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(get_hook(f"attn_{i}")))
        if mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(get_hook(f"mlp_{i}")))

    # 4) Run forward passes
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt")
            input_ids = enc["input_ids"]                    # shape (1, L)
            _ = model(input_ids)  

    # 5) Remove hooks
    for h in hooks:
        h.remove()

    # 6) Concatenate into single tensor per layer
    for k in list(acts.keys()):
        acts[k] = torch.cat(acts[k], dim=0)

    return acts

def get_logits(model, tokenizer, sequences, device):
    model.eval()
    results = []

    with torch.no_grad():
        for seq in sequences:
            encoded = tokenizer.encode(seq, return_tensors="pt").to(device)
            encoded = encoded[:,:-1]

            # Forward pass
            logits, _ = model(encoded)
            probs = F.softmax(logits.squeeze(1), dim=-1) #change log_softmax to softmax
            results.append(probs.detach().cpu())
    return results


def plot_logit_differences(sequences, diffs, case):
    """
    Plot the differences between model logits and ground truth logits
    with increasing sequence depth for multiple seeds.
    
    Draw individual seeds in gray and average across seeds as main line.
    Seeds are labeled at the right side of each seed line.
    
    Args:
        sequences: List of input sequences
        diffs: Dictionary of seed -> list of differences
        case: Case number for the filename
        plateau_start: Depth where plateau begins (default=30)
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
            plateau_start = 5
        if case == 2:
            plateau_start = 80
        elif case == 3:
            plateau_start = 190
        #plt.axvspan(plateau_start, len(x)-1, color="lightgray", alpha=0.2, label="Plateau region")
        
        # Add plateau mean line for average
        if plateau_start < len(avg_values):
            plateau_mean = np.mean(avg_values[plateau_start:])
            plt.hlines(plateau_mean, xmin=plateau_start, xmax=len(x)-1,
                      colors='blue', linestyles="dashed")
            plt.text(len(x), plateau_mean, f"{plateau_mean:.3f}",
                    color='blue', va="center", fontweight='bold')
    
    plt.xlabel('Sequence Depth')
    plt.xlim(-5, len(x)+30)
    plt.yticks([])
    #plt.ylabel('Prediction Error')
    plt.legend(loc='lower right')
    plt.xticks()
    #plt.yticks()
    
    # Add ticks only every 5 steps
    tick_positions = [i for i in x if i % 40 == 0]
    tick_labels = [f"{i}" for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    #plt.ylim(0, 0.25)
    
    plt.tight_layout()
    save_path = f'../results/logit_differences_case_{case}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--generate', action='store_true')

    return parser.parse_args()

def main():
    args = parse_args()
    config = map_model_name(args.model)
    model = GPT(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = f'{args.path}/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token="<|bos|>", eos_token="<|eos|>")

    gt_logit = ground_truth_logit(tokenizer).to(device)
    print(gt_logit)

    diffs = {}
    for s in args.seeds:

        checkpoint_path = f'{args.path}/{args.model}/new/seed_{s}/epoch_5_0.pt'
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        model.to(device)

        #sequences = generate_sequences(args.case, 80, "( ( ( ( ( ( ( ( ( a ) ) ) ) ) ) ) ) )")
        sequences = generate_sequences(args.case, 80, "( a ) ( a ) ( a ) ( a ) ( a ) ( a ) ")
        #sequences = generate_sequences(args.case, 80, "( a ) ( a ) ( a ) ) ( a a ) ( a ) ( a ) ")

        #sequences = "("
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

        logits = get_logits(model, tokenizer, sequences, device)
        print(logits[0])
        print(logits[-1])


        diffs[s] = []
        for l in logits:
            # loss = F.kl_div(l, gt_logit, reduction="batchmean", log_target=True)
            diff_vec = torch.abs(l - gt_logit)
            diff = diff_vec.sum().item()
            diffs[s].append(diff)

        # Call the plotting function with the calculated differences
    plot_logit_differences(sequences, diffs, args.case)


    # activations = collect_activations_per_input(model, tokenizer, SEQUENCES, agg='mean')
    # for layer, acts in activations.items():
    #     print(f"Layer: {layer}, Activations shape: {acts.shape}")

if __name__ == "__main__":
    main()