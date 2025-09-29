from  analysis_hierarchy import epoch_step_num

def build_fixed_tokenizer(grammar, tok_path, special_tokens=None):
    """
    Create a WordLevel tokenizer whose vocab = exactly the terminals
    of the grammar, plus any provided special_tokens.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    
    if special_tokens is None:
        special_tokens = ["<|bos|>", "<|eos|>"]
    
    # Extract terminals from the grammar
    terminals = set()
    print(grammar.items())
    for lhs, productions in grammar.items():
            for option, _ in productions:
                for sym in option:
                    if sym not in grammar:
                        terminals.add(sym)
    
    # Build vocabulary
    vocab = {}
    idx = 0
    for tok in special_tokens:
        vocab[tok] = idx
        idx += 1
    for term in sorted(terminals):
        vocab[term] = idx
        idx += 1
    
    # Create tokenizer
    wordlevel = models.WordLevel(vocab=vocab)
    tokenizer = Tokenizer(wordlevel)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair=None,
        special_tokens=[
            ("<|bos|>", vocab["<|bos|>"]),
            ("<|eos|>", vocab["<|eos|>"])
        ],
    )
    
    # Save tokenizer
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    tokenizer.save(tok_path)
    
    # Create fast tokenizer
    tok_fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        bos_token="<|bos|>",
        eos_token="<|eos|>"
    )
    
    return tok_fast

import os
import json
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerFast
from model import GPT
from train import map_model_name
from generate_pcfg import split_and_tokenize
from eval import compare_model_vs_real_probs

def create_grammar(p_value):
    return {
        "A": [
            (["a", "A"], 1.0 - p_value),
            (["a"], p_value)
        ]
    }
def create_grammar2(p_value):
    return {
        "A": [
            (["x"],            p_value),  
            (["A", "and", "A"], 1.0 - p_value)        
        ]
    }

def sample_direct(grammar, start_symbol, max_len=100):
    """Sample a single sequence plus its log-prob from the grammar."""
    seq, log_prob, stack = [], 0.0, [start_symbol]
    while stack and len(seq) < max_len:
        sym = stack.pop()
        if sym in grammar:
            prods, probs = zip(*grammar[sym])
            idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
            chosen_prod, chosen_prob = prods[idx], probs[idx]
            log_prob += np.log(chosen_prob)
            # push in reverse so that leftmost expands first
            for s in reversed(chosen_prod):
                stack.append(s)
        else:
            seq.append(sym)
    if stack:
        # didn't terminate—retry
        return sample_direct(grammar, start_symbol, max_len)
    return " ".join(seq), log_prob

def sample_many_direct(grammar, start_symbol, n, max_len=100):
    """Sample n sequences directly from the grammar."""
    return [sample_direct(grammar, start_symbol, max_len) for _ in range(n)]

def generate_dataset(func, p_value, dataset_size, out_dir="../data/kl_analysis"):
    grammar = func(p_value)
    grammar_name = f"A_p{p_value:.2f}"
    dataset_dir = os.path.join(out_dir, f"{grammar_name}_{dataset_size}")
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Generating sequences for p={p_value}")
    train_seqs = sample_many_direct(grammar, "A", dataset_size)
    test_seqs  = sample_many_direct(grammar, "A", 500)

    # save test set
    with open(os.path.join(dataset_dir, "test.jsonl"), "w") as f:
        for seq, logp in test_seqs:
            f.write(json.dumps({"sequence": seq, "real_log_prob": logp}) + "\n")

    # build tokenizer
    tok_path = os.path.join(dataset_dir, "tokenizer.json")
    print(grammar_name)
    tokenizer = build_fixed_tokenizer(grammar, tok_path)

    # prepare train/val splits and tokenize
    train_strs = [s for s, _ in train_seqs]
    split_and_tokenize(train_strs, tokenizer, dataset_dir)

    return dataset_dir, grammar_name, tokenizer, test_seqs

def train_model(dataset_dir, grammar_name, model_config, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(model_config).to(device)

    print(f"Training model on {grammar_name}")
    model.train_model(
        data_dir=os.path.dirname(dataset_dir),
        dataset=os.path.basename(dataset_dir),
        num_epochs=num_epochs,
        batch_size=8,
        learning_rate=6e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
        checkpoint_every=50,
        config=model_config.name,
        device=device,
        seed=42
    )
    return 

def evaluate_model_once(model, tokenizer, test_sequences, device, p_value):
    model.eval()
    kl_div = compare_model_vs_real_probs(model, tokenizer, test_sequences, device) # TODO: check if we can rewrite this
    return sum(a["abs_logprob_diff"] for a in kl_div) / len(test_sequences)

def evaluate_over_epochs(checkpoint_dir, tokenizer, test_sequences, device, p_value, model_config):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    print(checkpoint_dir, ckpts)
    
    # Extract epoch and step from checkpoint filenames
    datapoints = []
    for ckpt in ckpts:
        epoch, step = epoch_step_num(ckpt)
        datapoints.append((ckpt, epoch, step))
    
    # Sort by epoch, then by step
    datapoints.sort(key=lambda x: (x[1], x[2]))
    datapoints = datapoints[1:]  # Take every second checkpoint to reduce computation
    kl_by_epoch = []
    for ckpt, epoch, step in datapoints:
        path = os.path.join(checkpoint_dir, ckpt)
        model = GPT(model_config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        kl = evaluate_model_once(model, tokenizer, test_sequences, device, p_value)
        kl_by_epoch.append((epoch, step, kl))
    
    return kl_by_epoch

def run_experiment(p_values, dataset_size, model_type, num_epochs, func):
    out_dir = "../data/kl_analysis"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = map_model_name(model_type)

    results_by_p = {}
    for p in p_values:
        ds_dir, grammar_name, tokenizer, test_seqs = generate_dataset(func, p, dataset_size, out_dir)
        train_model(ds_dir, grammar_name, model_config, num_epochs)
        ckpt_dir = os.path.join(ds_dir, model_config.name, 'new/seed_42')
        kl_curve = evaluate_over_epochs(ckpt_dir, tokenizer, test_seqs, device, p, model_config)
        results_by_p[p] = kl_curve

    return results_by_p

def plot_kl_over_epochs(results_by_p, model_type):
    plt.figure(figsize=(4.8, 3))
    print(results_by_p)
    
    # Constant for normalizing steps
    DIVIDER = 200  # Adjust based on your steps per epoch
    
    for p, curve in results_by_p.items():
        # Extract epochs and kl values
        if p == 0.6:
            x_values = [epoch*200 + step for epoch, step, _ in curve]
        else:
            x_values = [epoch*150 + step for epoch, step, _ in curve]
        y_values = [kl for _, _, kl in curve]

        x_values, y_values = zip(
            *[(x, y) for x, y in zip(x_values, y_values) if x <= 150]
        )
        
        plt.plot(x_values, y_values, label=f"p={p:.2f}")
    
    plt.xlabel("Mini-batch Iterations")
    plt.ylabel("KL Divergence")
    #plt.title(f"KL Divergence over Epochs", fontsize=16)
    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/kl_over_epochs_{model_type}.png", dpi=300)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot KL divergence over epochs for varying p values"
    )
    parser.add_argument("--p_values", nargs="+", type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of probability values to test"
    )
    parser.add_argument(
        "--dataset_size", type=int, required=True,
        help="Number of sequences per dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Which GPT configuration to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs"
    )
    parser.add_argument("--grammar", type=str, required=True, help="Type of PCFG to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    if args.grammar == "tail_recursive":
        func = create_grammar
    elif args.grammar == "two_sided_recursion":
        func = create_grammar2
    else:
        raise ValueError(f"Unknown grammar type: {args.grammar}")

    results = run_experiment(
        args.p_values,
        args.dataset_size,
        args.model,
        args.epochs,
        func
    )
    plot_kl_over_epochs(results, args.model)
