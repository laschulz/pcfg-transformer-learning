import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
from eval import evaluate_generated_sequences
from model import TwoLayer, FourLayer, SixLayer, GPT

from train import map_model_name
import argparse

def plot_results(results_log, model_name):

    for pcfg_name, ckpt_dict in results_log.items():
        # collect (epoch, accuracy) pairs
        epoch_acc = []
        for ckpt_file, data in ckpt_dict.items():
            try:
                # assumes your checkpoints are named like "checkpoint_epoch_10.pt"
                epoch = int(os.path.splitext(ckpt_file)[0].split("_")[-1])
            except ValueError:
                continue
            epoch_acc.append((epoch, data["accuracy"]))

        epoch_acc.sort(key=lambda x: x[0])
        epochs, accuracies = zip(*epoch_acc)

        # plot this grammar’s curve
        plt.plot(epochs, accuracies, marker='o', label=pcfg_name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Epochs for {model_name}")
    plt.grid(True)
    plt.legend(title="Grammar")
    plt.tight_layout()

    plt.savefig(f"../results/accuracy_plot_{model_name}.png")
    plt.show()
    plt.close()

def plot_avg_length(results_log, model_name):
    for pcfg_name, ckpt_dict in results_log.items():
        epoch_len = []
        for ckpt_file, data in ckpt_dict.items():
            # parse epoch from filename, e.g. "checkpoint_10.pt" → 10
            try:
                epoch = int(os.path.splitext(ckpt_file)[0].split("_")[-1])
            except ValueError:
                continue

            seqs = data.get("generated_sequences", [])

            # assume each seq is a whitespace‐joined string of tokens
            lengths = [
                len(s.split()) if isinstance(s, str) and s.strip() != "" else 0
                for s in seqs
            ]
            avg_len = float(np.mean(lengths))
            epoch_len.append((epoch, avg_len))

        # sort by epoch
        epoch_len.sort(key=lambda x: x[0])
        epochs, avg_lengths = zip(*epoch_len)

        # plot
        plt.plot(epochs, avg_lengths, marker='o', label=pcfg_name)

    plt.xlabel("Epoch")
    plt.ylabel("Average Generated Sequence Length")
    plt.title(f"Average Generated Sequence Length over Epochs ({model_name})")
    plt.grid(True)
    plt.legend(title="Grammar")
    plt.tight_layout()

    plt.savefig(f"../results/avg_length_plot_{model_name}.png")
    plt.show()
    plt.close()

def plot_kl_divergence(results_log, model_name):
    for pcfg_name, ckpt_dict in results_log.items():
        epoch_kl = []
        for ckpt_file, data in ckpt_dict.items():
            # parse epoch from filename, e.g. "checkpoint_10.pt" → 10
            try:
                epoch = int(os.path.splitext(ckpt_file)[0].split("_")[-1])
            except ValueError:
                continue

            kl_divergence = data.get("kl_divergence", 0.0)
            epoch_kl.append((epoch, kl_divergence))

        # sort by epoch
        epoch_kl.sort(key=lambda x: x[0])
        epochs, kl_values = zip(*epoch_kl)

        # plot
        plt.plot(epochs, kl_values, marker='o', label=pcfg_name)

    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title(f"KL Divergence over Epochs ({model_name})")
    plt.grid(True)
    plt.legend(title="Grammar")
    plt.tight_layout()

    plt.savefig(f"../results/kl_divergence_plot_{model_name}.png")
    plt.show()
    plt.close()

def analyze(pcfgs, dataset_size, model_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_log = {}
    model = GPT(model_config).to(device)

    seed = 42
    torch.manual_seed(seed)

    #Load existing results if file exists
    results_path = os.path.join("../results", "results_log.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results_log = json.load(f)

    for pcfg in pcfgs:
        main_path = os.path.join('../data', pcfg, f"{pcfg}_{dataset_size}")
        with open(f"{main_path}/train.jsonl", 'r') as f:
            training_sequences = [json.loads(line)["sequence"] for line in f]
        with open(f"{main_path}/test.jsonl", 'r') as f:
            test_sequences = [(json.loads(line)["sequence"], json.loads(line)["real_log_prob"]) for line in f]

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{main_path}/tokenizer.json",
            bos_token="<|bos|>",
            eos_token="<|eos|>"
        )

        checkpoints_dir = os.path.join(main_path, model_config.name)
        if pcfg not in results_log:
            results_log[pcfg] = {}

        for ckpt in sorted(os.listdir(checkpoints_dir)):
            if not ckpt.endswith(".pt"): #or ckpt in results_log[pcfg]:
                continue  # Skip already analyzed checkpoints

            print(f"[INFO] Analyzing {ckpt} for {pcfg}...")

            model.load_state_dict(
                torch.load(os.path.join(checkpoints_dir, ckpt), map_location=device)
            )

            generated_sequences, accuracy, train_overlap, res = evaluate_generated_sequences(
                model,
                tokenizer,
                training_sequences,
                pcfg,
                test_sequences,
                device,
                num_samples=50,
                max_length=100
            )

            kl_divergence = np.sum([r['abs_logprob_diff'] for r in res])

            results_log[pcfg][ckpt] = {
                "generated_sequences": generated_sequences,
                "accuracy": accuracy,
                "train_overlap": train_overlap,
                "kl_divergence": kl_divergence,
                "res": res
            }
    
    results_path = os.path.join("../results", "results_log.json")

    # Load existing results if file exists
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # Append new results
    existing_results.update(results_log)

    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(existing_results, f)

    plot_results(results_log, model_config.name)
    plot_avg_length(results_log, model_config.name)
    plot_kl_divergence(results_log, model_config.name)


def argument_parser():
    parser = argparse.ArgumentParser(description="Analyze PCFG Transformer models.")
    parser.add_argument("--pcfgs", nargs='+', help="List of PCFG names to analyze.")
    parser.add_argument("--dataset_size", type=str, default='1000', help="Size of the dataset to analyze.")
    parser.add_argument("--model", type=str, choices=["TwoLayer", "FourLayer", "SixLayer", "GPT"], default="SixLayer", help="Model configuration to use.")
    parser.add_argument('--load_data', action='store_true')  # default is False
    return parser.parse_args()

def main():
    args = argument_parser()
    model_config = map_model_name(args.model)
    if args.load_data:
        with open("../results/results_log.json", "r") as f:
            results_log = json.load(f)
        plot_results(results_log, model_config.name)
    else:
        analyze(args.pcfgs, args.dataset_size, model_config)

if __name__ == "__main__":
    main()