import torch
import argparse
from transformers import PreTrainedTokenizerFast
from model import GPT, TwoLayer, FourLayer, SixLayer
from eval import evaluate_generated_sequences
import json
import os
import numpy as np

def map_model_name(model_name):
    if model_name == "TwoLayer":
        return TwoLayer()
    elif model_name == "FourLayer":
        return FourLayer()
    elif model_name == "SixLayer":
        return SixLayer()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train or load a GPT model on a PCFG dataset")
    parser.add_argument("--pcfg", type=str, required=True, help="Type of PCFG to use")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., LinearRecursion_1000)")
    parser.add_argument("--model", type=str, choices=["TwoLayer", "FourLayer", "SixLayer"], default="FourLayer", help="Type of GPT model to use")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional path to checkpoint to load")
    parser.add_argument("--continue_training", action='store_true', help="Continue training from the checkpoint if provided")
    return parser.parse_args()

def main():
    args = parse_args()
    pcfg = args.pcfg
    dataset = args.dataset
    
    main_path = f'../data/{pcfg}/{dataset}'
    checkpoint_path = f'{main_path}/{args.checkpoint_path}' if args.checkpoint_path else None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    config = map_model_name(args.model)
    model = GPT(config).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path) and not args.continue_training:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif checkpoint_path and args.continue_training:
        print(f"Continuing training from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        epoch = int(checkpoint_path.split('_')[-1].split('.')[0])  # Extract epoch number from filename
        model.train_model(
            data_dir=f'../data/{pcfg}',
            dataset=dataset,
            num_epochs=50,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            early_stopping=15,
            checkpoint_every=5,
            config=config.name,
            device=device, 
            continue_from=epoch
        )
    else:
        print(f"Training new model on dataset {dataset}")
        model.train_model(
            data_dir=f'../data/{pcfg}',
            dataset=dataset,
            num_epochs=200,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            early_stopping=15,
            checkpoint_every=5,
            config=config.name,
            device=device
        )

    jsonl_path = f'{main_path}/train.jsonl'
    with open(jsonl_path, 'r') as f:
        training_sequences = [json.loads(line)["sequence"] for line in f]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{main_path}/tokenizer.json", bos_token="<|bos|>", eos_token="<|eos|>")   
    with open(f"{main_path}/test.jsonl", 'r') as f:
        test_sequences = [(json.loads(line)["sequence"], json.loads(line)["real_log_prob"]) for line in f]
    generated_sequences, accuracy, train_overlap, res = evaluate_generated_sequences(model, tokenizer, training_sequences, pcfg, test_sequences, device, num_samples=50, max_length=100)

    kl_divergence = np.sum([r['abs_logprob_diff'] for r in res])

    file_name = f"{os.path.basename(checkpoint_path).removesuffix('.pt')}.jsonl" if checkpoint_path else "best.jsonl"
    path_dir = f'{main_path}/{config.name}'
    os.makedirs(path_dir, exist_ok=True)
    with open(f"{path_dir}/{file_name}", 'w') as f:
        for seq in generated_sequences:
            f.write(f"{seq}\n")
        f.write(f"Accuracy: {accuracy:.4f}, Train Overlap: {train_overlap:.4f}, KL Divergence: {kl_divergence:.4f}\n\n\n")
        f.write("Known sequences log probabilities:\n")
        for r in res:
            f.write(f"{r['text']}: ")
            f.write(f"Model Log Prob: {np.exp(r['log_prob_model'])} ")
            f.write(f"Real Log Prob: {np.exp(r['log_prob_real'])} ")
            f.write(f"Absolute Log Prob Difference: {r['abs_logprob_diff']}\n ")

if __name__ == "__main__":
    main()

