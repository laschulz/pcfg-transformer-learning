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
    parser.add_argument("--continue_from", type=int, default=0, help="Epoch to continue training from if continuing")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train the model")
    return parser.parse_args()

def main():
    args = parse_args()
    pcfg = args.pcfg
    dataset = args.dataset
    
    main_path = f'../data/{pcfg}/{dataset}'
    checkpoint_path = f'../data/{args.checkpoint_path}' if args.checkpoint_path else None

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
            num_epochs=args.num_epochs,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
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
            num_epochs=args.num_epochs,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            checkpoint_every=5,
            config=config.name,
            device=device,
            continue_from=args.continue_from
        )

if __name__ == "__main__":
    main()

