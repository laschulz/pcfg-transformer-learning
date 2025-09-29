import torch
import argparse
from model import *
import os
import shutil

def map_model_name(model_name):
    if model_name == "FourLayer":
        return FourLayer()
    elif model_name == "TwoLayer":
        return TwoLayer()
    elif model_name == "TwoLayer_SMALL":
        return TwoLayer_SMALL()
    elif model_name == "TwoLayer_31":
        return TwoLayer_31()
    elif model_name == "OneLayer":
        return OneLayer()
    elif model_name == "OneLayer_BIG":
        return OneLayer_BIG()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train or load a GPT model on a PCFG dataset")
    parser.add_argument("--grammar", type=str, required=True, help="Type of PCFG to use")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model", type=str, default="FourLayer", help="Type of GPT model to use")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional path to checkpoint to load")
    parser.add_argument("--continue_from", type=int, default=0, help="Epoch to continue training from if continuing")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def trainer(model, grammar, config, dataset_name, checkpoint_path, checkpoint_every, 
            num_epochs, continue_from, continue_training, device, seed, safe_only_last=False):
    dataset = f"{grammar}_{dataset_name}"
    dir = f'../data/{grammar}/{dataset}'

    if checkpoint_path and continue_training:
        print(f"Continuing training from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.train_model(
            data_dir=f'../data/{grammar}',
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            checkpoint_every=checkpoint_every,
            config=config.name,
            device=device, 
            continue_from=continue_from,
            train_type="continued", 
            seed=seed,
            safe_only_last=safe_only_last
        )
        # Copy checkpoint to the standard checkpoints directory
        copy_to_dir = f'{dir}/{config.name}/continued/seed_{seed}'
        checkpoint_path = checkpoint_path.rsplit('/', 1)[0]

        for file in os.listdir(checkpoint_path):
            if file.startswith('epoch_') and file.endswith('.pt'):
                file_epoch = int(file.split('_')[1].split('.')[0])
                if file_epoch < continue_from:
                    shutil.copy(os.path.join(checkpoint_path, file), os.path.join(copy_to_dir, file))
    else:
        print(f"Training new model on dataset {dataset}")
        model.train_model(
            data_dir=f'../data/{grammar}',
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            checkpoint_every=checkpoint_every,
            config=config.name,
            device=device,
            continue_from=continue_from,
            train_type="new",
            seed=seed,
            safe_only_last=safe_only_last
        )

def main():
    args = parse_args()
    checkpoint_path = f'../data/{args.checkpoint_path}' if args.checkpoint_path else None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = map_model_name(args.model)
    model = GPT(config).to(device)

    trainer(model, args.grammar, config, args.dataset_name, checkpoint_path=checkpoint_path, checkpoint_every=50,
            num_epochs=args.num_epochs, continue_from=args.continue_from, 
            continue_training=args.continue_training, device=device, seed=args.seed)

if __name__ == "__main__":
    main()

