import torch
import argparse
from model import GPT, TwoLayer, FourLayer, SixLayer, OneLayer
import os
import shutil

def map_model_name(model_name):
    if model_name == "TwoLayer":
        return TwoLayer()
    elif model_name == "FourLayer":
        return FourLayer()
    elif model_name == "SixLayer":
        return SixLayer()
    elif model_name == "OneLayer":
        return OneLayer()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train or load a GPT model on a PCFG dataset")
    parser.add_argument("--grammar", type=str, required=True, help="Type of PCFG to use")
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--model", type=str, choices=["TwoLayer", "FourLayer", "SixLayer", "OneLayer"], default="FourLayer", help="Type of GPT model to use")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional path to checkpoint to load")
    parser.add_argument("--continue_training", action='store_true', help="Continue training from the checkpoint if provided")
    parser.add_argument("--continue_from", type=int, default=0, help="Epoch to continue training from if continuing")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train the model")
    return parser.parse_args()

def trainer(model, grammar, config, dataset_size, checkpoint_path, checkpoint_every, num_epochs, save_first_x_epochs, continue_from, continue_training, device, seed):
    dataset = f"{grammar}_{dataset_size}"
    if checkpoint_path and os.path.exists(checkpoint_path) and not continue_training:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif checkpoint_path and continue_training:
        print(f"Continuing training from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        epoch = int(checkpoint_path.split('_')[-1].split('.')[0])  # Extract epoch number from filename
        model.train_model(
            data_dir=f'../data/{grammar}',
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            save_first_x_epochs=save_first_x_epochs,
            checkpoint_every=checkpoint_every,
            config=config.name,
            device=device, 
            continue_from=epoch,
            train_type="continued", 
            seed=seed
        )
        # Copy checkpoint to the standard checkpoints directory
        copy_to_dir = f'../data/{grammar}/{dataset}/{config.name}/continued/seed_{seed}'
        checkpoint_path = checkpoint_path.rsplit('/', 1)[0]

        for file in os.listdir(checkpoint_path):
            if file.startswith('epoch_') and file.endswith('.pt'):
                file_epoch = int(file.split('_')[1].split('.')[0])
                if file_epoch < epoch:
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
            save_first_x_epochs=save_first_x_epochs,
            checkpoint_every=checkpoint_every,
            config=config.name,
            device=device,
            continue_from=continue_from,
            train_type="new",
            seed=seed
        )

def main():
    args = parse_args()
    checkpoint_path = f'../data/{args.checkpoint_path}' if args.checkpoint_path else None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = map_model_name(args.model)
    model = GPT(config).to(device)

    trainer(model, args.grammar, config, args.dataset_size, checkpoint_path=checkpoint_path, checkpoint_every=5,
            num_epochs=args.num_epochs, save_first_x_epochs=10, continue_from=args.continue_from, 
            continue_training=args.continue_training, device=device, seed=42)

if __name__ == "__main__":
    main()

