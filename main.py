import torch
import argparse
from transformers import PreTrainedTokenizerFast
from model import GPT, TwoLayerTransformerConfig
from eval import evaluate_generated_sequences
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train or load a GPT model on a PCFG dataset")
    parser.add_argument("--pcfg", type=str, required=True, help="Type of PCFG to use")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., LinearRecursion_1000)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional path to checkpoint to load")
    return parser.parse_args()

def main():
    args = parse_args()
    pcfg = args.pcfg
    dataset = args.dataset
    
    main_path = f'data/{pcfg}/{dataset}'
    checkpoint_path = f'{main_path}/{args.checkpoint_path}' if args.checkpoint_path else None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = TwoLayerTransformerConfig()
    model = GPT(config).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Training new model on dataset {dataset}")
        model.train_model(
            data_dir=f'data/{pcfg}',
            dataset=dataset,
            num_epochs=10,
            batch_size=8,
            learning_rate=6e-4,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            early_stopping=3,
            checkpoint_every=1,
            device=device
        )

    jsonl_path = f'{main_path}/train.jsonl'
    with open(jsonl_path, 'r') as f:
        training_sequences = [json.loads(line)["sequence"] for line in f]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{main_path}/tokenizer.json", bos_token="<|bos|>")   
    evaluate_generated_sequences(model, tokenizer, training_sequences, pcfg)

if __name__ == "__main__":
    main()

