"This file computes the L2-Norm between two models' parameters."
"We assume that the training data already exists (and the training data is fixed)"
import torch
import torch.nn.functional as F
from model import GPT
import argparse
from train import map_model_name, trainer
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

NUM_SEEDS = 50

def per_layer_l2(model1, model2):
    layer_distances = {}
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        l2 = torch.norm(param1 - param2, p=2).item()
        layer_distances[name1] = l2
    return layer_distances

def compute_per_layer_distances(models_list1, models_list2):
    """
    Compute per-layer L2 distances for pairs of models across seeds.
    Returns a DataFrame: rows = seeds, columns = layers.
    """
    seed_pairs = []
    per_layer_distances = []
    num_models = len(models_list1)
    for i in range(num_models):
        for j in range(i, num_models):
            dist = per_layer_l2(models_list1[i], models_list2[j])
            per_layer_distances.append(dist)
            seed_pairs.append(f'seed_{i}_vs_seed_{j}')

    df = pd.DataFrame(per_layer_distances)
    df['seed_pair'] = seed_pairs
    df = df.set_index('seed_pair')
    return df

def cosine_similarity_model_weights(model1, model2):
    """
    Computes cosine similarity between the full flattened weight vectors of two models.
    """
    params1 = torch.cat([p.detach().flatten() for p in model1.parameters()])
    params2 = torch.cat([p.detach().flatten() for p in model2.parameters()])
    cos_sim = F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()
    return cos_sim

def l2_distance_model(model1, model2):
    total_l2 = 0.0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        total_l2 += torch.linalg.norm(param1 - param2, ord=2).item() ** 2
    return total_l2 ** 0.5

def l2_model(model):
    total_l2 = 0.0
    for name, param in model.named_parameters():
        total_l2 += torch.linalg.norm(param, ord=2).item() ** 2
    return total_l2 ** 0.5

def compute_l2_distances(models_list1, models_list2, skip_diagonals):
    """
    Compute pairwise L2 distances between a list of models.
    :param models: List of models to compare.
    :return: A matrix of L2 distances as values.
    """
    distances = []
    for i in range(len(models_list1)):
        distances.append([])
        for j in range(i, len(models_list2)):
            if (not skip_diagonals) or (i != j):
                dist = l2_distance_model(models_list1[i], models_list2[j])
                distances[i].append(dist)
    return distances

def compute_cosine_similarity_matrix(models_list1, models_list2):
    """
    Compute pairwise cosine similarities between two lists of models.
    Returns a DataFrame: rows = model i, columns = model j.
    """
    num_models1 = len(models_list1)
    num_models2 = len(models_list2)

    similarity_matrix = np.zeros((num_models1, num_models2))
    for i in range(num_models1):
        for j in range(num_models2):
            cos_sim = cosine_similarity_model_weights(models_list1[i], models_list2[j])
            similarity_matrix[i, j] = cos_sim

    df = pd.DataFrame(similarity_matrix, 
                      index=[f'seed_{i}' for i in range(num_models1)],
                      columns=[f'seed_{j}' for j in range(num_models2)])
    return df

def find_checkpoints(dir):
    "Find the latest checkpoints in our directory structure."
    latest_checkpoints = []
    seed_folders = [f for f in os.listdir(dir) if f.startswith('seed_')]

    for seed_folder in seed_folders:
        seed_path = os.path.join(dir, seed_folder)
        checkpoint_files = glob.glob(os.path.join(seed_path, 'epoch_*.pt'))
        if checkpoint_files:
            # Extract epoch numbers and find the largest one
            latest_checkpoint = max(checkpoint_files, 
                                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            latest_checkpoints.append(latest_checkpoint)
    return latest_checkpoints

def difference_pretrain_and_direct(grammar, dataset_size, config):
    dir = f'../data/{grammar}/{grammar}_{dataset_size}/{config.name}'
    direct_folder = os.path.join(dir, 'new')
    pretrain_folder = os.path.join(dir, 'continued')

    direct_checkpoints = find_checkpoints(direct_folder)
    pretrain_checkpoints = find_checkpoints(pretrain_folder)

    models_direct = list_of_models_from_checkpoints(direct_checkpoints, config)
    models_pretrain = list_of_models_from_checkpoints(pretrain_checkpoints, config)

    l2_distances = compute_l2_distances(models_direct, models_pretrain, skip_diagonals=False)
    l2_per_layer_distances = compute_per_layer_distances(models_direct, models_pretrain)
    cosine_distances = compute_cosine_similarity_matrix(models_direct, models_pretrain)

    return l2_distances, l2_per_layer_distances, cosine_distances

def difference_same_seed(grammar, dataset_size, train_type, config):
    dir = f'../data/{grammar}/{grammar}_{dataset_size}/{config.name}/{train_type}'
    checkpoints = find_checkpoints(dir)
    models = list_of_models_from_checkpoints(checkpoints, config)

    l2_distances = compute_l2_distances(models, models, skip_diagonals=True)
    per_layer_distances = compute_per_layer_distances(models, models)
    cosine_distances = compute_cosine_similarity_matrix(models, models)

    return l2_distances, per_layer_distances, cosine_distances

def list_of_models_from_checkpoints(checkpoints, config):
    models = []
    for checkpoint in checkpoints:
        model_instance = GPT(config)  # create a fresh model instance
        state_dict = torch.load(checkpoint)
        model_instance.load_state_dict(state_dict)
        models.append(model_instance)
    return models

def loop_over_seeds_and_train(grammar, dataset_size, subgrammar, 
                              num_epochs_direct, num_epochs_pretrain, 
                              config, device):
    "train model (once through pretrain and once directly), only saves weights at beginning and end of training"
    for i in range(NUM_SEEDS):
        #initialize the model with random weights
        torch.manual_seed(i)
        model = GPT(config).to(device)
        model.apply(model._init_weights)

        # train directly
        trainer(model, grammar, config, dataset_size, checkpoint_path=None, checkpoint_every=num_epochs_direct,
                num_epochs=num_epochs_direct, save_first_x_epochs=0, continue_from=num_epochs_pretrain, 
                continue_training=False, device=device, seed=i)

        # trainer(model, grammar, config, dataset_size, checkpoint_path=None, checkpoint_every=num_epochs_direct+num_epochs_pretrain,
        #         num_epochs=num_epochs_direct+num_epochs_pretrain, save_first_x_epochs=0, continue_from=0, 
        #         continue_training=False, device=device, seed=i)
            
        # train on subgrammar
        model = GPT(config).to(device) 
        model.apply(model._init_weights)

        trainer(model, subgrammar, config, dataset_size, checkpoint_path=None, checkpoint_every=num_epochs_pretrain,
                num_epochs=num_epochs_pretrain, save_first_x_epochs=0,continue_from=0, 
                continue_training=False, device=device, seed=i)

        # train on full grammar after pretraining
        checkpoint_path = f'../data/{subgrammar}/{subgrammar}_{dataset_size}/{config.name}/new/seed_{i}/epoch_{num_epochs_pretrain}.pt'
        trainer(model, grammar, config, dataset_size, checkpoint_path=checkpoint_path, checkpoint_every=num_epochs_direct,
                num_epochs=num_epochs_direct, save_first_x_epochs=0, continue_from=num_epochs_pretrain, 
                continue_training=True, device=device, seed=i) 

def plot_l2_distances(distances, grammar, dataset_size, train_type):
    # plot the distances as a distribution
    plt.figure(figsize=(10, 6))
    plt.hist([dist for sublist in distances for dist in sublist if dist>0], bins=50, alpha=0.7, color='blue')
    plt.title(f'L2 Distances Distribution ({train_type})\nGrammar: {grammar}, Dataset Size: {dataset_size}')
    plt.xlabel('L2 Distance')
    plt.ylabel('Frequency')
    plt.savefig(f'l2_distances_distribution_{grammar}_{dataset_size}_{train_type}.png') 

def plot_per_layer_l2_distances(df, grammar, dataset_size, train_type):
    """
    Plot per-layer L2 distances as a bar chart.
    :param df: DataFrame with per-layer distances.
    :param grammar: Grammar used for training.
    :param dataset_size: Size of the dataset.
    :param train_type: Type of training (e.g., 'new', 'continued').
    """
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(df, cmap="coolwarm", cbar=True)
    plt.title(f"Per-Layer L2 Distances Heatmap - {train_type}")
    plt.xlabel("Layers")
    plt.ylabel("Seeds")
    plt.tight_layout()
    
    # Create a proper filename and save with full path
    filename = f'per_layer_l2_distances_{grammar}_{dataset_size}_{train_type}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

def plot_cosine_similarity_heatmap(df, title, filename):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap='coolwarm', cbar=True, square=True, vmin=-1.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Seed")
    plt.ylabel("Seed")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Compute L2 distances between models.")
    parser.add_argument("--grammar", type=str, required=True, help="Grammar to use for training.")
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--subgrammar", type=str, required=True, help="Subgrammar to use for pretraining.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for training.")
    parser.add_argument("--num_epochs_direct", type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("--num_epochs_pretrain", type=int, default=10, help="Number of epochs to pretrain the model.")

    return parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()
    config = map_model_name(args.model)

    loop_over_seeds_and_train(args.grammar, 
                              args.dataset_size, 
                              args.subgrammar, 
                              args.num_epochs_direct, 
                              args.num_epochs_pretrain,
                              config, 
                              device)
    direct_l2_distances, direct_l2_per_layer_distances, direct_cosine_distances = difference_same_seed(args.grammar,
                         args.dataset_size,
                         train_type='new',
                         config=config)
    continued_l2_distances, continued_l2_per_layer_distances, continued_cosine_distances = difference_same_seed(args.grammar,
                        args.dataset_size,
                        train_type='continued',
                        config=config)
    pretrain_vs_direct_l2_distances, pretrain_vs_direct_l2_per_layer_distances, pretrain_vs_direct_cosine_distances = difference_pretrain_and_direct(args.grammar, args.dataset_size, config)


    plot_l2_distances(direct_l2_distances, args.grammar, args.dataset_size, 'direct')
    plot_l2_distances(continued_l2_distances, args.grammar, args.dataset_size, 'continued')
    plot_l2_distances(pretrain_vs_direct_l2_distances, args.grammar, args.dataset_size, 'pretrain_vs_direct')

    plot_per_layer_l2_distances(direct_l2_per_layer_distances, args.grammar, args.dataset_size, 'direct')
    plot_per_layer_l2_distances(continued_l2_per_layer_distances, args.grammar, args.dataset_size, 'continued')
    plot_per_layer_l2_distances(pretrain_vs_direct_l2_per_layer_distances, args.grammar, args.dataset_size, 'pretrain_vs_direct')

    plot_cosine_similarity_heatmap(direct_cosine_distances, 
                                    title=f'Cosine Similarity Heatmap - Direct Training\nGrammar: {args.grammar}, Dataset Size: {args.dataset_size}', 
                                    filename=f'cosine_similarity_heatmap_direct_{args.grammar}_{args.dataset_size}.png')
    plot_cosine_similarity_heatmap(continued_cosine_distances, 
                                    title=f'Cosine Similarity Heatmap - Continued Training\nGrammar: {args.grammar}, Dataset Size: {args.dataset_size}', 
                                    filename=f'cosine_similarity_heatmap_continued_{args.grammar}_{args.dataset_size}.png')
    plot_cosine_similarity_heatmap(pretrain_vs_direct_cosine_distances, 
                                    title=f'Cosine Similarity Heatmap - Pretrain vs Direct\nGrammar: {args.grammar}, Dataset Size: {args.dataset_size}', 
                                    filename=f'cosine_similarity_heatmap_pretrain_vs_direct_{args.grammar}_{args.dataset_size}.png')    

if __name__ == "__main__":
    main()