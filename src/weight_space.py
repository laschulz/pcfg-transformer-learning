"This file computes the L2-Norm between two models' parameters."
"We assume that the training data already exists (and the training data is fixed)"
import torch
from model import GPT
import argparse
from train import map_model_name, trainer
import os
import glob
import matplotlib.pyplot as plt

NUM_SEEDS = 50

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
    # distances = compute_l2_distances(models_direct, models_pretrain, skip_diagonals=False)

    direct_distances  = [l2_model(model) for model in models_direct]
    pretrain_distances = [l2_model(model) for model in models_pretrain]

    #return distances

    return direct_distances, pretrain_distances

def difference_same_seed(grammar, dataset_size, train_type, config):
    dir = f'../data/{grammar}/{grammar}_{dataset_size}/{config.name}/{train_type}'
    checkpoints = find_checkpoints(dir)
    models = list_of_models_from_checkpoints(checkpoints, config)

    distances = compute_l2_distances(models, models, skip_diagonals=True)

    return distances

def list_of_models_from_checkpoints(checkpoints, config):
    models = []
    for checkpoint in checkpoints:
        model_instance = GPT(config)  # create a fresh model instance
        state_dict = torch.load(checkpoint)
        model_instance.load_state_dict(state_dict)
        models.append(model_instance)
    return models

def loop_over_seeds_and_train(model, grammar, dataset_size, subgrammar, 
                              num_epochs_direct, num_epochs_pretrain, 
                              config, device):
    "train model (once through pretrain and once directly), only saves weights at beginning and end of training"
    for i in range(NUM_SEEDS):
        #initialize the model with random weights
        torch.manual_seed(i)
        model.apply(model._init_weights)
        beginning_weights = model.state_dict()

        # train directly
        trainer(model, grammar, config, dataset_size, checkpoint_path=None, checkpoint_every=num_epochs_direct,
                num_epochs=num_epochs_direct, safe_first_x_epochs=0, continue_from=num_epochs_pretrain, 
                continue_training=False, device=device, seed=i)
            
        # train on subgrammar
        model.load_state_dict(beginning_weights) # reset to original weights (such that we can compare)
        trainer(model, subgrammar, config, dataset_size, checkpoint_path=None, checkpoint_every=num_epochs_pretrain,
                num_epochs=num_epochs_pretrain, safe_first_x_epochs=0,continue_from=0, 
                continue_training=False, device=device, seed=i)

        # train on full grammar after pretraining
        checkpoint_path = f'../data/{subgrammar}/{subgrammar}_{dataset_size}/{config.name}/new/seed_{i}/epoch_{num_epochs_pretrain}.pt'
        trainer(model, grammar, config, dataset_size, checkpoint_path=checkpoint_path, checkpoint_every=num_epochs_direct,
                num_epochs=num_epochs_direct, safe_first_x_epochs=0, continue_from=num_epochs_pretrain, 
                continue_training=True, device=device, seed=i) 

def plot_l2_distances(distances, grammar, dataset_size, train_type):
    # plot the distances as a distribution
    plt.figure(figsize=(10, 6))
    plt.hist([dist for dist in distances], bins=50, alpha=0.7, color='blue')
    #plt.hist([dist for sublist in distances for dist in sublist if dist>0], bins=50, alpha=0.7, color='blue')
    plt.title(f'L2 Distances Distribution ({train_type})\nGrammar: {grammar}, Dataset Size: {dataset_size}')
    plt.xlabel('L2 Distance')
    plt.ylabel('Frequency')
    plt.savefig(f'l2_distances_distribution_{grammar}_{dataset_size}_{train_type}.png') 

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
    model = GPT(config).to(device)

    # loop_over_seeds_and_train(model, 
    #                           args.grammar, 
    #                           args.dataset_size, 
    #                           args.subgrammar, 
    #                           args.num_epochs_direct, 
    #                           args.num_epochs_pretrain,
    #                           config, 
    #                           device)
    # distances_direct = difference_same_seed(args.grammar,
    #                      args.dataset_size,
    #                      train_type='new',
    #                      config=config)
    # distances_continued = difference_same_seed(args.grammar,
    #                     args.dataset_size,
    #                     train_type='continued',
    #                     config=config)
    # distances_pretrain_vs_direct = difference_pretrain_and_direct(args.grammar, args.dataset_size, config)

    distances_direct, distances_continued = difference_pretrain_and_direct(args.grammar, args.dataset_size, config)

    plot_l2_distances(distances_direct, args.grammar, args.dataset_size, 'direct')
    plot_l2_distances(distances_continued, args.grammar, args.dataset_size, 'continued')
    #plot_l2_distances(distances_pretrain_vs_direct, args.grammar, args.dataset_size, 'pretrain_vs_direct')

if __name__ == "__main__":
    main()