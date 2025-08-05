"This file computes the L2-Norm between two models' parameters."
"We assume that the training data already exists (and the training data is fixed)"
import torch
from model import GPT
import argparse
from train import map_model_name, trainer

NUM_SEEDS = 10

def l2_distance_model(model1, model2):
    total_l2 = 0.0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        total_l2 += torch.norm(param1 - param2, p=2).item() ** 2
    return total_l2 ** 0.5

def compute_l2_distances(models):
    """
    Compute pairwise L2 distances between a list of models.
    :param models: List of models to compare.
    :return: A matrix of L2 distances as values.
    """
    distances = []
    for i in range(len(models)):
        distances.append([])
        for j in range(len(models)):
            if i != j:
                dist = l2_distance_model(models[i], models[j])
                distances[i].append(dist)
            else:
                distances[i].append(0.0)
    return distances


def difference_pretrain_and_direct():
    pass

def difference_same_seed():
    pass


def loop_over_seeds_and_train(model, grammar, dataset_size, subgrammar, 
                              num_epochs_direct, num_epochs_pretrain, 
                              config, device):
    for i in range(NUM_SEEDS):
        # train model (once through pretrain and once directly)
        # ensure that last version of direct is saved, as well as after pretrain and after "fine-tuning"
        # save these model weights 
        beginning_weights = model.state_dict()

        # train directly
        trainer(model, grammar, config, dataset_size, checkpoint_path=None, 
                num_epochs=num_epochs_direct, continue_from=num_epochs_pretrain, 
                continue_training=False, device=device, seed=i)
            
        # train on subgrammar
        model.load_state_dict(beginning_weights) # reset to original weights (such that we can compare)
        trainer(model, subgrammar, config, dataset_size, checkpoint_path=None,
                num_epochs=num_epochs_pretrain, continue_from=0, 
                continue_training=False, device=device, seed=i)

        # train on full grammar after pretraining
        checkpoint_path = f'../data/{subgrammar}/{subgrammar}_{dataset_size}/{config.name}/new/seed_{i}/epoch_{num_epochs_pretrain}.pt'
        trainer(model, grammar, config, dataset_size, checkpoint_path=checkpoint_path,
                num_epochs=num_epochs_direct, continue_from=num_epochs_pretrain, 
                continue_training=True, device=device, seed=i) 

    # once trained, we compute the L2-distances

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

    loop_over_seeds_and_train(model, 
                              args.grammar, 
                              args.dataset_size, 
                              args.subgrammar, 
                              args.num_epochs_direct, 
                              args.num_epochs_pretrain,
                              config, 
                              device)
    

