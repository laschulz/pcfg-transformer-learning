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
import json
from transformers import PreTrainedTokenizerFast
from scipy.stats import spearmanr

NUM_SEEDS = 50

# per-layer L2 distance between two models
def per_layer_l2(model1, model2):
    layer_distances = {}
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        if name1 in {'transformer.wte.weight', 'transformer.wpe.weight' }:
            continue
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

# total L2 distance between two models
def l2_distance_model(model1, model2):
    total_l2 = 0.0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2
        if name1 in {'transformer.wte.weight', 'transformer.wpe.weight'}:
            continue
        total_l2 += torch.linalg.norm(param1 - param2, ord=2).item() ** 2
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

# cosine similarity between two models
def cosine_similarity_model_weights(model1, model2):
    """
    Computes cosine similarity between the full flattened weight vectors of two models.
    """
    params1 = torch.cat([p.detach().flatten() for p in model1.parameters()])
    params2 = torch.cat([p.detach().flatten() for p in model2.parameters()])
    cos_sim = F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()
    return cos_sim

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

# RSA (Representational Similarity Analysis) for per-layer weights
def _rowwise_pearson_corr(X: torch.Tensor) -> torch.Tensor:
    """
    X: (N, D). Returns (N, N) Pearson corr between rows.
    Handles zero-variance rows by adding eps.
    """
    X = X.float()
    mean = X.mean(dim=1, keepdim=True)
    std  = X.std(dim=1, keepdim=True)
    Xz = (X - mean) / (std + 1e-8)      # (N, D)
    # cosine of z-scored rows == Pearson
    return Xz @ Xz.T / X.shape[1]

def compute_rdms_per_layer(seq_reps: dict, device=None) -> dict:
    """
    For each layer key: RDM = 1 - Pearson(rowwise). Returns CPU numpy arrays.
    """
    rdms = {}
    for k, X in seq_reps.items():
        if not isinstance(X, torch.Tensor) or X.numel() == 0:
            continue
        X = X.to(device)
        C = _rowwise_pearson_corr(X).clamp(min=-1.0, max=1.0)  # (N, N)
        D = 1.0 - C
        rdms[k] = D.detach().cpu().numpy()
    return rdms

# def subsample_seq_reps(seq_reps: dict, max_seqs: int = None, seed: int = 0) -> dict:
#     """Pick same subset of sequences across all layers for a model."""
#     if max_seqs is None:
#         return seq_reps
#     # find N from first present layer
#     first = next((v for v in seq_reps.values() if isinstance(v, torch.Tensor) and v.numel() > 0), None)
#     if first is None:
#         return seq_reps
#     N = first.shape[0]
#     if N <= max_seqs:
#         return seq_reps
#     g = torch.Generator().manual_seed(seed)
#     idx = torch.randperm(N, generator=g)[:max_seqs]
#     out = {}
#     for k, X in seq_reps.items():
#         out[k] = X[idx] if isinstance(X, torch.Tensor) and X.shape[0] == N else X
#     return out

def _num_sequences(seq_reps: dict) -> int:
    """Return N for the first valid (N, D) tensor found, else 0."""
    for v in seq_reps.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.numel() > 0:
            return v.shape[0]
    return 0

def _subsample_to_N(seq_reps: dict, N_target: int, seed: int = 0) -> dict:
    """Deterministically subsample (or keep) exactly N_target rows for all layers present."""
    if N_target <= 0:
        return seq_reps
    N_cur = _num_sequences(seq_reps)
    if N_cur == 0:
        return seq_reps
    if N_cur == N_target:
        return seq_reps

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N_cur, generator=g)[:N_target]
    out = {}
    for k, X in seq_reps.items():
        if isinstance(X, torch.Tensor) and X.ndim == 2 and X.shape[0] == N_cur:
            out[k] = X[idx]
        else:
            out[k] = X
    return out

def _common_subsample(seq_reps_list1, seq_reps_list2, max_seqs=None, seed=0):
    """
    Force a common N across *all* models in both lists, capped by max_seqs if provided.
    Returns two new lists with each dict subsampled to the same N_common.
    """
    Ns = []
    for sr in (seq_reps_list1 + seq_reps_list2):
        Ns.append(_num_sequences(sr))
    N_min = min([n for n in Ns if n > 0], default=0)
    if max_seqs is not None:
        N_common = min(N_min, max_seqs)
    else:
        N_common = N_min

    out1 = [_subsample_to_N(sr, N_common, seed=seed) for sr in seq_reps_list1]
    out2 = [_subsample_to_N(sr, N_common, seed=seed) for sr in seq_reps_list2]
    return out1, out2, N_common

def _upper_tri_vec(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]

def rsa_similarity_between_rdms(rdm1: np.ndarray, rdm2: np.ndarray, method) -> float:
    """
    Spearman correlation between vectorized upper triangles of two RDMs.
    """
    v1 = _upper_tri_vec(rdm1)
    v2 = _upper_tri_vec(rdm2)
    if method == "spearman":
        rho, _ = spearmanr(v1, v2)
        return float(rho)
    else:
        # Pearson fallback
        v1m = v1 - v1.mean()
        v2m = v2 - v2.mean()
        denom = (np.linalg.norm(v1m) * np.linalg.norm(v2m) + 1e-12)
        return float((v1m @ v2m) / denom)

def compute_rsa_matrix(seq_reps_list1, seq_reps_list2, method, max_seqs=300, seed=0, device=None, diagonal_only=False):
    """
    Efficient RSA: subsample once per model, precompute RDMs once per model,
    then compare across pairs.
    """

    seq_reps_list1, seq_reps_list2, N_common = _common_subsample(
        seq_reps_list1, seq_reps_list2, max_seqs=max_seqs, seed=seed
    )

    rdms_list1 = [compute_rdms_per_layer(sr, device=device) for sr in seq_reps_list1]
    rdms_list2 = [compute_rdms_per_layer(sr, device=device) for sr in seq_reps_list2]

    keys = sorted(set().union(*[r.keys() for r in rdms_list1 + rdms_list2]))
    seed_pairs, rows = [], []
    if diagonal_only:
        for i in range(len(rdms_list1)):
            for k in keys:
                if k not in rdms_list1[i] or k not in rdms_list2[i]:
                    continue
                row = {k: rsa_similarity_between_rdms(rdms_list1[i][k], rdms_list2[i][k], method=method)}
                rows.append(row)
                seed_pairs.append(f"seed_{i}_vs_seed_{i}")
    else:
        for i in range(len(rdms_list1)):
            for j in range(i, len(rdms_list2)):
                row = {}
                for k in keys:
                    if k not in rdms_list1[i] or k not in rdms_list2[j]:
                        continue
                    row[k] = rsa_similarity_between_rdms(rdms_list1[i][k], rdms_list2[j][k], method=method)
                rows.append(row)
                seed_pairs.append(f"seed_{i}_vs_seed_{j}")
    return pd.DataFrame(rows, index=seed_pairs)

def average_rsa(df: pd.DataFrame, skip_diagonal=True):
    df = df.apply(pd.to_numeric, errors="coerce")
    if skip_diagonal and isinstance(df.index, pd.Index):
        pairs = df.index.to_series().str.extract(r"seed_(\d+)_vs_seed_(\d+)").astype(float)
        offdiag = pairs[0] != pairs[1]
        df = df[offdiag.values]

    overall_mean = df.stack(dropna=True).mean()
    overall_std  = df.stack(dropna=True).std()
    per_pair_mean  = df.mean(axis=1, skipna=True)
    per_layer_mean = df.mean(axis=0, skipna=True)
    return {
        "overall_mean": float(overall_mean),
        "overall_std": float(overall_std),
        "per_pair_mean": per_pair_mean,
        "per_layer_mean": per_layer_mean,
    }

def plot_per_layer_rsa_heatmap(df, grammar, dataset_size, train_type, config):
    col_mean = df.mean(axis=0, skipna=True)
    overall  = df.stack(dropna=True).mean()

    def _fmt(v): return "NA" if pd.isna(v) else f"{v:.3f}"
    labels = [f"{col}\n{_fmt(col_mean[col])}" for col in df.columns]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df, ax=ax, cmap="magma", cbar=True, vmin=-1, vmax=1)  # RSA can be negative

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10, linespacing=1.2)
    ax.tick_params(axis='x', pad=12)
    fig.subplots_adjust(bottom=0.22)

    fig.text(0.995, 0.01, f"Overall mean RSA: {overall:.3f}",
             ha='center', va='bottom',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0), fontsize=10)

    ax.set_xlabel("Layer (mean shown below)")
    ax.set_ylabel("Seed pair")
    ax.set_title(f"Per-Layer RSA – {train_type}\nGrammar: {grammar}, Dataset Size: {dataset_size}")

    fig.savefig(f"../results/weight_space/pl_rsa_{grammar}_{dataset_size}_{config.name}_{train_type}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

# CKA (Centered Kernel Alignment) for per-layer weights between two models
def collect_activations(model, tokenizer, base_dir, agg=None):
    # 1) Prepare empty buffers
    n_layers = len(model.transformer.h)
    acts = {f"attn_{i}": [] for i in range(n_layers)}
    acts.update({f"mlp_{i}": [] for i in range(n_layers)})

    # 2) Define hook factory
    def get_hook(key):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            B, L, D = out.shape
            if agg is None:
                rep = out.reshape(B*L, D)
            elif agg == "mean":
                rep = out.mean(dim=1)  # (B, D) -> (1, D)
            elif agg == "mean_unit":
                z = out / (out.norm(dim=-1, keepdim=True) + 1e-8)  # per-token unit
                rep = z.mean(dim=1)  
            acts[key].append(rep.detach().cpu())
        return hook

    # 3) Register hooks
    hooks = []
    for i, block in enumerate(model.transformer.h):
        # be robust to naming
        attn_mod = getattr(block, "attn", None) or getattr(block, "mha", None) or getattr(block, "self_attn", None)
        mlp_mod  = getattr(block, "mlp",  None) or getattr(block, "ffn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(get_hook(f"attn_{i}")))
        if mlp_mod is not None:
            hooks.append(mlp_mod.register_forward_hook(get_hook(f"mlp_{i}")))

    # 4) Run forward passes
    with open(f"{base_dir}/test.jsonl", 'r') as f:
            sequences = [json.loads(line)["sequence"] for line in f]
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            enc = tokenizer(seq, return_tensors="pt")
            input_ids = enc["input_ids"]                    # shape (1, L)
            _ = model(input_ids)  

    # 5) Remove hooks
    for h in hooks:
        h.remove()

    # 6) Concatenate into single tensor per layer
    for k in list(acts.keys()):
        acts[k] = torch.cat(acts[k], dim=0)

    return acts

def compute_cka(X, Y):
    """
    Computes linear CKA between two representations X and Y.
    X, Y: (n_samples, d) tensors
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    num = torch.linalg.norm(X.T @ Y, ord="fro")**2
    den = torch.linalg.norm(X.T @ X, ord="fro") * torch.linalg.norm(Y.T @ Y, ord="fro") + 1e-12
    return float((num / den).item())

def compute_cka_matrix(acts_list1, acts_list2, diagonal_only=False):
    """
    Compute the CKA distances between activations of two sets of models.
    activations1, activations2: lists of activations for each model.
    Returns a list of L2 distances for each layer.
    """
    keys = sorted(set().union(*[a.keys() for a in acts_list1 + acts_list2]))
    seed_pairs, rows = [], []
    if diagonal_only:
        for i in range(len(acts_list1)):
            for k in keys:
                if k not in acts_list1[i] or k not in acts_list2[i]:
                    continue
                row = {k: compute_cka(acts_list1[i][k], acts_list2[i][k])}
                rows.append(row)
                seed_pairs.append(f"seed_{i}_vs_seed_{i}")
    else:
        for i in range(len(acts_list1)):
            for j in range(i, len(acts_list2)):
                row = {}
                for k in keys:
                    if k not in acts_list1[i] or k not in acts_list2[j]:
                        continue
                    row[k] = compute_cka(acts_list1[i][k], acts_list2[j][k])
                rows.append(row)
                seed_pairs.append(f"seed_{i}_vs_seed_{j}")
    df = pd.DataFrame(rows, index=seed_pairs)
    return df

def average_cka(df: pd.DataFrame, skip_diagonal=True):
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Optionally drop i==j pairs (self-similarity ~1.0)
    if skip_diagonal and isinstance(df.index, pd.Index):
        pairs = df.index.to_series().str.extract(r"seed_(\d+)_vs_seed_(\d+)").astype(float)
        offdiag = pairs[0] != pairs[1]
        df = df[offdiag.values]

    overall_mean = df.stack(dropna=True).mean()
    overall_std  = df.stack(dropna=True).std()

    per_pair_mean  = df.mean(axis=1, skipna=True)  # avg across layers for each seed pair
    per_layer_mean = df.mean(axis=0, skipna=True)  # avg across pairs for each layer

    return {
        "overall_mean": float(overall_mean),
        "overall_std": float(overall_std),
        "per_pair_mean": per_pair_mean,    # Series indexed by seed_i_vs_seed_j
        "per_layer_mean": per_layer_mean,  # Series indexed by layer keys
    }

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

def difference_pretrain_and_direct(config, tokenizer, dir1, dir2, base_dir):
    direct_folder = dir1
    pretrain_folder = dir2

    direct_checkpoints = find_checkpoints(direct_folder)
    pretrain_checkpoints = find_checkpoints(pretrain_folder)

    models_direct = list_of_models_from_checkpoints(direct_checkpoints, config)
    models_pretrain = list_of_models_from_checkpoints(pretrain_checkpoints, config)

    #load activations for all models
    act_direct = [collect_activations(model, tokenizer, base_dir, agg=None) for model in models_direct] #was without mean
    act_pretrain = [collect_activations(model, tokenizer, base_dir, agg=None) for model in models_pretrain] # was without mean
    cka_distances = compute_cka_matrix(act_direct, act_pretrain, diagonal_only=True)

        # RSA sequence-level reps
    seq_direct = [collect_activations(m, tokenizer, base_dir, agg="mean")
                  for m in models_direct]
    seq_pretrain = [collect_activations(m, tokenizer, base_dir, agg="mean")
                    for m in models_pretrain]
    rsa_distances = compute_rsa_matrix(seq_direct, seq_pretrain, method="spearman", diagonal_only=True)


    l2_distances = compute_l2_distances(models_direct, models_pretrain, skip_diagonals=False)
    l2_per_layer_distances = compute_per_layer_distances(models_direct, models_pretrain)
    cosine_distances = compute_cosine_similarity_matrix(models_direct, models_pretrain)

    return l2_distances, l2_per_layer_distances, cosine_distances, cka_distances, rsa_distances

def difference_same_seed(train_type, config, tokenizer, base_dir, test_dir):
    dir = f'{base_dir}/{config.name}/{train_type}'
    checkpoints = find_checkpoints(dir)
    models = list_of_models_from_checkpoints(checkpoints, config)

    # load activations 
    acts = [collect_activations(model, tokenizer, test_dir) for model in models]
    cka_distances = compute_cka_matrix(acts, acts, diagonal_only=False)

    seq_reps = [collect_activations(m, tokenizer, test_dir, agg="mean") for m in models]
    rsa_distances = compute_rsa_matrix(seq_reps, seq_reps, method="spearman", diagonal_only=False)

    l2_distances = compute_l2_distances(models, models, skip_diagonals=True)
    per_layer_distances = compute_per_layer_distances(models, models)
    cosine_distances = compute_cosine_similarity_matrix(models, models)

    return l2_distances, per_layer_distances, cosine_distances, cka_distances, rsa_distances

def list_of_models_from_checkpoints(checkpoints, config):
    models = []
    for checkpoint in checkpoints:
        model_instance = GPT(config)  # create a fresh model instance
        state_dict = torch.load(checkpoint)
        model_instance.load_state_dict(state_dict)
        models.append(model_instance)
    return models

def loop_over_seeds_and_train(grammar, dataset_size, grammar_startsymbol, subgrammar_startsymbol,
                              num_epochs_direct, num_epochs_pretrain, 
                              config, device, start_seed):
    "train model (once through pretrain and once directly), only saves weights at beginning and end of training"
    for i in range(start_seed, NUM_SEEDS):
        #initialize the model with random weights
        print("Training seed:", i)
        torch.manual_seed(i)
        model = GPT(config).to(device)
        model.apply(model._init_weights)

        # train directly
        trainer(model, grammar, config, f'{dataset_size}_{grammar_startsymbol}', checkpoint_path=None, checkpoint_every=num_epochs_direct,
                num_epochs=num_epochs_direct, save_first_x_epochs=0, continue_from=num_epochs_pretrain, 
                continue_training=False, device=device, seed=i, safe_only_last=True)

        # trainer(model, grammar, config, dataset_size, checkpoint_path=None, checkpoint_every=num_epochs_direct+num_epochs_pretrain,
        #         num_epochs=num_epochs_direct+num_epochs_pretrain, save_first_x_epochs=0, continue_from=0, 
        #         continue_training=False, device=device, seed=i)
            
        # train on subgrammar
        # model = GPT(config).to(device) 
        # model.apply(model._init_weights)

        # trainer(model, grammar, config, f'{dataset_size}_{subgrammar_startsymbol}', checkpoint_path=None, checkpoint_every=num_epochs_pretrain,
        #         num_epochs=num_epochs_pretrain, save_first_x_epochs=0,continue_from=0, 
        #         continue_training=False, device=device, seed=i, safe_only_last=True)

        # # train on full grammar after pretraining
        # checkpoint_path = f'../data/{grammar}/{grammar}_{dataset_size}_{subgrammar_startsymbol}/{config.name}/new/seed_{i}/epoch_{num_epochs_pretrain}_0.pt'
        # trainer(model, grammar, config, f'{dataset_size}_{grammar_startsymbol}', checkpoint_path=checkpoint_path, checkpoint_every=num_epochs_direct,
        #         num_epochs=num_epochs_direct, save_first_x_epochs=0, continue_from=num_epochs_pretrain, 
        #         continue_training=True, device=device, seed=i, safe_only_last=True) 

def plot_l2_distances(distances, grammar, dataset_size, train_type):
    # plot the distances as a distribution
    plt.figure(figsize=(10, 6))
    plt.hist([dist for sublist in distances for dist in sublist if dist>0], bins=50, alpha=0.7, color='blue')
    plt.title(f'L2 Distances Distribution ({train_type})\nGrammar: {grammar}, Dataset Size: {dataset_size}')
    plt.xlabel('L2 Distance')
    plt.ylabel('Frequency')
    plt.savefig(f'../results/weight_space/l2_{grammar}_{dataset_size}_{train_type}.png') 
    plt.close()

def plot_per_layer_l2_distances(df, grammar, dataset_size, train_type):

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(df, cmap="coolwarm", cbar=True)
    plt.title(f"Per-Layer L2 Distances Heatmap - {train_type}\nGrammar: {grammar}, Dataset Size: {dataset_size}")
    plt.xlabel("Layers")
    plt.ylabel("Seeds")
    plt.tight_layout()
    
    # Create a proper filename and save with full path
    filename = f'../results/weight_space/pl_l2_{grammar}_{dataset_size}_{train_type}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_cosine_similarity_heatmap(df, grammar, dataset_size, train_type):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap='coolwarm', cbar=True, square=True, vmin=-1.0, vmax=1.0)
    plt.title(f"Cosine Similarity Heatmap - {train_type}\nGrammar: {grammar}, Dataset Size: {dataset_size}")
    plt.xlabel("Seed")
    plt.ylabel("Seed")
    plt.tight_layout()
    filename = f'../results/weight_space/cos_sim_{grammar}_{dataset_size}_{train_type}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_per_layer_cka_heatmap(df, grammar, dataset_size, train_type, config):
    # ensure numeric
    # df = df.apply(pd.to_numeric, errors='coerce')
    col_mean = df.mean(axis=0, skipna=True)
    overall  = df.stack(dropna=True).mean()

    # build two-line tick labels: "layer\nmean"
    def _fmt(v): return "NA" if pd.isna(v) else f"{v:.3f}"
    labels = [f"{col}\n{_fmt(col_mean[col])}" for col in df.columns]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df, ax=ax, cmap="viridis", cbar=True, vmin=0, vmax=1)

    # show labels at bottom; add padding so the second line clears the plot
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10, linespacing=1.2)
    ax.tick_params(axis='x', pad=12)         # space between heatmap and labels
    fig.subplots_adjust(bottom=0.22)         # extra bottom margin for two lines

    # overall mean outside the axes (no overlay)
    fig.text(0.995, 0.01, f"Overall mean CKA: {overall:.3f}",
             ha='center', va='bottom',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0), fontsize=10)

    ax.set_xlabel("Layer (mean shown below)")
    ax.set_ylabel("Seed pair")
    ax.set_title(f"Per-Layer CKA – {train_type}\nGrammar: {grammar}, Dataset Size: {dataset_size}")

    fig.savefig(f"../results/weight_space/pl_cka_{grammar}_{dataset_size}_{config.name}_{train_type}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute L2 distances between models.")
    parser.add_argument("--grammar", type=str, required=True, help="Grammar to use for training.")
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, help="Model name to use for training.")
    parser.add_argument("--num_epochs_direct", type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("--num_epochs_pretrain", type=int, default=10, help="Number of epochs to pretrain the model.")
    parser.add_argument("--grammar_startsymbol", type=str, required=True, help="Start symbol for the grammar.")
    parser.add_argument("--subgrammar_startsymbol", type=str, required=True, help="End symbol for the grammar.")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed for training.")

    return parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()
    config = map_model_name(args.model)

    base_dir = f'../data/{args.grammar}/{args.grammar}_{args.dataset_size}'

    # loop_over_seeds_and_train(args.grammar, 
    #                           args.dataset_size, 
    #                           args.grammar_startsymbol,
    #                           args.subgrammar_startsymbol,
    #                           args.num_epochs_direct, 
    #                           args.num_epochs_pretrain,
    #                           config, 
    #                           device,
    #                           start_seed=args.start_seed)

    
    # load tokenizer
    tokenizer_path = f'{base_dir}_{args.grammar_startsymbol}/tokenizer.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, bos_token="<|bos|>", eos_token="<|eos|>")

    # direct_l2, direct_l2_pl, direct_cos, direct_cka, direct_rsa = difference_same_seed(train_type='new',
    #                      config=config,
    #                      tokenizer=tokenizer,
    #                      base_dir=f"{base_dir}_{args.subgrammar_startsymbol}",
    #                      test_dir=f"{base_dir}_{args.subgrammar_startsymbol}")
    # continued_l2, continued_l2_pl, continued_cos, continued_cka, continued_rsa = difference_same_seed(train_type='continued',
    #                     config=config,
    #                     tokenizer=tokenizer,
    #                     base_dir=f"{base_dir}_{args.grammar_startsymbol}",
    #                     test_dir=f"{base_dir}_{args.subgrammar_startsymbol}")
    # pretrain_vs_direct_l2, pretrain_vs_direct_l2_pl, pretrain_vs_direct_cos, pretrain_vs_direct_cka, pretrain_vs_direct_rsa = difference_pretrain_and_direct(config, 
    #                                                                                 tokenizer=tokenizer, 
    #                                                                                 dir1=f'{base_dir}_{args.subgrammar_startsymbol}/{args.model}/new', 
    #                                                                                 dir2=f'{base_dir}_{args.grammar_startsymbol}/{args.model}/continued',
    #                                                                                 base_dir =f'{base_dir}_{args.subgrammar_startsymbol}'
    #                                                                                 )

    direct_l2, direct_l2_pl, direct_cos, direct_cka, direct_rsa = difference_same_seed(train_type='new',
                         config=config,
                         tokenizer=tokenizer,
                         base_dir=f"{base_dir}_{args.grammar_startsymbol}",
                         test_dir=f"{base_dir}_{args.grammar_startsymbol}")
    continued_l2, continued_l2_pl, continued_cos, continued_cka, continued_rsa = difference_same_seed(train_type='continued',
                        config=config,
                        tokenizer=tokenizer,
                        base_dir=f"{base_dir}_{args.grammar_startsymbol}",
                        test_dir=f"{base_dir}_{args.grammar_startsymbol}")
    pretrain_vs_direct_l2, pretrain_vs_direct_l2_pl, pretrain_vs_direct_cos, pretrain_vs_direct_cka, pretrain_vs_direct_rsa = difference_pretrain_and_direct(config, 
                                                                                    tokenizer=tokenizer, 
                                                                                    dir1=f'{base_dir}_{args.grammar_startsymbol}/{args.model}/new', 
                                                                                    dir2=f'{base_dir}_{args.grammar_startsymbol}/{args.model}/continued',
                                                                                    base_dir =f'{base_dir}_{args.grammar_startsymbol}'
                                                                                    )

    plot_l2_distances(direct_l2, args.grammar, args.dataset_size, 'direct')
    plot_l2_distances(continued_l2, args.grammar, args.dataset_size, 'continued')
    plot_l2_distances(pretrain_vs_direct_l2, args.grammar, args.dataset_size, 'pretrain_vs_direct')

    plot_per_layer_l2_distances(direct_l2_pl, args.grammar, args.dataset_size, 'direct')
    plot_per_layer_l2_distances(continued_l2_pl, args.grammar, args.dataset_size, 'continued')
    plot_per_layer_l2_distances(pretrain_vs_direct_l2_pl, args.grammar, args.dataset_size, 'pretrain_vs_direct')

    plot_cosine_similarity_heatmap(direct_cos, args.grammar, args.dataset_size, 'direct')
    plot_cosine_similarity_heatmap(continued_cos, args.grammar, args.dataset_size, 'continued')
    plot_cosine_similarity_heatmap(pretrain_vs_direct_cos, args.grammar, args.dataset_size, 'pretrain_vs_direct')    

    plot_per_layer_cka_heatmap(direct_cka, args.grammar, args.dataset_size, 'direct', config)
    plot_per_layer_cka_heatmap(continued_cka, args.grammar, args.dataset_size, 'continued', config)
    plot_per_layer_cka_heatmap(pretrain_vs_direct_cka, args.grammar, args.dataset_size, 'pretrain_vs_direct', config) 

    plot_per_layer_rsa_heatmap(direct_rsa, args.grammar, args.dataset_size, 'direct', config)
    plot_per_layer_rsa_heatmap(continued_rsa, args.grammar, args.dataset_size, 'continued', config)
    plot_per_layer_rsa_heatmap(pretrain_vs_direct_rsa, args.grammar, args.dataset_size, 'pretrain_vs_direct', config)

if __name__ == "__main__":
    main()