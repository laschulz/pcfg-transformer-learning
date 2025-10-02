# pcfg-transformer-learning

This repository contains tools for analyzing transformer models trained on probabilistic context-free grammars (PCFGs).  
It was used to run the experiments in the paper:

[Unraveling Syntax: How Language Models Learn Context-Free Grammars](Learning_of_CFGs_ICLR_2026.pdf)

---

## Repository Structure
- **`bash_files`**
  Shell scripts to run the main experiments described in the paper.

- **`data`**
  Stores all training checkpoints.

- **`results`**
Stores analysis outputs and figures.

- **`src`**
  Contains all scripts of code to run the experiments


## Key Files

- **`def_pcfgs.py`**  
  Contains all PCFG grammars defined for the project.  
  If you wish to add more grammars, extend the `GRAMMARS` dictionary in this file.

- **`generate_pcfg.py`**  
  This is the main script for dataset preparation; generates the tokenizer, training set, and test set for the predefined grammars. The test set size is fixed at 1,000 datapoints, while the training set size is variable and specified by the user. The tokenizers do not have an UNK or PAD token.

- **`train.py`**
    Contains the function `trainer` that trains the transformer model. 
    It supports two modes:

    - **Training from scratch**:  
        Starts a fresh training run, with results stored in the `new` directory.

    - **Continuing from a checkpoint**: Loads the model from an existing checkpoint and continues training. All pre-existing checkpoints from the source directory are copied into the new target directory, and results are stored under `continued`.
  
    This directory structure ensures a clear distinction between newly initialized models and those extended from checkpoints.  
    If the same model type is trained on the same PCFG and start symbol, the corresponding folder will be overwritten!

- **`analysis_hierarchy.py`** 
  This is the main script to analyze the models *after* being trained on a grammar. 
  To analyze a specific subgrammar, provide its name; otherwise, set `--subgrammar` to the name of the full grammar. It estimates the KL divergence across training epochs and plots it.
  If `create_table` is passed, it stores the final epoch value for each seed in `kl_table.csv`.  

- **`cka_analysis.py`**
  This script trains the transformer models on a PCFG dataset (dataset already has to exist) under 2 different regimens (training from scratch and pretraining on the subgrammar) across seeds; both regimens run for the same total number of epochs. It then computes and summarizes multiple similarity metrics:
  - L2 distances (overall and per-layer)
  - Cosine similarity of flattened weights
  - CKA (Centered Kernel Alignment) of activations (split by attention vs MLP layers)
  - RSA (Representational Similarity Analysis) correlations (split by attention vs MLP layers)
  The script aggregates results into a CSV table of final summary values for each training comparison. Optionally (with `--generate_plots`), it also produces heatmaps and histograms for per-layer similarities and distributions.

- **`activation_space.py`**
    To analyze the cosine similarity of models, supports both within-set and cross-set comparisons, aggregates results across seeds and can output per-layer heatmaps. 
    To analyze within-set comparison, the `.txt` file should contain a single block of sequences, one sequence per line. For cross-set comparison, the `.txt` file should contain **two blocks** of sequences, separated by an **empty line** to ensure correct parsing.

- **`depth_recursion_exp.py`**
    This script trains and evaluates the transformer **TwoLayer_LARGE** model on the **NestedParentheses** grammar. It generates synthetic sequences, compares model-predicted logits with handcrafted ground-truth distributions, and visualizes prediction errors across depths and random seeds. The pipeline includes tokenizer generation, training the model on multiple seeds, and systematic analysis of different input cases and prefixes.

- **`kl.py`**
  This runs the experiment show how model learning dynamics depend on the production probability *p*. Specifically, it investigates the double-sided recursion and tail recursion. 
  This script creates the tokenizer and dataset, trains the modle on each dataset and finally evaluates the model across checkpoints to finally plot the KL divergence curves for different *p* values.

- **`generate_arithmetic_expr.ipynb`**  
  A Jupyter Notebook that can be used to generate random nested arithmetic expressions,  
  up to a maximum nesting depth and number of terms.


---

## Usage

1. **Define or extend grammars** in `def_pcfgs.py` if needed.
2. **Generate data** with `generate_pcfg.py` for the grammar of interest.
3. **Train models** using `train.py` (direct or checkpoint continuation).
4. **Run analyses** using the analysis scripts (analysis_hierarchy.py, cka_analysis.py, etc.)


## Notes
- The Transformer architecture is based on GPT-2 but scaled down for PCFG datasets, which are smaller and more compact than natural language corpora.

- Training runs and analysis scripts may overwrite results if re-run with the same grammar/start symbol and model type—ensure unique naming if you want to preserve outputs.
