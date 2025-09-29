# pcfg-transformer-learning

This repository contains tools for analyzing transformer models trained on probabilistic context-free grammars (PCFGs).  
It was used to run the experiments in the paper:

> *Unraveling Syntax: How Language Models Learn Context-Free Grammars*

---

## Repository Structure

### Files

- **`def_pcfgs.py`**  
  Contains all PCFG grammars defined for the project.  
  If you wish to add more grammars, extend the `GRAMMARS` dictionary in this file.

- **`generate_pcfg.py`**  
  Generates the tokenizer, training set, and test set for the predefined grammars.  
  This is the main script for dataset preparation.

- **`math.ipynb`**  
  A Jupyter Notebook that can be used to generate random nested arithmetic expressions,  
  up to a maximum nesting depth and number of terms.

- **`activation_space.py`**
    To analyze the cosine similarity of models, supports both within-set and cross-set comparisons, aggregates results across seeds and can output per-layer heatmaps. 
    To analyze within-set comparison, the `.txt` file should contain a single block of sequences, one sequence per line. For cross-set comparison, the `.txt` file should contain **two blocks** of sequences, separated by an **empty line** to ensure correct parsing.


---

## Usage

1. **Define or extend grammars** in `def_pcfgs.py` if needed.
2. **Generate data** with `generate_pcfg.py` for the grammar of interest.
3. **Optionally experiment** with arithmetic datasets using `math.ipynb`.
