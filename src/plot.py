import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("../results/kl_table.csv")

# Convert kl_divergence to numeric, coercing errors to NaN
df["kl_divergence"] = pd.to_numeric(df["kl_divergence"], errors="coerce")

# Keep only STMTS and STMTS_direct
subset = df[(df["nonTerminal"].isin(["STMTS", "STMTS_direct"]))]

# Check if we have data
print(f"Number of STMTS_direct entries: {subset[subset['nonTerminal'] == 'STMTS_direct'].shape[0]}")
print(f"Number of STMTS entries: {subset[subset['nonTerminal'] == 'STMTS'].shape[0]}")

# Split into groups
stmts_direct = subset[subset["nonTerminal"] == "STMTS_direct"]["kl_divergence"].values
stmts = subset[subset["nonTerminal"] == "STMTS"]["kl_divergence"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot - use tick_labels instead of labels (deprecated parameter)
axes[0].boxplot([stmts_direct, stmts], tick_labels=["STMTS_direct", "STMTS"])
axes[0].set_title("KL Divergence Distribution (Boxplot)")
axes[0].set_ylabel("KL Divergence")

# Find range for histogram automatically
min_val = min(np.min(stmts_direct), np.min(stmts))
max_val = max(np.max(stmts_direct), np.max(stmts))
bins = np.linspace(min_val - 0.1, max_val + 0.1, 15)  # automatic range

# Histogram
axes[1].hist(stmts_direct, bins=bins, alpha=0.6, label="STMTS_direct")
axes[1].hist(stmts, bins=bins, alpha=0.6, label="STMTS")
axes[1].set_title("KL Divergence Distribution (Histogram)")
axes[1].set_xlabel("KL Divergence")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("../results/kl_comparison_plot.png")
plt.show()

# Print statistics
print("\nStatistics:")
print(f"STMTS_direct: mean={np.mean(stmts_direct):.4f}, std={np.std(stmts_direct):.4f}")
print(f"STMTS: mean={np.mean(stmts):.4f}, std={np.std(stmts):.4f}")
print(f"Difference (direct - pretrained): {np.mean(stmts_direct) - np.mean(stmts):.4f}")