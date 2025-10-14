import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("../results/kl_table.csv")

# Convert kl_divergence to numeric, coercing errors to NaN
df["kl_divergence"] = pd.to_numeric(df["kl_divergence"], errors="coerce")
df["training_type"] = np.where(df["nonTerminal"].str.endswith("_direct"), "from scratch", "with pretraining")

# Extract values
from_scratch = df[df["training_type"] == "from scratch"]["kl_divergence"].dropna().values
with_pretraining = df[df["training_type"] == "with pretraining"]["kl_divergence"].dropna().values

# --- Boxplot ---
fig, ax = plt.subplots(figsize=(4, 3))
ax.boxplot([from_scratch, with_pretraining], tick_labels=["from scratch", "with pretraining"])
#ax.set_title("KL Divergence Distribution (Boxplot)")
ax.set_ylabel("KL Divergence")

plt.tight_layout()
plt.savefig("../results/kl_comparison_boxplot.png", dpi=300)
plt.show()
plt.close()

# --- Histogram ---
min_val = min(np.min(from_scratch), np.min(with_pretraining))
max_val = max(np.max(from_scratch), np.max(with_pretraining))
bins = np.linspace(min_val - 0.1, max_val + 0.1, 15)

fig, ax = plt.subplots(figsize=(4.3, 3))
ax.hist(from_scratch, bins=bins, alpha=0.6, label="from scratch")
ax.hist(with_pretraining, bins=bins, alpha=0.6, label="with pretraining")
#ax.set_title("KL Divergence Distribution", fontsize=18)
ax.set_xlabel("final KL Divergence")
ax.set_ylabel("Frequency")
ax.tick_params(axis='both', which='major')
ax.legend()
plt.tight_layout()
plt.savefig("../results/kl_comparison_hist.png", dpi=300)
plt.show()
plt.close()

# --- Statistics ---
print("\nStatistics:")
print(f"from scratch: mean={np.mean(from_scratch):.4f}, std={np.std(from_scratch):.4f}")
print(f"with pretraining: mean={np.mean(with_pretraining):.4f}, std={np.std(with_pretraining):.4f}")
print(f"Difference (from scratch - with pretraining): {np.mean(from_scratch) - np.mean(with_pretraining):.4f}")
