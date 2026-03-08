"""
Comparison of DLA cluster statistics for SOR vs Monte Carlo — plotting.

Loads precomputed simulation data from data/dla_sor_vs_mc.pkl and produces
a figure with three subplots, one per cluster statistic (highest point,
broadness, fractal dimension). Each subplot shows SOR results as a function
of eta and MC results as a function of sticking probability, with error bars
showing the standard deviation across the batch.

Input:  data/dla_sor_vs_mc.pkl
Output: images/figures/dla_sor_vs_mc.png
"""

import shutil
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path

import scienceplots  # noqa: F401

styles = (
    ["science"]
    if (shutil.which("latex") and shutil.which("dvipng"))
    else ["science", "no-latex"]
)
plt.style.use(styles)

# Select only the SOR data from index 0 to index SOR_CUTOFF
SOR_CUTOFF = 10

# Load data
data_dir = Path(__file__).parent.parent / "data"
filename = "dla_sor_vs_mc.pkl"
results = load(data_dir / filename)

sor_batches = results["sor_batches"]
mc_batches = results["mc_batches"]
etas = results["etas"][:SOR_CUTOFF]
sticking_probabilities = results["sticking_probabilities"]
N = results["N"]
n_steps = results["n_steps"]
batch_size = results["batch_size"]

# Create the plot
COLOUR_SOR = "tab:green"
COLOUR_MC = "tab:blue"

STAT_NAMES = ["Highest point", "Broadness", "Fractal dimension"]
N_STATS = len(STAT_NAMES)

fig, axes = plt.subplots(1, N_STATS, figsize=(12, 4))

for stat_idx, (ax_mc, stat_name) in enumerate(zip(axes, STAT_NAMES)):
    # Plot MC results vs sticking probability
    mc_means = [mc_batches[p][:, stat_idx].mean() for p in sticking_probabilities]
    mc_stds = [mc_batches[p][:, stat_idx].std() for p in sticking_probabilities]
    ax_mc.errorbar(
        sticking_probabilities,
        mc_means,
        yerr=mc_stds,
        marker="s",
        label="MC",
        color=COLOUR_MC,
        linestyle="--",
    )

    # Plot SOR results vs eta
    ax_sor = ax_mc.twiny()
    sor_means = [sor_batches[eta][:, stat_idx].mean() for eta in etas][:SOR_CUTOFF]
    sor_stds = [sor_batches[eta][:, stat_idx].std() for eta in etas][:SOR_CUTOFF]
    ax_sor.errorbar(
        etas, sor_means, yerr=sor_stds, marker="o", label="SOR", color=COLOUR_SOR
    )

    ax_sor.set_xlabel("$\\eta$ (SOR)", color=COLOUR_SOR)
    ax_sor.tick_params(axis="x", colors=COLOUR_SOR)
    ax_mc.set_xlabel("$p_s$ (MC)", color=COLOUR_MC)
    ax_mc.tick_params(axis="x", colors=COLOUR_MC)
    ax_mc.set_title(stat_name)

# Combined legend from both axes
lines1, labels1 = ax_sor.get_legend_handles_labels()
lines2, labels2 = ax_mc.get_legend_handles_labels()
axes[0].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.suptitle(
    f"DLA cluster statistics: SOR vs MC (${N}\\times{N}$ grid, cluster size = {n_steps}, sample size = {batch_size})",
    fontsize=12,
)
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "images" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "a2_2_dla_sor_vs_mc.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved → {fig_path}")

plt.show()
