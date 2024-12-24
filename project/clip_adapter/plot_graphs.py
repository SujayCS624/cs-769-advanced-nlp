import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

save_dir = "graphs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path = "Results.xlsx"  # This is the excel file containing the results.
file = pd.read_excel(path, sheet_name="results_fewshot")

datasets = [
    "Caltech101"
]

shots = [1, 2, 4, 8, 16]
COLORS = {
    "clip_adapter_reimplementation": "C0",
    "clip_adapter_original": "C1"      
}
MS = 3
ALPHA = 1
plt.rcParams.update({"font.size": 12})

our_scores = np.array([0., 0., 0., 0., 0.])
orig_scores = np.array([0., 0., 0., 0., 0.])

for dataset in datasets:
    print(f"Processing {dataset} ...")

    our_scores = file[dataset][0:5]
    orig_scores = file[dataset][5:10]
    
    our_scores = [float(num) for num in our_scores]
    orig_scores = [float(num) for num in orig_scores]

    print(our_scores)
    print(orig_scores)

    fig, ax = plt.subplots()
    ax.set_facecolor("#EBEBEB")

    ax.set_xticks([0] + shots)
    ax.set_xticklabels([0] + shots)
    ax.set_xlabel("Number of labeled training examples per class")
    ax.set_ylabel("Score (%)")
    ax.grid(axis="x", color="white", linewidth=1)
    ax.set_title(dataset)
    ax.set_ylim(min(our_scores) - 1, max(orig_scores) + 1)

    ax.plot(
        shots, our_scores,
        marker="o",
        markersize=MS,
        color=COLORS["clip_adapter_reimplementation"],
        label="CLIP Adapter Reimplementation",
        alpha=ALPHA
    )

    ax.plot(
        shots, orig_scores,
        marker="o",
        markersize=MS,
        color=COLORS["clip_adapter_original"],
        label="CLIP Adapter Original",
        alpha=ALPHA
    )

    ax.legend(loc="lower right")
    fig.savefig(f"{save_dir}/{dataset}.pdf", bbox_inches="tight")
