import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import textwrap
# Data
data = {
    "Team": [
        "MEVIS-ProSurvival",
        "MartelLab",
        "AIRA Matrix",
        "Paicon",
        "LEOPARD Baseline",
        "KatherTeam",
        "HITSZLab",
        "QuIIL Lab",
        "IUCompPath"
    ],
    "External C-index": [0.706, 0.705, 0.665, 0.697, 0.677, 0.655, 0.658, 0.646, 0.631],
    "External Data": [1, 1, 1, 1, 0, 1, 0, 0, 0],
    "Pathology Foundation Model": [1, 0, 0, 1, 1, 1, 1, 1, 1],
    "Multiple Instance Learning (MIL)": [1, 0, 1, 1, 1, 0, 1, 1, 1],
    "Attention-Based MIL": [1, 0, 1, 1, 1, 0, 0, 0, 1],
    "Censoring-Aware Survival Loss": [1, 0, 0, 1, 1, 1, 1, 0, 1],
    "Colour Augmentation": [1, 1, 0, 0, 0, 0, 0, 0, 0],
}

df = pd.DataFrame(data).sort_values("External C-index", ascending=False).reset_index(drop=True)

feature_cols = [
    "External Data",
    "Pathology Foundation Model",
    "Multiple Instance Learning (MIL)",
    "Attention-Based MIL",
    "Censoring-Aware Survival Loss",
    "Colour Augmentation"
]

matrix = df[feature_cols].values
cindex = df["External C-index"].values

column_colors = ["#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",  "#CC79A7"]

fig, ax = plt.subplots(figsize=(12, 5))

# Box gap settings
gap = 0.25
box_w = 1.0 - gap
box_h = 1.0 - gap
x_pad = gap / 2.0
y_pad = gap / 2.0

# Vertical guide lines (behind boxes)
for j in range(len(feature_cols)):
    ax.vlines(j + 0.5, 0, len(df), colors="black", linewidth=1.0, zorder=0)

# Draw only colored "Yes" boxes
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i, j] == 1:
            ax.add_patch(
                plt.Rectangle(
                    (j + x_pad, i + y_pad),
                    box_w, box_h,
                    facecolor=column_colors[j],
                    edgecolor="none",
                    zorder=2
                )
            )

# Limits and orientation
ax.set_ylim(0, len(df))
ax.set_xlim(-0.6, len(feature_cols))
ax.invert_yaxis()

#print(df["Team"])

# Y-axis labels (teams)
ax.set_yticks(np.arange(len(df)) + 0.5)
ax.set_yticklabels(df["Team"])

# Remove x-axis ticks and labels
ax.set_xticks([])
ax.set_xlabel("")

# Add C-index on the left
ax.text(-0.10, -0.45, "C-index", va="center", ha="right", fontsize=10)

# Add C-index on the left
ax.text(-0.7, -0.45, "Teams", va="center", ha="right", fontsize=10)

for i, val in enumerate(cindex):
    ax.text(-0.15, i + 0.5, f"{val:.3f}", va="center", ha="right", fontsize=10)
    
#for i, val in enumerate(feature_cols):
#    ax.text(i + 0.15, 0.5, val, va="center", ha="right", fontsize=10)
    
# Remove all spines (axis lines)
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)


# Column titles on top (multiline, centered over each column)
for j, label in enumerate(feature_cols):
    wrapped = "\n".join(textwrap.wrap(label, width=18))  # adjust width to fit
    ax.text(
        j + 0.5, -0.6, wrapped,
        ha="center", va="top",
        fontsize=10,
        color="black",
        linespacing=1.1
    )
#ax.set_title("Reported Methodological Components Used")

# Legend
legend_elements = [
    Patch(facecolor=column_colors[i], edgecolor="none", label=feature_cols[i])
    for i in range(len(feature_cols))
]
#ax.legend(handles=legend_elements, bbox_to_anchor=(1.0,0), loc="lower left")
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig("/data/pathology/projects/leopard/rebuttal/methodologies.png")
plt.show()

# multiresolution(1,0,), color augmentation(1,1,0,0,) aira- loss is not a survival
