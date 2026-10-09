# Source: run.ipynb, cell 24
# Team methodology overview figure (methodologies.png).

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import FancyBboxPatch
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
    "Multiple Instance Learning (MIL)": [1, 0, 1, 1, 1, 1, 1, 1, 1],
    "Attention-Based MIL": [1, 0, 1, 1, 1, 0, 1, 0, 1],
    "Censoring-Aware Survival Loss": [1, 0, 1, 1, 1, 1, 1, 0, 1],
    "Colour Augmentation": [1, 1, 0, 0, 0, 0, 0, 0, 0],
}

# Rows stay ranked by external C-index (best on top); the value itself is not shown
df = pd.DataFrame(data).sort_values("External C-index", ascending=False).reset_index(drop=True)

# Column -> header text (explicit line breaks keep headers within their column)
feature_labels = {
    "External Data": "External\nData",
    "Pathology Foundation Model": "Pathology\nFoundation\nModel",
    "Multiple Instance Learning (MIL)": "Multiple\nInstance\nLearning (MIL)",
    "Attention-Based MIL": "Attention-\nBased MIL",
    "Censoring-Aware Survival Loss": "Censoring-\nAware\nSurvival Loss",
    "Colour Augmentation": "Colour\nAugmentation",
}
feature_cols = list(feature_labels)

matrix = df[feature_cols].values
n_rows, n_cols = matrix.shape

# Style
yes_color = "#009E73"     # component used
no_color = "#EDEDED"      # component not used
text_color = "#222222"
muted_color = "#666666"
baseline_team = "LEOPARD Baseline"

# Geometry (data units; aspect is equal so rounded corners stay circular)
col_w = 2.9               # column pitch, wider than rows to fit the headers
row_h = 1.0
gap = 0.16                # white gap between neighbouring cells
radius = 0.12
unit_in = 0.42            # inches per data unit

fig, ax = plt.subplots(figsize=(n_cols * col_w * unit_in + 1.8, n_rows * row_h * unit_in + 1.0))

for i in range(n_rows):
    for j in range(n_cols):
        ax.add_patch(
            FancyBboxPatch(
                (j * col_w + gap / 2, i * row_h + gap / 2),
                col_w - gap, row_h - gap,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=yes_color if matrix[i, j] else no_color,
                edgecolor="none",
            )
        )

# Limits and orientation
ax.set_xlim(0, n_cols * col_w)
ax.set_ylim(0, n_rows * row_h)
ax.invert_yaxis()
ax.set_aspect("equal")

# Column headers on top
ax.xaxis.tick_top()
ax.set_xticks((np.arange(n_cols) + 0.5) * col_w)
ax.set_xticklabels([feature_labels[c] for c in feature_cols], fontsize=10.5,
                   color=text_color, linespacing=1.15)

# Team names on the left; the baseline is set apart as a reference row
ax.set_yticks((np.arange(n_rows) + 0.5) * row_h)
ax.set_yticklabels(df["Team"], fontsize=11, color=text_color)
for label in ax.get_yticklabels():
    if label.get_text() == baseline_team:
        label.set_fontstyle("italic")
        label.set_color(muted_color)

ax.tick_params(length=0, pad=6)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig("/Users/khrystynafaryna/Documents/leopard-rebuttal/evaluation/methodologies.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# multiresolution(1,0,), color augmentation(1,1,0,0,) aira- loss is not a survival
