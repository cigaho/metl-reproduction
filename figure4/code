import numpy as np
import matplotlib.pyplot as plt


# 1. Axes definitions


# Simulated dataset size (x axis, in thousands)
sim_sizes = np.array(
    [1, 2, 4, 8, 16, 32, 64,
     128, 256, 512, 1024, 2048, 4096, 8192],
    dtype=float,
)

# Experimental dataset size (these will be y-axis *labels*, not coordinates)
exp_sizes = np.array(
    [10, 20, 40, 80, 160, 320, 640,
     1280, 2560, 5120, 10240, 20480],
    dtype=int,
)

# We plot rows at indices 0..11, all equally spaced
row_idx = np.arange(len(exp_sizes))   # 0,1,...,11


# 2. Spearman grid (rows follow exp_sizes order, extracted from paper)

Z = np.array([ 
    [0.21, 0.25, 0.26, 0.31, 0.37, 0.40, 0.41, 0.41, 0.42, 0.44, 0.41, 0.42, 0.42, 0.42],  # 10
    [0.33, 0.41, 0.44, 0.45, 0.50, 0.50, 0.49, 0.53, 0.52, 0.53, 0.55, 0.54, 0.53, 0.55],  # 20
    [0.36, 0.46, 0.44, 0.54, 0.59, 0.60, 0.64, 0.63, 0.62, 0.63, 0.66, 0.65, 0.63, 0.65],  # 40
    [0.51, 0.60, 0.60, 0.66, 0.68, 0.69, 0.73, 0.72, 0.71, 0.73, 0.74, 0.74, 0.74, 0.74],  # 80
    [0.57, 0.68, 0.63, 0.73, 0.75, 0.76, 0.79, 0.79, 0.78, 0.79, 0.81, 0.79, 0.79, 0.81],  # 160
    [0.67, 0.75, 0.74, 0.79, 0.81, 0.83, 0.85, 0.84, 0.83, 0.85, 0.86, 0.85, 0.85, 0.86],  # 320
    [0.78, 0.81, 0.81, 0.86, 0.86, 0.87, 0.89, 0.89, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89],  # 640
    [0.86, 0.88, 0.88, 0.90, 0.91, 0.91, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92],  # 1,280
    [0.91, 0.91, 0.92, 0.93, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.95],  # 2,560
    [0.92, 0.94, 0.94, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96],  # 5,120
    [0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96],  # 10,240
    [0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97],  # 20,480
], dtype=float)

assert Z.shape == (len(exp_sizes), len(sim_sizes))

# Meshgrid uses row indices (categorical y), NOT the numeric exp_sizes
X, Y = np.meshgrid(sim_sizes, row_idx)


# 3. Plot

fig, ax = plt.subplots(figsize=(7, 6))

levels = np.linspace(0.2, 1.0, 17)
cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")

cont_levels = np.arange(0.3, 1.0, 0.1)
cs = ax.contour(X, Y, Z, levels=cont_levels, colors="white", linewidths=0.75)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

# annotate each cell with its Spearman value
for i, yi in enumerate(row_idx):
    for j, xj in enumerate(sim_sizes):
        ax.text(xj, yi, f"{Z[i, j]:.2f}",
                ha="center", va="center",
                fontsize=7, color="black")

# X axis: log2 like the paper
ax.set_xscale("log", base=2)
ax.set_xticks(sim_sizes)
ax.get_xaxis().set_major_formatter(
    plt.FixedFormatter([str(int(v)) for v in sim_sizes])
)

# Y axis: use row indices, but label with experimental sizes
ax.set_yticks(row_idx)
ax.set_yticklabels([f"{v:,}" for v in exp_sizes])  # 1,280 style
ax.set_ylim(-0.5, len(row_idx) - 0.5)  # center rows nicely

ax.set_xlabel("Simulated dataset size (in thousands)")
ax.set_ylabel("Experimental dataset size")

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Spearman's correlation")

ax.set_title("Fig. 4 – GB1: relationship between experimental and simulated data")

fig.tight_layout()
plt.show()
