import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


# ----Paths----
improved_loss_path = Path(
    "results/q1/after change (sigma to ReLU)/loss_curve.png"
)

original_loss_path = Path(
    "results/q1/before change/loss_curve.png"
)


# ----Load Images----
improved_loss = mpimg.imread(improved_loss_path)
original_loss = mpimg.imread(original_loss_path)


# ----Create Figure----
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# ----Improved CNN----
axes[0].imshow(improved_loss)
axes[0].axis("off")

axes[0].set_title(
    "A. Improved CNN (ReLU + BatchNorm)",
    fontsize=14,
    fontweight="bold"
)


# ----Original CNN----
axes[1].imshow(original_loss)
axes[1].axis("off")

axes[1].set_title(
    "B. Original CNN (Sigmoid)",
    fontsize=14,
    fontweight="bold"
)


# ----Main Title----
fig.suptitle(
    "Loss Curves Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()


# ----Save----
plt.savefig(
    "results/q1/loss_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


# ----Paths----
improved_grad_path = Path(
    "results/q1/after change (sigma to ReLU)/gradient_magnitudes.png"
)

original_grad_path = Path(
    "results/q1/before change/gradient_magnitudes.png"
)


# ----Load Images----
improved_grad = mpimg.imread(improved_grad_path)
original_grad = mpimg.imread(original_grad_path)


# ----Create Figure----
fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)


# ----Improved CNN----
axes[0].imshow(improved_grad)
axes[0].axis("off")

axes[0].set_title(
    "A. Improved CNN (ReLU + BatchNorm)",
    fontsize=14,
    fontweight="bold"
)


# ----Original CNN----
axes[1].imshow(original_grad)
axes[1].axis("off")

axes[1].set_title(
    "B. Original CNN (Sigmoid)",
    fontsize=14,
    fontweight="bold"
)


# ----Main Title----
fig.suptitle(
    "Gradient Magnitude Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()


# ----Save----
plt.savefig(
    "results/q1/gradient_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()