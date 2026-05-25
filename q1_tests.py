import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    classification_report
)


# ----Paths----
results_dir = Path("results/q1_tests")
results_dir.mkdir(parents=True, exist_ok=True)

model_path = Path("results/q1/cnn_model.pth")


# ----Device----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----Load CIFAR-10 Test Set----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

classes = test_dataset.classes


# ----CNN Architecture----
# Must match the saved model exactly
class InitialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ----Load Trained Model----
model = InitialCNN().to(device)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()


# ----Run Predictions----
all_labels = []
all_preds = []
all_probs = []
all_images = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = F.softmax(outputs, dim=1)

        _, preds = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        all_images.extend(images.cpu())


all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)


# ----F1 and F-beta Metrics----
f1_macro = f1_score(
    all_labels,
    all_preds,
    average="macro"
)

f1_weighted = f1_score(
    all_labels,
    all_preds,
    average="weighted"
)

fbeta_macro = fbeta_score(
    all_labels,
    all_preds,
    beta=2,
    average="macro"
)

fbeta_weighted = fbeta_score(
    all_labels,
    all_preds,
    beta=2,
    average="weighted"
)

summary_df = pd.DataFrame({
    "metric": [
        "f1_macro",
        "f1_weighted",
        "fbeta_macro_beta2",
        "fbeta_weighted_beta2"
    ],
    "value": [
        f1_macro,
        f1_weighted,
        fbeta_macro,
        fbeta_weighted
    ]
})

summary_df.to_csv(
    results_dir / "classification_metrics.csv",
    index=False
)

print("\nClassification metrics:")
print(summary_df)


# ----Detailed Classification Report----
report = classification_report(
    all_labels,
    all_preds,
    target_names=classes,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    results_dir / "classification_report.csv"
)

# ----Per-Class Precision / Recall / F1 Plot----

class_report_df = report_df.loc[classes]

plt.figure(figsize=(10, 6))

x = np.arange(len(classes))

width = 0.25

plt.bar(

    x - width,

    class_report_df["precision"],

    width,

    label="Precision"

)

plt.bar(

    x,

    class_report_df["recall"],

    width,

    label="Recall"

)

plt.bar(

    x + width,

    class_report_df["f1-score"],

    width,

    label="F1-score"

)

plt.xticks(

    x,

    classes,

    rotation=45,

    ha="right"

)

plt.ylabel("Score")

plt.title("Per-Class Precision, Recall, and F1-score")

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(

    results_dir / "per_class_precision_recall_f1.png",

    dpi=300

)

plt.show()

# ----Confusion Matrix----
cm = confusion_matrix(
    all_labels,
    all_preds
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.to_csv(
    results_dir / "confusion_matrix.csv"
)

plt.figure(figsize=(10, 9))

plt.imshow(cm)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(classes)),
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(classes)),
    classes
)

# Add numbers inside each square
for i in range(len(classes)):
    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=8,
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.colorbar(label="Number of images")

plt.tight_layout()

plt.savefig(
    results_dir / "confusion_matrix.png",
    dpi=300
)

plt.show()


# ----Top Confused Class Pairs----
confusions = []

for true_idx in range(len(classes)):

    for pred_idx in range(len(classes)):

        if true_idx != pred_idx:

            confusions.append({
                "true_label": classes[true_idx],
                "predicted_label": classes[pred_idx],
                "count": cm[true_idx, pred_idx]
            })

confusions_df = pd.DataFrame(confusions)

confusions_df = confusions_df.sort_values(
    by="count",
    ascending=False
)

confusions_df.to_csv(
    results_dir / "top_confused_class_pairs.csv",
    index=False
)

print("\nTop confused class pairs:")
print(confusions_df.head(10))


# ----Top Individual Wrong Predictions----
wrong_indices = np.where(
    all_labels != all_preds
)[0]

wrong_examples = []

for idx in wrong_indices:

    true_label = all_labels[idx]
    pred_label = all_preds[idx]

    confidence = all_probs[idx, pred_label]

    wrong_examples.append({
        "index": idx,
        "true_label": classes[true_label],
        "predicted_label": classes[pred_label],
        "confidence_in_wrong_prediction": confidence
    })

wrong_examples_df = pd.DataFrame(wrong_examples)

wrong_examples_df = wrong_examples_df.sort_values(
    by="confidence_in_wrong_prediction",
    ascending=False
)

wrong_examples_df.to_csv(
    results_dir / "top_wrong_predictions.csv",
    index=False
)

print("\nTop wrong individual predictions:")
print(wrong_examples_df.head(10))


# ----Plot Top Wrong Predictions----
top_n = 10

top_wrong = wrong_examples_df.head(top_n)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)

for ax, (_, row) in zip(
    axes.flat,
    top_wrong.iterrows()
):

    img = all_images[int(row["index"])]

    # Undo normalization for plotting
    img = img / 2 + 0.5

    img = img.permute(1, 2, 0)

    # Smoother interpolation for visualization
    ax.imshow(img)

    ax.set_title(
        f'True: {row["true_label"]}\nPred: {row["predicted_label"]}',
        fontsize=9
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    results_dir / "top_wrong_predictions.png",
    dpi=300
)

plt.show()