import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results/q1")
results_dir.mkdir(parents=True, exist_ok=True)

all_grad_norms = []

# ----Imports and Device----
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np


# ----Device Configuration and Seed Initialization----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# ----Load CIFAR-10 Dataset----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ----CNN Starter Code----
class InitialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Added BatchNorm layer after conv1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        # Added BatchNorm after conv1
        # changed activation to ReLU to improve training performance
        
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


# ----Initialize Network----
model = InitialCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ----Training Function----
def train(model, loader):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    gradient_norms = {}

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        for name, param in model.named_parameters():

            if param.grad is not None:

                grad_norm = param.grad.norm().item()

                if name not in gradient_norms:
                    gradient_norms[name] = []

                gradient_norms[name].append(grad_norm)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy, gradient_norms


# ----Testing Function----
def test(model, loader):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy


# ----Main Training Loop----
num_epochs = 15

train_losses = []
test_losses = []

train_accs = []
test_accs = []

for epoch in range(num_epochs):

    train_loss, train_acc, grad_norms = train(
        model,
        train_loader
    )

    # Average gradient size for each layer in this epoch
    epoch_grad_means = {"epoch": epoch + 1}

    for name, values in grad_norms.items():
        epoch_grad_means[name] = np.mean(values)

    all_grad_norms.append(epoch_grad_means)

    test_loss, test_acc = test(
        model,
        test_loader
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("-" * 40)

# Save Metrics
metrics_df = pd.DataFrame({
    "epoch": range(1, num_epochs + 1),
    "train_loss": train_losses,
    "test_loss": test_losses,
    "train_accuracy": train_accs,
    "test_accuracy": test_accs
})

metrics_df.to_csv(results_dir / "metrics.csv", index=False)


# Save Gradient Statistics
grad_df = pd.DataFrame(all_grad_norms)

grad_df.to_csv(
    results_dir / "gradient_magnitudes.csv",
    index=False
)

# Loss Curves
plt.figure(figsize=(8, 5))

plt.plot(
    metrics_df["epoch"],
    metrics_df["train_loss"],
    label="Train loss"
)

plt.plot(
    metrics_df["epoch"],
    metrics_df["test_loss"],
    label="Test loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Loss Across Epochs")

plt.legend()

plt.tight_layout()

plt.savefig(
    results_dir / "loss_curve.png",
    dpi=300
)

plt.show()

# Accuracy Curves
plt.figure(figsize=(8, 5))

plt.plot(
    metrics_df["epoch"],
    metrics_df["train_accuracy"],
    label="Train accuracy"
)

plt.plot(
    metrics_df["epoch"],
    metrics_df["test_accuracy"],
    label="Test accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("Accuracy Across Epochs")

plt.legend()

plt.tight_layout()

plt.savefig(
    results_dir / "accuracy_curve.png",
    dpi=300
)

plt.show()

# Gradient Magnitudes
plt.figure(figsize=(10, 6))

for column in grad_df.columns:

    if column != "epoch":

        plt.plot(
            grad_df["epoch"],
            grad_df[column],
            label=column
        )

plt.xlabel("Epoch")
plt.ylabel("Mean gradient norm")

plt.title("Gradient Magnitudes Across Epochs")

plt.legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    results_dir / "gradient_magnitudes.png",
    dpi=300
)

plt.show()