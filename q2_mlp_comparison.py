import random
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ----Results Folder----
results_dir = Path("results/q2")
results_dir.mkdir(parents=True, exist_ok=True)


# ----Device and Seed----
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


# ----MLP Model----
class InitialMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):

        # Flatten image from 32x32x3 into one long vector
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ----Initialize Model----
model = InitialMLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ----Count Parameters----
num_params = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

print(f"Number of trainable parameters: {num_params}")


# ----Training Function----
def train(model, loader):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy


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

epoch_times = []

for epoch in range(num_epochs):

    start_time = time.time()

    train_loss, train_acc = train(
        model,
        train_loader
    )

    test_loss, test_acc = test(
        model,
        test_loader
    )

    end_time = time.time()
    epoch_time = end_time - start_time

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    epoch_times.append(epoch_time)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Epoch Time: {epoch_time:.2f} seconds")
    print("-" * 40)


# ----Save Metrics----
metrics_df = pd.DataFrame({
    "epoch": range(1, num_epochs + 1),
    "train_loss": train_losses,
    "test_loss": test_losses,
    "train_accuracy": train_accs,
    "test_accuracy": test_accs,
    "epoch_time_seconds": epoch_times
})

metrics_df.to_csv(results_dir / "mlp_metrics.csv", index=False)


# ----Save Model Summary----
summary_df = pd.DataFrame({
    "model": ["MLP"],
    "trainable_parameters": [num_params],
    "best_test_accuracy": [max(test_accs)],
    "best_epoch": [test_accs.index(max(test_accs)) + 1],
    "final_test_accuracy": [test_accs[-1]],
    "average_epoch_time_seconds": [np.mean(epoch_times)]
})

summary_df.to_csv(results_dir / "mlp_summary.csv", index=False)

'''
# ----Loss Curves----
plt.figure(figsize=(8, 5))

plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train loss")
plt.plot(metrics_df["epoch"], metrics_df["test_loss"], label="Test loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Loss Across Epochs")

plt.legend()
plt.tight_layout()

plt.savefig(results_dir / "mlp_loss_curve.png", dpi=300)
plt.show()


# ----Accuracy Curves----
plt.figure(figsize=(8, 5))

plt.plot(metrics_df["epoch"], metrics_df["train_accuracy"], label="Train accuracy")
plt.plot(metrics_df["epoch"], metrics_df["test_accuracy"], label="Test accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("MLP Accuracy Across Epochs")

plt.legend()
plt.tight_layout()

plt.savefig(results_dir / "mlp_accuracy_curve.png", dpi=300)
plt.show()

'''

# ----Visualize MLP First-Layer Weights----
mlp_weights = model.fc1.weight.detach().cpu()

num_neurons = 16
fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    weight_img = mlp_weights[i].view(3, 32, 32)

    # Convert from C x H x W to H x W x C
    weight_img = weight_img.permute(1, 2, 0)

    # Normalize for display
    weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min())

    ax.imshow(weight_img)
    ax.axis("off")

plt.suptitle("MLP First-Layer Weights")
plt.tight_layout()

plt.savefig(results_dir / "mlp_first_layer_weights.png", dpi=300)
plt.show()