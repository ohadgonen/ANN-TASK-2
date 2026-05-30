# -*- coding: utf-8 -*-

# 1. Imports and Device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from pathlib import Path

results_dir = Path("results/q5")
results_dir.mkdir(parents=True, exist_ok=True)

all_grad_norms = []

# 2. Device Configuration and Seed Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 3. Load CIFAR-10 Dataset
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# Two train dataset objects for k-fold: augmented (for training splits)
# and clean (for validation splits, so val metrics aren't inflated by augmentation)
train_dataset_aug = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

train_dataset_eval = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=test_transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

batch_size = 128

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# 4. Fixed frequency-separation filters
def make_freq_filters():
    # Gaussian 3x3 (low-pass)
    gauss = torch.tensor([[1., 2., 1.],
                          [2., 4., 2.],
                          [1., 2., 1.]]) / 16.0
    # Laplacian 3x3 (high-pass)
    lap = torch.tensor([[ 0., -1.,  0.],
                        [-1.,  4., -1.],
                        [ 0., -1.,  0.]])

    gauss = gauss.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    lap   = lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return gauss, lap


# 5. CNN Branch
# IMPROVEMENTS:
#   - ReLU activation (replaces sigmoid: avoids vanishing gradients)
#   - BatchNorm after each conv (normalization: faster, more stable training)
#   - Deeper: 5 conv layers instead of 4 (architectural depth)
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,  64,  3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,  128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        # Extra conv layer for added depth (no pool, keeps 2x2 map)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)               # 32 -> 16

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)               # 16 -> 8

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)               # 8 -> 4

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)               # 4 -> 2

        x = F.relu(self.bn5(self.conv5(x)))  # extra depth, no pool

        x = F.adaptive_avg_pool2d(x, 1)     # global avg pool -> 256 x 1 x 1
        x = x.view(x.size(0), -1)           # 256
        return x


# 6. Hybrid Network
class HybridNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.high_freq_cnn = CNNBranch()
        self.low_freq_cnn  = CNNBranch()

        gauss, lap = make_freq_filters()
        self.register_buffer("gauss_kernel", gauss)
        self.register_buffer("lap_kernel", lap)

        # MLP head: 256 + 256 = 512 (global avg pool reduces each branch to 256-dim)
        # Dropout reduced to 0.3 from 0.5 to allow more information to pass through
        self.fc1     = nn.Linear(512, 256)
        self.drop1   = nn.Dropout(0.3)
        self.fc2     = nn.Linear(256, 128)
        self.drop2   = nn.Dropout(0.3)
        self.fc3     = nn.Linear(128, num_classes)

    def split_frequencies(self, x):
        low  = F.conv2d(x, self.gauss_kernel, padding=1, groups=3)
        high = F.conv2d(x, self.lap_kernel,   padding=1, groups=3)
        return low, high

    def forward(self, x):
        low, high = self.split_frequencies(x)

        low_feat  = self.low_freq_cnn(low)
        high_feat = self.high_freq_cnn(high)

        combined = torch.cat([low_feat, high_feat], dim=1)

        x = F.relu(self.fc1(combined))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)                       # raw logits
        return x


criterion  = nn.CrossEntropyLoss()
num_epochs = 15


# 7. Training Function
def train(model, loader, optimizer):
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

        # Save gradient norms
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


# 8. Testing Function
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


# 9. K-Fold Cross Validation
k_folds = 5

n_samples = len(train_dataset_aug)
indices = np.arange(n_samples)
rng = np.random.default_rng(seed)
rng.shuffle(indices)

fold_size = n_samples // k_folds
folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]

fold_val_accs = []
cv_metrics = []

for fold in range(k_folds):

    print(f"\n{'='*40}")
    print(f"Fold {fold + 1}/{k_folds}")
    print(f"{'='*40}")

    val_idx   = folds[fold]
    train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold])

    fold_train_loader = DataLoader(
        Subset(train_dataset_aug, train_idx),
        batch_size=batch_size,
        shuffle=True
    )
    fold_val_loader = DataLoader(
        Subset(train_dataset_eval, val_idx),
        batch_size=batch_size,
        shuffle=False
    )

    fold_model     = HybridNet().to(device)
    fold_optimizer = optim.AdamW(fold_model.parameters(), lr=0.001, weight_decay=5e-4)
    fold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fold_optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        train_loss, train_acc, grad_norms = train(fold_model, fold_train_loader, fold_optimizer)
        val_loss, val_acc = test(fold_model, fold_val_loader)
        fold_scheduler.step()

        cv_metrics.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    fold_val_accs.append(val_acc)

print(f"\nK-Fold CV Results ({k_folds} folds):")
for i, acc in enumerate(fold_val_accs, 1):
    print(f"  Fold {i}: {acc:.2f}%")
print(f"  Mean Val Accuracy: {np.mean(fold_val_accs):.2f}% ± {np.std(fold_val_accs):.2f}%")

# Save k-fold metrics
cv_df = pd.DataFrame(cv_metrics)
cv_df.to_csv(results_dir / "cv_metrics.csv", index=False)

cv_summary_df = pd.DataFrame({
    "fold": list(range(1, k_folds + 1)),
    "final_val_accuracy": fold_val_accs
})
cv_summary_df.to_csv(results_dir / "cv_summary.csv", index=False)


# 10. Final Model: Train on Full Training Set
print(f"\n{'='*40}")
print("Final training on full training set")
print(f"{'='*40}")

full_train_loader = DataLoader(
    train_dataset_aug,
    batch_size=batch_size,
    shuffle=True
)

model     = HybridNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

train_losses = []
test_losses  = []
train_accs   = []
test_accs    = []

for epoch in range(num_epochs):
    train_loss, train_acc, grad_norms = train(model, full_train_loader, optimizer)
    test_loss,  test_acc              = test(model, test_loader)
    scheduler.step()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    epoch_grad_means = {"epoch": epoch + 1}
    for name, values in grad_norms.items():
        epoch_grad_means[name] = np.mean(values)
    all_grad_norms.append(epoch_grad_means)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    print(f"Test  Loss: {test_loss:.4f} | Test  Accuracy: {test_acc:.2f}%")
    print("-" * 40)


# 11. Save Metrics
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

# Loss Curves (final model)
plt.figure(figsize=(8, 5))

plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train loss")
plt.plot(metrics_df["epoch"], metrics_df["test_loss"],  label="Test loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Across Epochs (Final Model)")
plt.legend()
plt.tight_layout()

plt.savefig(results_dir / "loss_curve.png", dpi=300)
plt.show()

# Accuracy Curves (final model)
plt.figure(figsize=(8, 5))

plt.plot(metrics_df["epoch"], metrics_df["train_accuracy"], label="Train accuracy")
plt.plot(metrics_df["epoch"], metrics_df["test_accuracy"],  label="Test accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Across Epochs (Final Model)")
plt.legend()
plt.tight_layout()

plt.savefig(results_dir / "accuracy_curve.png", dpi=300)
plt.show()

# K-Fold CV Validation Accuracy per Fold
plt.figure(figsize=(10, 5))

for fold in range(1, k_folds + 1):
    fold_data = cv_df[cv_df["fold"] == fold]
    plt.plot(fold_data["epoch"], fold_data["val_accuracy"], alpha=0.6, label=f"Fold {fold}")

avg_by_epoch = cv_df.groupby("epoch")["val_accuracy"].mean()
plt.plot(avg_by_epoch.index, avg_by_epoch.values, "k--", linewidth=2, label="Mean")

plt.xlabel("Epoch")
plt.ylabel("Val Accuracy (%)")
plt.title(f"{k_folds}-Fold CV Validation Accuracy")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(results_dir / "cv_val_accuracy.png", dpi=300)
plt.show()

# Gradient Magnitudes
plt.figure(figsize=(10, 6))

for column in grad_df.columns:
    if column != "epoch":
        plt.plot(grad_df["epoch"], grad_df[column], label=column)

plt.xlabel("Epoch")
plt.ylabel("Mean gradient norm")
plt.title("Gradient Magnitudes Across Epochs")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(results_dir / "gradient_magnitudes.png", dpi=300)
plt.show()


# 12. Confusion Matrix
def get_all_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

true_labels, pred_labels = get_all_predictions(model, test_loader)

cm = np.zeros((10, 10), dtype=int)
for t, p in zip(true_labels, pred_labels):
    cm[t][p] += 1

cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv(results_dir / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Test Set)")

thresh = cm.max() / 2
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black', fontsize=7)

plt.tight_layout()
plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
plt.show()
