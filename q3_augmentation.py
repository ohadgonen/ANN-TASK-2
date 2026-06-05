import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("/Users/ohadgonen/Desktop/Neuroscience/Year 3/2nd semester/Neural networks/HW/task2_cnn_mlp_homework/results/q3")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Extraction of the exact data from your metrics history
epochs = list(range(1, 16))

# הנתונים המדויקים מתוך קובץ ה-csv 
train_loss = [1.4337, 0.9232, 0.7273, 0.5867, 0.4860, 0.3867, 0.3057, 0.2348, 0.1935, 0.1554, 0.1310, 0.1154, 0.1170, 0.0875, 0.0936]
val_loss   = [1.0490, 0.8491, 0.7832, 0.7341, 0.6756, 0.7216, 0.7812, 0.8309, 0.9289, 1.0010, 1.1542, 1.2112, 1.1149, 1.2709, 1.2542]

train_acc  = [47.298, 67.574, 74.728, 79.694, 83.024, 86.424, 89.270, 91.792, 93.080, 94.590, 95.370, 95.980, 95.890, 96.936, 96.752]
val_acc    = [63.050, 70.540, 72.970, 74.810, 77.630, 77.460, 76.680, 76.900, 75.200, 76.380, 74.320, 75.620, 75.950, 75.220, 76.290]

# ==========================================
# GRAPH 1: LOSS CURVES (Train vs Validation)
# ==========================================
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2.5, marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='#d62728', linewidth=2.5, marker='s')

# סימון קו אנכי באופק 5 המציג את נקודת המפנה לפני תחילת ה-Overfitting
plt.axvline(x=5, color='#2ca02c', linestyle='--', linewidth=2, label='Optimal Point (Epoch 5)')

# עיצוב וכותרות
plt.title('Overfitting Analysis: Training vs. Validation Loss', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper center')
plt.tight_layout()

# שמירה אוטומטית של הגרף כקובץ תמונה
plt.savefig(output_dir / 'overfitting_loss_curves.png', dpi=300)
plt.show()

# ==============================================
# GRAPH 2: ACCURACY CURVES (Train vs Validation)
# ==============================================
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_acc, label='Train Accuracy', color='#1f77b4', linewidth=2.5, marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='#d62728', linewidth=2.5, marker='s')

# סימון קו אנכי באופק 5 המציג את נקודת המפנה לפני תחילת ה-Overfitting
plt.axvline(x=5, color='#2ca02c', linestyle='--', linewidth=2, label='Optimal Point (Epoch 5)')

# עיצוב וכותרות
plt.title('Overfitting Analysis: Training vs. Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='lower right')
plt.tight_layout()

# שמירה אוטומטית של הגרף כקובץ תמונה
plt.savefig(output_dir / 'overfitting_accuracy_curves.png', dpi=300)
plt.show()