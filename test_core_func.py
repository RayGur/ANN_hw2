"""
測試核心計算函數
驗證 loss 和 accuracy 的計算是否正確
"""

import numpy as np
import sys

sys.path.append("src")

from mlp import MLP
from metrics import Metrics

print("=" * 70)
print("測試核心計算函數")
print("=" * 70)

# 創建簡單的測試數據
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 創建 MLP
mlp = MLP([2, 4, 2], activation="tanh", learning_rate=0.3, random_seed=42)

print("\n[1] 測試 compute_loss")
print("-" * 70)
for i in range(5):
    loss = mlp.compute_loss(X, y)
    print(f"  Iteration {i+1}: Loss = {loss:.6f}")

    # 檢查
    if loss < 0:
        print(f"  ❌ ERROR: Loss is negative! ({loss})")
    else:
        print(f"  ✓ Loss is non-negative")

    # 訓練一步
    activations = mlp.forward(X)
    gradients = mlp.backward(X, y, activations)
    mlp.update_weights(gradients)

print("\n[2] 測試 Metrics.accuracy")
print("-" * 70)
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])

acc = Metrics.accuracy(y_true, y_pred)
print(f"  y_true: {y_true}")
print(f"  y_pred: {y_pred}")
print(f"  Accuracy: {acc:.4f}")

# 檢查
if acc < 0 or acc > 1:
    print(f"  ❌ ERROR: Accuracy out of range! ({acc})")
else:
    print(f"  ✓ Accuracy is in [0, 1]")

print("\n[3] 測試 MLP.predict")
print("-" * 70)
pred = mlp.predict(X)
acc = Metrics.accuracy(y, pred)
print(f"  Predictions: {pred}")
print(f"  True labels: {y}")
print(f"  Accuracy: {acc:.4f}")

# 檢查
if acc < 0 or acc > 1:
    print(f"  ❌ ERROR: Accuracy out of range! ({acc})")
else:
    print(f"  ✓ Accuracy is in [0, 1]")

print("\n[4] 測試大量隨機數據")
print("-" * 70)
np.random.seed(42)
for trial in range(10):
    # 隨機生成數據
    n_samples = np.random.randint(10, 100)
    n_features = np.random.randint(2, 10)
    n_classes = np.random.randint(2, 5)

    X_random = np.random.randn(n_samples, n_features)
    y_random = np.random.randint(0, n_classes, n_samples)

    # 創建 MLP
    mlp_test = MLP([n_features, 8, n_classes], activation="sigmoid", random_seed=trial)

    # 計算 loss
    loss = mlp_test.compute_loss(X_random, y_random)

    # 計算 accuracy
    pred = mlp_test.predict(X_random)
    acc = Metrics.accuracy(y_random, pred)

    print(f"  Trial {trial+1}: Loss={loss:.6f}, Acc={acc:.4f}", end="")

    # 驗證
    if loss < 0:
        print(" ❌ Negative loss!")
    elif acc < 0 or acc > 1:
        print(" ❌ Invalid accuracy!")
    else:
        print(" ✓")

print("\n" + "=" * 70)
print("測試完成!")
print("=" * 70)
