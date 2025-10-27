"""
訓練器診斷腳本
"""

import numpy as np
from src.mlp import MLP
from src.trainers import StandardBPTrainer, MomentumTrainer, ResilientPropTrainer

# XOR 資料
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

print("=" * 70)
print("診斷訓練器")
print("=" * 70)

# 測試 Momentum
print("\n[診斷 Momentum]")
print("-" * 70)

mlp = MLP([2, 4, 2], activation="tanh", learning_rate=0.5, random_seed=42)
trainer = MomentumTrainer(mlp, momentum=0.9, max_epochs=100, verbose=False)

print("訓練前:")
print(f"  Loss: {mlp.compute_loss(X, y):.6f}")
print(f"  Predictions: {mlp.predict(X)}")

# 訓練幾步並檢查
for epoch in range(10):
    # Forward
    activations = mlp.forward(X)

    # Backward
    gradients = mlp.backward(X, y, activations)

    # 檢查梯度
    if epoch == 0:
        print(f"\nEpoch {epoch}:")
        print(f"  Gradient[0] sample:\n{gradients[0][:2, :3]}")
        print(f"  Velocity[0] sample:\n{trainer.velocities[0][:2, :3]}")

    # Update
    trainer.update_weights(gradients, epoch)

    if epoch == 0:
        print(f"  After update Velocity[0] sample:\n{trainer.velocities[0][:2, :3]}")
        print(f"  Weight[0] sample:\n{mlp.weights[0][:2, :3]}")

    loss = mlp.compute_loss(X, y)
    if epoch % 20 == 0 or epoch < 3:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"\n訓練後:")
print(f"  Loss: {mlp.compute_loss(X, y):.6f}")
print(f"  Predictions: {mlp.predict(X)}")
print(f"  Accuracy: {np.mean(mlp.predict(X) == y):.2%}")

# 對比 Standard BP
print("\n[對比 Standard BP]")
print("-" * 70)

mlp2 = MLP([2, 4, 2], activation="tanh", learning_rate=0.5, random_seed=42)
trainer2 = StandardBPTrainer(mlp2, max_epochs=100, verbose=False)

for epoch in range(10):
    activations = mlp2.forward(X)
    gradients = mlp2.backward(X, y, activations)

    if epoch == 0:
        print(f"\nEpoch {epoch}:")
        print(f"  Gradient[0] sample:\n{gradients[0][:2, :3]}")

    trainer2.update_weights(gradients, epoch)

    if epoch == 0:
        print(f"  Weight[0] sample:\n{mlp2.weights[0][:2, :3]}")

    loss = mlp2.compute_loss(X, y)
    if epoch % 20 == 0 or epoch < 3:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"\n訓練後:")
print(f"  Loss: {mlp2.compute_loss(X, y):.6f}")
print(f"  Predictions: {mlp2.predict(X)}")
print(f"  Accuracy: {np.mean(mlp2.predict(X) == y):.2%}")
