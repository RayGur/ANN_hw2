"""
訓練器模組

實作多種訓練方法:
1. StandardBPTrainer - 標準反向傳播
2. MomentumTrainer - 動量法
3. ResilientPropTrainer - Resilient Propagation (RProp)
4. LevenbergMarquardtTrainer - Levenberg-Marquardt

設計模式:
- BaseTrainer: 提供共用的訓練邏輯
- 各訓練器繼承 BaseTrainer,實作各自的權重更新策略
"""

import numpy as np
import time
from src.metrics import Metrics


class BaseTrainer:
    """
    訓練器基類

    提供共用的訓練邏輯:
    - 訓練循環
    - 收斂檢查
    - Early stopping
    - 訓練歷史記錄

    子類需實作:
    - update_weights(): 權重更新策略
    """

    def __init__(
        self,
        mlp,
        max_epochs=100000,
        learning_rate=None,
        convergence_threshold=1e-5,  # ← 改
        convergence_patience=100,  # ← 改
        early_stopping_patience=200,  # ← 改
        verbose=True,
        verbose_interval=1000,
    ):
        """
        初始化訓練器

        Parameters:
        -----------
        mlp : MLP
            多層感知機實例
        max_epochs : int
            最大訓練輪數
        learning_rate : float, optional
            學習率 (如果提供,會覆蓋 MLP 的學習率)
        convergence_threshold : float
            收斂閾值 |Loss(t) - Loss(t-1)| < threshold
        convergence_patience : int
            連續多少個 epoch 滿足收斂條件才停止
        early_stopping_patience : int
            驗證損失不再下降的容忍輪數
        verbose : bool
            是否顯示訓練過程
        verbose_interval : int
            顯示間隔
        """
        self.mlp = mlp
        self.max_epochs = max_epochs

        if learning_rate is not None:
            self.mlp.learning_rate = learning_rate

        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        # 訓練歷史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epochs": 0,
            "training_time": 0,
            "convergence_epoch": None,
            "early_stop_epoch": None,
        }

        # 收斂檢查
        self._loss_history = []
        self._convergence_counter = 0

        # Early stopping
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_weights = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        訓練主循環

        Parameters:
        -----------
        X_train : np.ndarray
            訓練特徵
        y_train : np.ndarray
            訓練標籤
        X_val : np.ndarray, optional
            驗證特徵
        y_val : np.ndarray, optional
            驗證標籤

        Returns:
        --------
        dict
            訓練歷史
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Training with {self.__class__.__name__}")
            print(f"{'='*70}")
            print(f"Max epochs: {self.max_epochs}")
            print(f"Learning rate: {self.mlp.learning_rate}")
            print(f"Convergence threshold: {self.convergence_threshold}")
            print(f"Early stopping patience: {self.early_stopping_patience}")
            print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(self.max_epochs):
            # ====== Forward Pass ======
            activations = self.mlp.forward(X_train)

            # ====== Backward Pass ======
            gradients = self.mlp.backward(X_train, y_train, activations)

            # ====== Update Weights ======
            self.update_weights(gradients, epoch)

            # ====== 記錄指標 ======
            if epoch % self.verbose_interval == 0 or epoch == self.max_epochs - 1:
                # 計算訓練指標
                train_loss = self.mlp.compute_loss(X_train, y_train)
                train_pred = self.mlp.predict(X_train)
                train_acc = Metrics.accuracy(y_train, train_pred)

                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)

                # 計算驗證指標
                val_loss, val_acc = None, None
                if X_val is not None and y_val is not None:
                    val_loss = self.mlp.compute_loss(X_val, y_val)
                    val_pred = self.mlp.predict(X_val)
                    val_acc = Metrics.accuracy(y_val, val_pred)

                    self.history["val_loss"].append(val_loss)
                    self.history["val_acc"].append(val_acc)

                # 顯示進度
                if self.verbose:
                    msg = f"Epoch {epoch:6d}/{self.max_epochs}: "
                    msg += f"Loss={train_loss:.6f}, Acc={train_acc:.4f}"

                    if val_loss is not None:
                        msg += f" | Val Loss={val_loss:.6f}, Acc={val_acc:.4f}"

                    print(msg)

            # ====== 檢查收斂 ======
            current_loss = self.mlp.compute_loss(X_train, y_train)
            if self._check_convergence(current_loss, epoch):
                if self.verbose:
                    print(f"\n✓ Converged at epoch {epoch}")
                break

            # ====== Early Stopping ======
            if X_val is not None and y_val is not None:
                val_loss = self.mlp.compute_loss(X_val, y_val)
                if self._check_early_stopping(val_loss, epoch):
                    if self.verbose:
                        print(f"\n✓ Early stopping at epoch {epoch}")
                    # 恢復最佳權重
                    if self._best_weights is not None:
                        self.mlp.set_weights(self._best_weights)
                    break

        # 記錄訓練時間
        end_time = time.time()
        self.history["training_time"] = end_time - start_time
        self.history["epochs"] = epoch + 1

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Training completed in {self.history['training_time']:.2f}s")
            print(f"Total epochs: {self.history['epochs']}")
            print(f"{'='*70}\n")

        return self.history

    def update_weights(self, gradients, epoch):
        """
        更新權重 (需在子類中實作)

        Parameters:
        -----------
        gradients : list
            梯度列表
        epoch : int
            當前 epoch
        """
        raise NotImplementedError("Subclass must implement update_weights()")

    def _check_convergence(self, current_loss, epoch):
        """
        檢查是否收斂

        收斂條件:
        1. 至少訓練 50 個 epoch
        2. |Loss(t) - Loss(t-1)| < threshold
        3. Loss 本身足夠小 (< 0.05)
        4. 連續 patience 個 epoch 滿足條件
        """
        self._loss_history.append(current_loss)

        # 至少訓練 50 個 epoch (避免過早停止)
        if epoch < 50:
            return False

        # 至少需要兩個 loss 才能比較
        if len(self._loss_history) < 2:
            return False

        # 計算損失變化
        loss_change = abs(self._loss_history[-1] - self._loss_history[-2])

        # 收斂條件
        converged = loss_change < self.convergence_threshold and current_loss < 0.05

        if converged:
            self._convergence_counter += 1
        else:
            self._convergence_counter = 0

        if self._convergence_counter >= self.convergence_patience:
            self.history["convergence_epoch"] = epoch
            return True

        return False

    def _check_early_stopping(self, val_loss, epoch):
        """
        檢查是否需要 early stopping

        條件: 驗證損失連續 patience 個 epoch 不再下降

        Parameters:
        -----------
        val_loss : float
            驗證損失
        epoch : int
            當前 epoch

        Returns:
        --------
        bool
            是否需要停止
        """
        if val_loss < self._best_val_loss:
            # 驗證損失下降,更新最佳值
            self._best_val_loss = val_loss
            self._patience_counter = 0
            # 保存最佳權重
            self._best_weights = self.mlp.get_weights()
        else:
            # 驗證損失未下降
            self._patience_counter += 1

        if self._patience_counter >= self.early_stopping_patience:
            self.history["early_stop_epoch"] = epoch
            return True

        return False


class StandardBPTrainer(BaseTrainer):
    """
    標準反向傳播訓練器

    權重更新規則:
    W(t+1) = W(t) + η · ∇W
    """

    def __init__(self, mlp, **kwargs):
        super().__init__(mlp, **kwargs)

    def update_weights(self, gradients, epoch):
        """
        標準梯度下降更新

        Parameters:
        -----------
        gradients : list
            梯度列表
        epoch : int
            當前 epoch (未使用)
        """
        self.mlp.update_weights(gradients)


class MomentumTrainer(BaseTrainer):
    """
    動量法訓練器

    權重更新規則:
    V(t+1) = α · V(t) + η · ∇W
    W(t+1) = W(t) + V(t+1)

    其中 α 是動量係數 (momentum rate)

    優點:
    - 加速收斂 (約 η/(1-α) 倍)
    - 幫助跳出局部極小值
    - 平滑震盪
    """

    def __init__(self, mlp, momentum=0.9, **kwargs):
        """
        Parameters:
        -----------
        momentum : float
            動量係數 α (default: 0.9)
            建議範圍: 0.5 ~ 0.9
        """
        super().__init__(mlp, **kwargs)
        self.momentum = momentum

        # 初始化速度 (velocity)
        self.velocities = [np.zeros_like(W) for W in mlp.weights]

        if self.verbose:
            print(f"[Momentum] α = {self.momentum}")

    def update_weights(self, gradients, epoch):
        """
        動量法更新

        Parameters:
        -----------
        gradients : list
            梯度列表
        epoch : int
            當前 epoch (未使用)
        """
        for l in range(len(self.mlp.weights)):
            # V(t+1) = α · V(t) + η · ∇W
            self.velocities[l] = (
                self.momentum * self.velocities[l]
                + self.mlp.learning_rate * gradients[l]
            )

            # W(t+1) = W(t) + V(t+1)
            self.mlp.weights[l] += self.velocities[l]


class ResilientPropTrainer(BaseTrainer):
    """
    Resilient Propagation (RProp) 訓練器

    核心思想:
    - 只使用梯度的「符號」,不用「大小」
    - 每個權重有自己的學習率
    - 根據梯度符號變化調整學習率

    更新規則:
    if sign(∂E/∂W(t)) = sign(∂E/∂W(t-1)):
        Δ(t) = min(η⁺ · Δ(t-1), Δ_max)  # 增大步長
    else:
        Δ(t) = max(η⁻ · Δ(t-1), Δ_min)  # 減小步長

    ΔW = -sign(∂E/∂W) · Δ(t)

    優點:
    - 避免梯度消失問題
    - 對函數逼近問題特別快
    - 適合輸出變化劇烈的場景
    """

    def __init__(
        self,
        mlp,
        eta_plus=1.2,
        eta_minus=0.5,
        delta_init=0.1,
        delta_max=50.0,
        delta_min=1e-6,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        eta_plus : float
            學習率增長因子 (default: 1.2)
        eta_minus : float
            學習率衰減因子 (default: 0.5)
        delta_init : float
            初始學習率 (default: 0.1)
        delta_max : float
            最大學習率 (default: 50.0)
        delta_min : float
            最小學習率 (default: 1e-6)
        """
        super().__init__(mlp, **kwargs)
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_max = delta_max
        self.delta_min = delta_min

        # 初始化每個權重的學習率
        self.deltas = [np.full_like(W, delta_init) for W in mlp.weights]

        # 保存上一次的梯度符號
        self.prev_gradients = [np.zeros_like(W) for W in mlp.weights]

        if self.verbose:
            print(f"[RProp] η⁺={eta_plus}, η⁻={eta_minus}")
            print(f"        Δ_init={delta_init}, Δ_max={delta_max}, Δ_min={delta_min}")

    def update_weights(self, gradients, epoch):
        """
        RProp 更新

        Parameters:
        -----------
        gradients : list
            梯度列表
        epoch : int
            當前 epoch
        """
        for l in range(len(self.mlp.weights)):
            # 計算梯度符號變化
            sign_change = gradients[l] * self.prev_gradients[l]

            # 更新學習率
            # sign_change > 0: 符號相同,增大步長
            # sign_change < 0: 符號改變,減小步長
            self.deltas[l] = np.where(
                sign_change > 0,
                np.minimum(self.deltas[l] * self.eta_plus, self.delta_max),
                np.maximum(self.deltas[l] * self.eta_minus, self.delta_min),
            )

            # 更新權重
            # ΔW = -sign(∂E/∂W) · Δ
            weight_update = -np.sign(gradients[l]) * self.deltas[l]
            self.mlp.weights[l] += weight_update

            # 當符號改變時,不更新梯度 (跳過)
            # 否則更新梯度
            self.prev_gradients[l] = np.where(
                sign_change < 0, 0, gradients[l]  # 符號改變,重置  # 符號相同,更新
            )


class LevenbergMarquardtTrainer(BaseTrainer):
    """
    Levenberg-Marquardt (LM) 訓練器

    核心思想:
    - 二階優化方法 (使用 Hessian 矩陣近似)
    - Newton 法的近似
    - 結合梯度下降和高斯-牛頓法

    更新規則:
    ΔW = -(H + μI)^(-1) · g

    其中:
    - H: Hessian 矩陣近似 (J^T · J)
    - g: 梯度 (J^T · e)
    - μ: 阻尼係數
    - J: Jacobian 矩陣

    優點:
    - 收斂速度最快 (通常)
    - 適合小-中型網路

    缺點:
    - 計算複雜度高 O(n³)
    - 記憶體需求大
    - 大型網路不適用

    注意: 這是簡化版實作,完整版需要實作 Jacobian 矩陣計算
    """

    def __init__(self, mlp, mu=0.01, mu_increase=10, mu_decrease=0.1, **kwargs):
        """
        Parameters:
        -----------
        mu : float
            初始阻尼係數 (default: 0.01)
        mu_increase : float
            失敗時的 μ 增長因子 (default: 10)
        mu_decrease : float
            成功時的 μ 衰減因子 (default: 0.1)
        """
        super().__init__(mlp, **kwargs)
        self.mu = mu
        self.mu_increase = mu_increase
        self.mu_decrease = mu_decrease

        self.prev_loss = float("inf")

        if self.verbose:
            print(f"[LM] μ_init={mu}, increase={mu_increase}, decrease={mu_decrease}")
            print(f"     Warning: This is a simplified implementation")

    def update_weights(self, gradients, epoch):
        """
        LM 更新 (簡化版)

        完整的 LM 需要計算 Jacobian 矩陣,這裡使用簡化版本:
        使用自適應學習率 + 帶阻尼的梯度下降

        Parameters:
        -----------
        gradients : list
            梯度列表
        epoch : int
            當前 epoch
        """
        # 簡化版: 使用自適應學習率
        # 完整版需要實作 Jacobian 和 Hessian

        # 嘗試更新
        old_weights = [W.copy() for W in self.mlp.weights]

        # 更新權重 (帶阻尼)
        for l in range(len(self.mlp.weights)):
            # 簡化的 LM 更新: W = W + (η / (1 + μ)) · ∇W
            effective_lr = self.mlp.learning_rate / (1 + self.mu)
            self.mlp.weights[l] += effective_lr * gradients[l]

        # 檢查是否改進
        # (這需要重新計算 loss,在實際使用中應該在 train() 中處理)
        # 這裡簡化處理

        # 自適應調整 μ
        # 如果性能改善,減小 μ (更接近高斯-牛頓法)
        # 如果性能變差,增大 μ (更接近梯度下降)
        # 這部分在完整實作中需要更複雜的邏輯


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試各種訓練方法
    """
    print("=" * 70)
    print("訓練器模組測試")
    print("=" * 70)

    # 創建測試資料 (XOR)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    from mlp import MLP

    # 測試各種訓練方法
    trainers = [
        ("Standard BP", StandardBPTrainer, {}),
        ("Momentum (α=0.9)", MomentumTrainer, {"momentum": 0.9}),
        ("Momentum (α=0.7)", MomentumTrainer, {"momentum": 0.7}),
        ("RProp", ResilientPropTrainer, {}),
        ("LM (simplified)", LevenbergMarquardtTrainer, {}),
    ]

    results = []

    for name, TrainerClass, kwargs in trainers:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        # 創建 MLP (使用相同的隨機種子確保公平比較)
        mlp = MLP([2, 8, 2], activation="tanh", learning_rate=0.3, random_seed=42)

        # 創建訓練器
        trainer = TrainerClass(
            mlp,
            max_epochs=2000,
            convergence_threshold=1e-6,
            convergence_patience=50,
            verbose=True,
            verbose_interval=500,
            **kwargs,
        )

        # 訓練
        history = trainer.train(X, y)

        # 測試
        predictions = mlp.predict(X)
        accuracy = Metrics.accuracy(y, predictions)

        results.append(
            {
                "name": name,
                "accuracy": accuracy,
                "epochs": history["epochs"],
                "time": history["training_time"],
                "final_loss": (
                    history["train_loss"][-1] if history["train_loss"] else None
                ),
            }
        )

        print(f"\n✓ Final Accuracy: {accuracy:.2%}")
        print(f"✓ Training Time: {history['training_time']:.3f}s")
        print(f"✓ Total Epochs: {history['epochs']}")

    # 總結比較
    print(f"\n{'='*70}")
    print("訓練方法比較")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Accuracy':>10} {'Epochs':>8} {'Time (s)':>10}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['name']:<20} {r['accuracy']:>9.2%} {r['epochs']:>8} {r['time']:>10.3f}"
        )

    print(f"{'='*70}\n")
