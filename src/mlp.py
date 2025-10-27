"""
多層感知機 (MLP) 實作

基於 Backpropagation 演算法訓練的多層感知機
支援:
- 靈活的網路架構 (任意層數和神經元數)
- 多種激活函數 (Sigmoid, Tanh, ReLU)
- Batch/Mini-batch/SGD 訓練
- 多分類和二分類任務

實作基於 pseudocode:
- Training Phase: Forward + Backward propagation
- Operation Phase: Forward propagation only
"""

import numpy as np
from src.activations import get_activation


class MLP:
    """
    多層感知機 (Multilayer Perceptron)

    架構:
    Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer

    Parameters:
    -----------
    layer_sizes : list
        各層神經元數量，例如 [4, 8, 4, 3] 表示:
        - 輸入層: 4 個特徵
        - 隱藏層1: 8 個神經元
        - 隱藏層2: 4 個神經元
        - 輸出層: 3 個類別

    activation : str or list
        激活函數類型
        - str: 所有隱藏層使用相同激活函數
        - list: 為每層指定不同激活函數

    learning_rate : float
        學習率 η (default: 0.2)

    weight_scale : float
        權重初始化的尺度 (default: 1.0)
        使用 Xavier 初始化: weight_scale / sqrt(n_inputs)

    random_seed : int
        隨機種子，確保可重現性

    Examples:
    ---------
    >>> # 創建 MLP: 4 輸入 → 8 隱藏 → 4 隱藏 → 3 輸出
    >>> mlp = MLP([4, 8, 4, 3], activation='tanh', learning_rate=0.2)
    >>>
    >>> # 訓練
    >>> mlp.fit(X_train, y_train, epochs=1000)
    >>>
    >>> # 預測
    >>> predictions = mlp.predict(X_test)
    """

    def __init__(
        self,
        layer_sizes,
        activation="sigmoid",
        learning_rate=0.2,
        weight_scale=1.0,
        random_seed=42,
    ):
        """
        初始化 MLP
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)  # 包含輸入層
        self.learning_rate = learning_rate
        self.weight_scale = weight_scale
        self.random_seed = random_seed

        # 設置隨機種子
        np.random.seed(random_seed)

        # 設置激活函數
        self._setup_activations(activation)

        # 初始化權重和偏置
        self._initialize_weights()

        # 儲存訓練歷史
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _setup_activations(self, activation):
        """
        設置激活函數

        Parameters:
        -----------
        activation : str or list
            激活函數名稱或列表
        """
        if isinstance(activation, str):
            # 所有層使用相同激活函數
            self.activations = [
                get_activation(activation) for _ in range(self.num_layers - 1)
            ]
        else:
            # 為每層指定激活函數
            if len(activation) != self.num_layers - 1:
                raise ValueError(
                    f"Number of activations ({len(activation)}) must match "
                    f"number of layers ({self.num_layers - 1})"
                )
            self.activations = [get_activation(act) for act in activation]

        # 輸出層通常使用 Softmax (多分類) 或 Sigmoid (二分類)
        # 這裡先使用隱藏層的激活函數，後續可根據任務調整

    def _initialize_weights(self):
        """
        初始化權重和偏置

        使用改進的 Xavier 初始化:
        W ~ N(0, scale / sqrt(n_in))

        權重矩陣維度:
        W^(L) : (n_L, n_(L-1) + 1)  # +1 是 bias (第 0 列)

        注意: 這裡包含 bias 在權重矩陣的第一列
        """
        self.weights = []

        for l in range(self.num_layers - 1):
            n_in = self.layer_sizes[l]
            n_out = self.layer_sizes[l + 1]

            # Xavier/He 初始化
            # 第一列是 bias weights, 其餘是 input weights
            scale = self.weight_scale / np.sqrt(n_in)

            # W shape: (n_out, n_in + 1)
            # 第 0 列: bias weights
            # 第 1~n_in 列: input weights
            W = np.random.randn(n_out, n_in + 1) * scale

            self.weights.append(W)

        print(f"[MLP Initialized]")
        print(f"  Architecture: {' → '.join(map(str, self.layer_sizes))}")
        print(f"  Activations: {[act.name for act in self.activations]}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Total parameters: {self._count_parameters()}")

    def _count_parameters(self):
        """計算總參數數量"""
        return sum(w.size for w in self.weights)

    def forward(self, X):
        """
        前向傳播

        按照 pseudocode 實作:
        For each layer L:
          1. 計算 I^(L) = W^(L) · Y^(L-1)
          2. 計算 Y^(L) = g(I^(L))
          3. 加入 bias: Y^(L)_0 = 1

        Parameters:
        -----------
        X : np.ndarray
            輸入特徵 shape: (n_samples, n_features)

        Returns:
        --------
        dict
            包含所有層的激活值 {'I': [...], 'Y': [...]}
        """
        # 初始化儲存
        activations = {"I": [], "Y": []}  # 激活前的值 (weighted sum)  # 激活後的值

        # 輸入層
        # Y^(0) = [1, x_1, x_2, ..., x_n]^T  (加入 bias)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        Y_prev = np.column_stack([np.ones(n_samples), X])  # 加入 bias
        activations["Y"].append(Y_prev)

        # 逐層前向傳播
        for l in range(self.num_layers - 1):
            # Step 1: 計算 I^(L) = W^(L) · Y^(L-1)
            # W shape: (n_out, n_in+1), Y shape: (n_samples, n_in+1)
            # I shape: (n_samples, n_out)
            I = np.dot(Y_prev, self.weights[l].T)
            activations["I"].append(I)

            # Step 2: 計算 Y^(L) = g(I^(L))
            Y = self.activations[l].forward(I)

            # Step 3: 加入 bias (除了輸出層)
            if l < self.num_layers - 2:  # 不是最後一層
                Y = np.column_stack([np.ones(n_samples), Y])

            activations["Y"].append(Y)
            Y_prev = Y

        return activations

    def backward(self, X, y, activations):
        """
        反向傳播

        按照 pseudocode 實作:
        1. 計算輸出層誤差: δ^(L) = (d - Y^(L)) ⊙ g'(I^(L))
        2. 反向傳播誤差: δ^(l) = (W^(l+1)^T · δ^(l+1)) ⊙ g'(I^(l))
        3. 計算梯度: ∇W^(l) = δ^(l) ⊗ Y^(l-1)

        Parameters:
        -----------
        X : np.ndarray
            輸入特徵
        y : np.ndarray
            真實標籤
        activations : dict
            前向傳播的激活值

        Returns:
        --------
        list
            每層的梯度 [∇W^(1), ∇W^(2), ...]
        """
        n_samples = X.shape[0] if X.ndim > 1 else 1

        # 將標籤轉換為 one-hot encoding (如果是多分類)
        y_onehot = self._to_onehot(y, self.layer_sizes[-1])

        # 初始化梯度
        gradients = [np.zeros_like(W) for W in self.weights]

        # ====== 輸出層誤差 ======
        # δ^(L) = (d - Y^(L)) ⊙ g'(I^(L))
        L = self.num_layers - 2  # 最後一個權重層的索引

        Y_L = activations["Y"][-1]  # 輸出層激活值
        I_L = activations["I"][-1]  # 輸出層激活前的值

        # 誤差 = 期望輸出 - 實際輸出
        error = y_onehot - Y_L

        # δ^(L) = error ⊙ g'(I^(L))
        delta = error * self.activations[L].derivative(I_L)

        # ====== 反向傳播 ======
        deltas = [None] * (self.num_layers - 1)
        deltas[L] = delta

        # 從倒數第二層往前傳播
        for l in range(L - 1, -1, -1):
            # δ^(l) = (W^(l+1)^T · δ^(l+1)) ⊙ g'(I^(l))

            # W^(l+1) shape: (n_(l+2), n_(l+1)+1)
            # 移除 bias 的權重 (第 0 列)
            W_next = self.weights[l + 1][:, 1:]  # 去掉 bias

            # 反向傳播誤差
            delta_prev = np.dot(deltas[l + 1], W_next)

            # ⊙ g'(I^(l))
            I_l = activations["I"][l]
            delta_prev = delta_prev * self.activations[l].derivative(I_l)

            deltas[l] = delta_prev

        # ====== 計算梯度 ======
        for l in range(self.num_layers - 1):
            # ∇W^(l) = δ^(l)^T · Y^(l-1)
            # delta shape: (n_samples, n_out)
            # Y shape: (n_samples, n_in+1)
            # gradient shape: (n_out, n_in+1)

            Y_prev = activations["Y"][l]
            gradients[l] = np.dot(deltas[l].T, Y_prev) / n_samples

        return gradients

    def update_weights(self, gradients):
        """
        更新權重

        按照 pseudocode:
        W^(L)(t+1) = W^(L)(t) + η · ∇W^(L)

        Parameters:
        -----------
        gradients : list
            每層的梯度
        """
        for l in range(len(self.weights)):
            # 標準梯度上升 (因為我們最大化正確率，梯度方向是上升)
            # 注意: 我們的 error = (desired - actual)，所以是加法
            self.weights[l] += self.learning_rate * gradients[l]

    def _to_onehot(self, y, num_classes):
        """
        將標籤轉換為 one-hot encoding

        Parameters:
        -----------
        y : np.ndarray
            標籤 (整數)
        num_classes : int
            類別數量

        Returns:
        --------
        np.ndarray
            one-hot 編碼 shape: (n_samples, num_classes)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = len(y)
        y_onehot = np.zeros((n_samples, num_classes))
        y_onehot[np.arange(n_samples), y.flatten().astype(int)] = 1

        return y_onehot

    def compute_loss(self, X, y):
        """
        計算均方誤差 (MSE)

        E = (1/2p) Σ Σ (d_j - Y_j)^2

        Parameters:
        -----------
        X : np.ndarray
            輸入特徵
        y : np.ndarray
            真實標籤

        Returns:
        --------
        float
            損失值
        """
        activations = self.forward(X)
        Y_pred = activations["Y"][-1]

        y_onehot = self._to_onehot(y, self.layer_sizes[-1])

        # MSE
        loss = 0.5 * np.mean((y_onehot - Y_pred) ** 2)

        return loss

    def predict(self, X):
        """
        預測 (Operation Phase)

        只執行前向傳播，不更新權重

        Parameters:
        -----------
        X : np.ndarray
            輸入特徵

        Returns:
        --------
        np.ndarray
            預測的類別標籤
        """
        activations = self.forward(X)
        Y_pred = activations["Y"][-1]

        # 選擇最大機率的類別
        predictions = np.argmax(Y_pred, axis=1)

        return predictions

    def predict_proba(self, X):
        """
        預測機率分布

        Parameters:
        -----------
        X : np.ndarray
            輸入特徵

        Returns:
        --------
        np.ndarray
            各類別的機率 shape: (n_samples, n_classes)
        """
        activations = self.forward(X)
        Y_pred = activations["Y"][-1]

        return Y_pred

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=1000,
        verbose=True,
        verbose_interval=100,
    ):
        """
        訓練 MLP (簡化版，完整版在 Trainer 中)

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
        epochs : int
            訓練輪數
        verbose : bool
            是否顯示訓練過程
        verbose_interval : int
            顯示間隔

        Returns:
        --------
        dict
            訓練歷史
        """
        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X_train)

            # Backward pass
            gradients = self.backward(X_train, y_train, activations)

            # Update weights
            self.update_weights(gradients)

            # 記錄訓練歷史
            if epoch % verbose_interval == 0 or epoch == epochs - 1:
                train_loss = self.compute_loss(X_train, y_train)
                train_pred = self.predict(X_train)
                train_acc = np.mean(train_pred == y_train)

                self.training_history["train_loss"].append(train_loss)
                self.training_history["train_acc"].append(train_acc)

                if verbose:
                    msg = f"Epoch {epoch:5d}/{epochs}: "
                    msg += f"Train Loss={train_loss:.6f}, Acc={train_acc:.4f}"

                    if X_val is not None and y_val is not None:
                        val_loss = self.compute_loss(X_val, y_val)
                        val_pred = self.predict(X_val)
                        val_acc = np.mean(val_pred == y_val)

                        self.training_history["val_loss"].append(val_loss)
                        self.training_history["val_acc"].append(val_acc)

                        msg += f" | Val Loss={val_loss:.6f}, Acc={val_acc:.4f}"

                    print(msg)

        return self.training_history

    def get_weights(self):
        """獲取當前權重"""
        return [W.copy() for W in self.weights]

    def set_weights(self, weights):
        """設置權重"""
        if len(weights) != len(self.weights):
            raise ValueError("Number of weight matrices must match")
        self.weights = [W.copy() for W in weights]


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試 MLP 的基本功能
    """
    print("=" * 70)
    print("MLP 模組測試")
    print("=" * 70)

    # 創建簡單的測試資料 (XOR 問題)
    print("\n[1] 測試簡單 XOR 問題")
    print("-" * 70)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR

    # 創建 MLP: 2 輸入 → 4 隱藏 → 2 輸出
    mlp = MLP([2, 4, 2], activation="tanh", learning_rate=0.5)

    print("\n訓練前預測:")
    pred_before = mlp.predict(X)
    print(f"Predictions: {pred_before}")
    print(f"Accuracy: {np.mean(pred_before == y):.2%}")

    print("\n開始訓練...")
    history = mlp.fit(X, y, epochs=2000, verbose=True, verbose_interval=500)

    print("\n訓練後預測:")
    pred_after = mlp.predict(X)
    proba = mlp.predict_proba(X)
    print(f"Predictions: {pred_after}")
    print(f"Probabilities:\n{proba}")
    print(f"Accuracy: {np.mean(pred_after == y):.2%}")

    print("\n[2] 測試不同激活函數")
    print("-" * 70)

    for act in ["sigmoid", "tanh", "relu"]:
        mlp = MLP([2, 4, 2], activation=act, learning_rate=0.3, random_seed=42)
        mlp.fit(X, y, epochs=1000, verbose=False)
        pred = mlp.predict(X)
        acc = np.mean(pred == y)
        print(f"  {act:8s}: Accuracy = {acc:.2%}")

    print("\n" + "=" * 70)
    print("測試完成!")
    print("=" * 70)
