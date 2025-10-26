"""
激活函數模組

包含各種激活函數的實作,每個激活函數都提供:
1. forward(): 前向傳播計算
2. derivative(): 導數計算 (用於反向傳播)

支援的激活函數:
- Sigmoid
- Tanh (Hyperbolic Tangent)
- ReLU (Rectified Linear Unit)
- Softmax (用於多分類輸出層)
"""

import numpy as np


class ActivationFunction:
    """
    激活函數基類
    
    所有激活函數都應該繼承此類並實作 forward 和 derivative 方法
    """
    
    def forward(self, x):
        """
        前向傳播
        
        Parameters:
        -----------
        x : np.ndarray
            輸入值
            
        Returns:
        --------
        np.ndarray
            激活後的輸出值
        """
        raise NotImplementedError("Subclass must implement forward method")
    
    def derivative(self, x):
        """
        計算導數 (用於反向傳播)
        
        Parameters:
        -----------
        x : np.ndarray
            輸入值 (通常是激活前的值)
            
        Returns:
        --------
        np.ndarray
            導數值
        """
        raise NotImplementedError("Subclass must implement derivative method")
    
    def __call__(self, x):
        """使激活函數可以像函數一樣被調用"""
        return self.forward(x)


class Sigmoid(ActivationFunction):
    """
    Sigmoid 激活函數
    
    公式: g(x) = 1 / (1 + e^(-x))
    
    特性:
    - 輸出範圍: (0, 1)
    - 平滑可導
    - 容易梯度消失 (當 |x| 很大時)
    
    適用場景:
    - 二分類問題的輸出層
    - 需要輸出機率值的情況
    """
    
    def __init__(self):
        self.name = 'sigmoid'
    
    def forward(self, x):
        """
        Sigmoid 前向傳播
        
        使用數值穩定的實作方式避免溢位
        """
        # 數值穩定版本:避免 exp 溢位
        # 對於正數: sigmoid(x) = 1 / (1 + exp(-x))
        # 對於負數: sigmoid(x) = exp(x) / (1 + exp(x))
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def derivative(self, x):
        """
        Sigmoid 導數
        
        公式: g'(x) = g(x) * (1 - g(x))
        
        注意: 輸入 x 是激活前的值
        """
        fx = self.forward(x)
        return fx * (1 - fx)


class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent (雙曲正切) 激活函數
    
    公式: g(x) = (e^x - e^(-x)) / (e^x + e^(-x))
         或 g(x) = (1 - e^(-2x)) / (1 + e^(-2x))
    
    特性:
    - 輸出範圍: (-1, 1)
    - 零中心化 (zero-centered)
    - 比 Sigmoid 收斂更快
    - 仍可能梯度消失
    
    適用場景:
    - 隱藏層 (比 Sigmoid 更常用)
    - 需要負值輸出的情況
    """
    
    def __init__(self):
        self.name = 'tanh'
    
    def forward(self, x):
        """
        Tanh 前向傳播
        
        使用 numpy 內建的 tanh 函數 (已優化)
        """
        return np.tanh(x)
    
    def derivative(self, x):
        """
        Tanh 導數
        
        公式: g'(x) = 1 - g(x)^2
        
        注意: 輸入 x 是激活前的值
        """
        fx = self.forward(x)
        return 1 - fx ** 2


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (修正線性單元) 激活函數
    
    公式: g(x) = max(0, x)
    
    特性:
    - 輸出範圍: [0, +∞)
    - 計算簡單高效
    - 緩解梯度消失問題
    - 可能出現 "神經元死亡" (dying ReLU)
    
    適用場景:
    - 現代深度網路的隱藏層首選
    - 需要避免梯度消失的情況
    """
    
    def __init__(self):
        self.name = 'relu'
    
    def forward(self, x):
        """
        ReLU 前向傳播
        
        簡單的 max(0, x) 操作
        """
        return np.maximum(0, x)
    
    def derivative(self, x):
        """
        ReLU 導數
        
        公式: g'(x) = 1 if x > 0 else 0
        
        注意: 在 x=0 處不可導,但實作上通常設為 0 或 1
        """
        return (x > 0).astype(float)


class Softmax(ActivationFunction):
    """
    Softmax 激活函數
    
    公式: g(x_i) = e^(x_i) / Σ(e^(x_j))
    
    特性:
    - 輸出範圍: (0, 1)
    - 所有輸出和為 1 (機率分布)
    - 用於多分類問題
    
    適用場景:
    - 多分類問題的輸出層
    - 需要輸出機率分布的情況
    """
    
    def __init__(self):
        self.name = 'softmax'
    
    def forward(self, x):
        """
        Softmax 前向傳播
        
        使用數值穩定的實作方式避免溢位
        技巧: 減去最大值不影響結果但能避免 exp 溢位
        """
        # 數值穩定版本: 減去每行的最大值
        # 處理 1D 和 2D 輸入
        if x.ndim == 1:
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x)
        else:
            # 2D: 對每個樣本 (每行) 分別處理
            x_shifted = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def derivative(self, x):
        """
        Softmax 導數
        
        注意: Softmax 的導數是 Jacobian 矩陣
        實際使用時通常與 Cross-Entropy Loss 結合
        導數簡化為: y_pred - y_true
        
        這裡提供單獨的導數計算 (較少直接使用)
        """
        # 對於 Softmax,導數是 Jacobian 矩陣
        # s_i * (δ_ij - s_j), 其中 s = softmax(x)
        s = self.forward(x)
        
        if x.ndim == 1:
            # 1D 情況: 返回 Jacobian 矩陣
            jacobian = np.diag(s) - np.outer(s, s)
            return jacobian
        else:
            # 2D 情況: 每個樣本返回一個 Jacobian
            # 實務上較少使用,通常與 loss 結合
            # 這裡簡化實作
            return s * (1 - s)


def get_activation(activation_name):
    """
    工廠函數: 根據名稱返回對應的激活函數實例
    
    Parameters:
    -----------
    activation_name : str
        激活函數名稱 ('sigmoid', 'tanh', 'relu', 'softmax')
        
    Returns:
    --------
    ActivationFunction
        激活函數實例
        
    Raises:
    -------
    ValueError
        如果激活函數名稱不支援
        
    Examples:
    ---------
    >>> activation = get_activation('relu')
    >>> output = activation.forward(np.array([1, -1, 0]))
    >>> print(output)
    [1. 0. 0.]
    """
    activations = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU,
        'softmax': Softmax
    }
    
    activation_name = activation_name.lower()
    
    if activation_name not in activations:
        raise ValueError(
            f"Unsupported activation function: {activation_name}. "
            f"Supported: {list(activations.keys())}"
        )
    
    return activations[activation_name]()


# ============================================================
# 測試程式碼 (可選)
# ============================================================

if __name__ == "__main__":
    """
    簡單測試各個激活函數
    """
    print("=" * 60)
    print("激活函數模組測試")
    print("=" * 60)
    
    # 測試輸入
    x = np.array([-2, -1, 0, 1, 2])
    
    # 測試 Sigmoid
    print("\n[Sigmoid]")
    sigmoid = Sigmoid()
    output = sigmoid.forward(x)
    derivative = sigmoid.derivative(x)
    print(f"Input:      {x}")
    print(f"Output:     {output}")
    print(f"Derivative: {derivative}")
    
    # 測試 Tanh
    print("\n[Tanh]")
    tanh = Tanh()
    output = tanh.forward(x)
    derivative = tanh.derivative(x)
    print(f"Input:      {x}")
    print(f"Output:     {output}")
    print(f"Derivative: {derivative}")
    
    # 測試 ReLU
    print("\n[ReLU]")
    relu = ReLU()
    output = relu.forward(x)
    derivative = relu.derivative(x)
    print(f"Input:      {x}")
    print(f"Output:     {output}")
    print(f"Derivative: {derivative}")
    
    # 測試 Softmax
    print("\n[Softmax]")
    softmax = Softmax()
    output = softmax.forward(x)
    print(f"Input:      {x}")
    print(f"Output:     {output}")
    print(f"Sum:        {np.sum(output):.6f} (應該為 1)")
    
    # 測試工廠函數
    print("\n[工廠函數測試]")
    act = get_activation('relu')
    print(f"get_activation('relu'): {act.name}")
    
    print("\n" + "=" * 60)
    print("測試完成!")
    print("=" * 60)