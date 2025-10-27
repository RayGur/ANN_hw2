"""
資料處理模組

提供資料載入、預處理、分割等功能,包括:
1. 載入資料集
2. 資料標準化 (使用 Training set 的統計量)
3. K-fold 交叉驗證分割 (for Iris/Wine)
4. 簡單分割 (for Breast Cancer)
5. 資料打亂 (Shuffling)

設計原則:
- 避免 Data Leakage: 標準化只使用 Training set 的參數
- 支援可重現性: 所有隨機操作都支援 random_seed
"""

import numpy as np


class DataProcessor:
    """
    資料處理類別
    
    提供靜態方法處理資料載入、預處理和分割
    """
    
    @staticmethod
    def load_data(filepath, delimiter=None):
        """
        載入資料集
        
        Parameters:
        -----------
        filepath : str
            資料檔案路徑
        delimiter : str, optional
            分隔符號,如果為 None 則自動偵測 (空白或 tab)
            
        Returns:
        --------
        tuple
            (X, y) - 特徵矩陣和標籤向量
            
        Examples:
        ---------
        >>> X, y = DataProcessor.load_data('data/iris.txt')
        >>> print(X.shape, y.shape)
        (150, 4) (150,)
        """
        try:
            # 載入資料
            data = np.loadtxt(filepath, delimiter=delimiter)
            
            # 分離特徵和標籤
            X = data[:, :-1]  # 所有列,除了最後一欄
            y = data[:, -1]   # 最後一欄
            
            # 將標籤轉換為整數 (分類標籤)
            y = y.astype(int)
            
            print(f"[Data Loaded] {filepath}")
            print(f"  - Samples: {X.shape[0]}")
            print(f"  - Features: {X.shape[1]}")
            print(f"  - Classes: {len(np.unique(y))} {np.unique(y)}")
            
            return X, y
            
        except Exception as e:
            raise IOError(f"Error loading data from {filepath}: {str(e)}")
    
    @staticmethod
    def standardize(X_train, X_val=None, X_test=None):
        """
        標準化資料 (Z-score normalization)
        
        重要: 只使用 Training set 的統計量來標準化所有集合
        這樣可以避免 Data Leakage
        
        公式: X_scaled = (X - mean) / std
        
        Parameters:
        -----------
        X_train : np.ndarray
            訓練集特徵
        X_val : np.ndarray, optional
            驗證集特徵
        X_test : np.ndarray, optional
            測試集特徵
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_val_scaled, X_test_scaled, mean, std)
            如果 X_val 或 X_test 為 None,對應位置返回 None
            
        Examples:
        ---------
        >>> X_train_s, X_val_s, X_test_s, mean, std = DataProcessor.standardize(
        ...     X_train, X_val, X_test
        ... )
        """
        # 計算 Training set 的統計量
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        # 避免除以零 (如果某個特徵的標準差為 0)
        std[std == 0] = 1.0
        
        # 標準化 Training set
        X_train_scaled = (X_train - mean) / std
        
        # 標準化 Validation set (使用 Training 的參數!)
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = (X_val - mean) / std
        
        # 標準化 Test set (使用 Training 的參數!)
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = (X_test - mean) / std
        
        print(f"[Standardization]")
        print(f"  - Mean: {mean}")
        print(f"  - Std:  {std}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, mean, std
    
    @staticmethod
    def shuffle_data(X, y, random_seed=42):
        """
        打亂資料順序
        
        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        y : np.ndarray
            標籤向量
        random_seed : int
            隨機種子 (確保可重現性)
            
        Returns:
        --------
        tuple
            (X_shuffled, y_shuffled)
            
        Examples:
        ---------
        >>> X_shuffled, y_shuffled = DataProcessor.shuffle_data(X, y, random_seed=42)
        """
        np.random.seed(random_seed)
        
        # 生成隨機索引
        indices = np.random.permutation(len(X))
        
        # 根據索引打亂
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        print(f"[Shuffling] Data shuffled with seed={random_seed}")
        
        return X_shuffled, y_shuffled
    
    @staticmethod
    def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, 
                            random_seed=42, shuffle=True):
        """
        簡單分割資料集為 Train / Validation / Test
        
        適用於: Breast Cancer (資料量較大)
        
        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        y : np.ndarray
            標籤向量
        train_ratio : float
            訓練集比例 (default: 0.8)
        val_ratio : float
            驗證集比例 (default: 0.1)
        random_seed : int
            隨機種子
        shuffle : bool
            是否先打亂資料 (default: True)
            
        Returns:
        --------
        tuple
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
            
        Examples:
        ---------
        >>> train, val, test = DataProcessor.train_val_test_split(X, y)
        >>> X_train, y_train = train
        """
        # 先打亂資料 (如果需要)
        if shuffle:
            X, y = DataProcessor.shuffle_data(X, y, random_seed)
        
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # 分割
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        print(f"[Train/Val/Test Split]")
        print(f"  - Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        print(f"  - Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
        print(f"  - Test:  {len(X_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    @staticmethod
    def create_kfold_splits(X, y, k=10, random_seed=42, shuffle=True):
        """
        創建 K-fold 交叉驗證分割
        
        適用於: Iris, Wine (資料量較小)
        
        策略:
        - 將資料分成 K 份
        - 每次用 1 份作為 test, 1 份作為 validation, 其餘作為 train
        - 實際上是 K-fold,但我們取出一份當 validation
        
        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        y : np.ndarray
            標籤向量
        k : int
            fold 數量 (default: 10)
        random_seed : int
            隨機種子
        shuffle : bool
            是否先打亂資料
            
        Returns:
        --------
        list
            包含 k 個 tuple 的列表
            每個 tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
            
        Examples:
        ---------
        >>> folds = DataProcessor.create_kfold_splits(X, y, k=10)
        >>> for i, (train, val, test) in enumerate(folds):
        ...     X_train, y_train = train
        ...     print(f"Fold {i+1}: Train={len(X_train)}, Val={len(val[0])}, Test={len(test[0])}")
        """
        # 先打亂資料
        if shuffle:
            X, y = DataProcessor.shuffle_data(X, y, random_seed)
        
        n_samples = len(X)
        fold_size = n_samples // k
        
        folds = []
        
        for i in range(k):
            # 計算當前 fold 的測試集索引
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < k - 1 else n_samples
            
            # 計算驗證集索引 (使用下一個 fold)
            val_fold_idx = (i + 1) % k
            val_start = val_fold_idx * fold_size
            val_end = (val_fold_idx + 1) * fold_size if val_fold_idx < k - 1 else n_samples
            
            # 創建索引
            test_indices = list(range(test_start, test_end))
            val_indices = list(range(val_start, val_end))
            
            # 訓練集 = 剩下的所有數據
            train_indices = [idx for idx in range(n_samples) 
                           if idx not in test_indices and idx not in val_indices]
            
            # 分割數據
            X_train = X[train_indices]
            y_train = y[train_indices]
            
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            X_test = X[test_indices]
            y_test = y[test_indices]
            
            folds.append(((X_train, y_train), (X_val, y_val), (X_test, y_test)))
        
        print(f"[K-Fold Split] Created {k} folds")
        print(f"  - Each fold: ~{fold_size} test samples, ~{fold_size} val samples")
        
        return folds
    
    @staticmethod
    def prepare_classification_labels(y, num_classes=None):
        """
        準備分類標籤
        
        將標籤轉換為從 0 開始的連續整數
        (某些資料集的標籤可能從 1 開始)
        
        Parameters:
        -----------
        y : np.ndarray
            原始標籤
        num_classes : int, optional
            類別數量,如果為 None 則自動檢測
            
        Returns:
        --------
        tuple
            (y_processed, num_classes, label_mapping)
            
        Examples:
        ---------
        >>> y = np.array([1, 2, 3, 1, 2])  # 標籤從 1 開始
        >>> y_new, n_classes, mapping = DataProcessor.prepare_classification_labels(y)
        >>> print(y_new)  # [0, 1, 2, 0, 1] - 轉換為從 0 開始
        """
        # 獲取唯一標籤
        unique_labels = np.unique(y)
        
        # 如果標籤已經是 0, 1, 2, ...,則不需要轉換
        if np.array_equal(unique_labels, np.arange(len(unique_labels))):
            if num_classes is None:
                num_classes = len(unique_labels)
            return y, num_classes, None
        
        # 創建映射: 原始標籤 -> 0, 1, 2, ...
        label_mapping = {old_label: new_label 
                        for new_label, old_label in enumerate(unique_labels)}
        
        # 轉換標籤
        y_processed = np.array([label_mapping[label] for label in y])
        
        if num_classes is None:
            num_classes = len(unique_labels)
        
        print(f"[Label Processing]")
        print(f"  - Original labels: {unique_labels}")
        print(f"  - Mapped to: {np.arange(len(unique_labels))}")
        print(f"  - Mapping: {label_mapping}")
        
        return y_processed, num_classes, label_mapping


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試資料處理模組的各項功能
    """
    print("=" * 60)
    print("資料處理模組測試")
    print("=" * 60)
    
    # 創建模擬資料
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 類別
    
    print("\n[1] 測試資料打亂")
    print("-" * 60)
    X_shuffled, y_shuffled = DataProcessor.shuffle_data(X, y)
    print(f"Original first 5 labels: {y[:5]}")
    print(f"Shuffled first 5 labels: {y_shuffled[:5]}")
    
    print("\n[2] 測試簡單分割 (Train/Val/Test)")
    print("-" * 60)
    train, val, test = DataProcessor.train_val_test_split(X, y)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print("\n[3] 測試標準化")
    print("-" * 60)
    X_train_s, X_val_s, X_test_s, mean, std = DataProcessor.standardize(
        X_train, X_val, X_test
    )
    print(f"Train mean before: {X_train.mean(axis=0)}")
    print(f"Train mean after:  {X_train_s.mean(axis=0)}")
    print(f"Train std after:   {X_train_s.std(axis=0)}")
    
    print("\n[4] 測試 K-fold 分割")
    print("-" * 60)
    folds = DataProcessor.create_kfold_splits(X, y, k=10)
    print(f"Total folds: {len(folds)}")
    for i, (train, val, test) in enumerate(folds[:3]):  # 只顯示前 3 個
        print(f"  Fold {i+1}: Train={len(train[0])}, Val={len(val[0])}, Test={len(test[0])}")
    
    print("\n[5] 測試標籤處理")
    print("-" * 60)
    y_with_offset = np.array([1, 2, 3, 1, 2, 3, 1])  # 標籤從 1 開始
    y_processed, n_classes, mapping = DataProcessor.prepare_classification_labels(
        y_with_offset
    )
    print(f"Processed labels: {y_processed}")
    
    print("\n" + "=" * 60)
    print("測試完成!")
    print("=" * 60)